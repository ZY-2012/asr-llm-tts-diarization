#!/usr/bin/env python3
"""Live diarization+ASR demo viewer (diart-style), stdlib only.

Runs audio-pipeline in file-realtime mode (simulated microphone), tails the
pipeline's events.jsonl and pushes it to the browser over SSE.  The page plays
the same WAV in sync with the pipeline's stream start, so the gap between the
"now" cursor and the labels is the real end-to-end latency.

NOTE: no authentication -- anyone who can reach the port can start/stop runs.
Intended for a trusted lab LAN only.
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(REPO, "build/der-eval/config-2pw-inc-preview3.yaml")
WORK = "/tmp/rt-demo"
EVENTS = os.path.join(WORK, "ali/events.jsonl")
LOG = os.path.join(WORK, "run.log")
INPUTS = {
    "60": "/tmp/3dspk/2P_60s.16k.wav",
    "full": "/tmp/3dspk/20200327_2P.16k.wav",
}
REF_RTTM = os.path.join(REPO, "build/3dspk/out/label-2P-v8-recheck.rttm")
STEPS = ["10", "2", "1"]
MAXSEGS = ["10", "5"]
PSPLITS = ["0", "4", "3"]

state = {"proc": None, "input": None, "launched": 0.0}
lock = threading.Lock()


def make_config(wav, step, maxseg, psplit):
    text = open(TEMPLATE, encoding="utf-8").read()
    text = text.replace("/tmp/der-2pw-inc-preview3", WORK)
    text = re.sub(r"mp3_dir: .*", "mp3_dir: %s/mp3" % WORK, text)
    text = re.sub(r"input_wav: .*", "input_wav: %s" % wav, text)
    text = re.sub(r"session_step_sec: .*", "session_step_sec: %s" % step, text)
    text = re.sub(r"max_segment_seconds: .*", "max_segment_seconds: %s" % maxseg, text)
    text = re.sub(r"pause_split_seconds: .*", "pause_split_seconds: %s" % psplit, text)
    path = os.path.join(WORK, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def env():
    e = dict(os.environ)
    libs = ["build", "driver/apm/lib", "driver/vad/lib",
            "driver/asr/sensevoice/lib"]
    e["LD_LIBRARY_PATH"] = ":".join([os.path.join(REPO, p) for p in libs] +
                                    ["/soc/lib", "/opt/lib"])
    e["AX_SENSEVOICE_SHARE_DIR"] = os.path.join(REPO, "driver/asr/sensevoice")
    e["OMP_NUM_THREADS"] = "1"
    return e


def running():
    p = state["proc"]
    return p is not None and p.poll() is None


def start(key, step, maxseg, psplit):
    with lock:
        if running():
            return False, "already running"
        wav = INPUTS.get(key)
        if not wav or not os.path.isfile(wav):
            return False, "missing input %s" % wav
        if step not in STEPS or maxseg not in MAXSEGS or psplit not in PSPLITS:
            return False, "bad step/maxseg/psplit"
        subprocess.run(["rm", "-rf", WORK])
        os.makedirs(os.path.join(WORK, "mp3"), exist_ok=True)
        cfg = make_config(wav, step, maxseg, psplit)
        log = open(LOG, "w")
        state["proc"] = subprocess.Popen(
            [os.path.join(REPO, "build/audio-pipeline"), "--config", cfg],
            cwd=REPO, env=env(), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        state["input"] = key
        state["launched"] = time.time()
        return True, "started"


def stop():
    with lock:
        p = state["proc"]
        if p is not None and p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        return True, "stopped"


def load_ref(limit):
    out = []
    if not os.path.isfile(REF_RTTM):
        return out
    for line in open(REF_RTTM, encoding="utf-8"):
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            b, d = float(f[3]), float(f[4])
            if b < limit:
                out.append([b, b + d, f[7]])
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            data = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif u.path == "/audio":
            self.serve_file(INPUTS.get(q.get("input", ["60"])[0]), "audio/wav")
        elif u.path == "/ref":
            self.send_json(load_ref(float(q.get("limit", ["1e9"])[0])))
        elif u.path == "/status":
            self.send_json({"running": running(), "input": state["input"]})
        elif u.path == "/events":
            self.stream_events()
        else:
            self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/start":
            ok, msg = start(q.get("input", ["60"])[0], q.get("step", ["2"])[0],
                            q.get("maxseg", ["5"])[0], q.get("psplit", ["4"])[0])
            self.send_json({"ok": ok, "msg": msg}, 200 if ok else 409)
        elif u.path == "/stop":
            ok, msg = stop()
            self.send_json({"ok": ok, "msg": msg})
        else:
            self.send_error(404)

    def serve_file(self, path, ctype):
        if not path or not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        begin, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m:
            if m.group(1):
                begin = int(m.group(1))
            if m.group(2):
                end = min(int(m.group(2)), size - 1)
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (begin, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - begin + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(begin)
            left = end - begin + 1
            try:
                while left > 0:
                    chunk = f.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        launched = state["launched"]
        pos = 0
        try:
            while True:
                if state["launched"] != launched:
                    return  # new run: client reconnects
                if os.path.isfile(EVENTS):
                    with open(EVENTS, encoding="utf-8") as f:
                        f.seek(pos)
                        while True:
                            line = f.readline()
                            if not line or not line.endswith("\n"):
                                break
                            pos = f.tell()
                            self.wfile.write(("data: %s\n\n" % line.strip()).encode())
                status = {"ev": "status", "running": running(),
                          "since": round(time.time() - launched, 1) if launched else 0}
                self.wfile.write(("data: %s\n\n" % json.dumps(status)).encode())
                self.wfile.flush()
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print("live demo on http://0.0.0.0:%d  (no auth, LAN only)" % port, flush=True)
    try:
        srv.serve_forever()
    finally:
        stop()


if __name__ == "__main__":
    main()
