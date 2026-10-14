// asr-probe -- board-side experiment tool for the streaming-slicer refactor.
//
// Answers three P0 questions against the on-board SenseVoice axmodel:
//   1. stream: does AX_ASR_StreamFeed work (incremental CTC) and how do
//      partial results evolve / what does a chunk cost on the NPU?
//   2. runpcm: AX_ASR_RunPCM in-memory latency vs clip duration (tick budget).
//   3. shift:  transcription stability under window shifts (LocalAgreement
//      fallback viability if streaming is unsupported).
//
// Usage (cwd = repo root on the board):
//   asr-probe stream <wav> [chunk_ms=500]
//   asr-probe runpcm <wav> <start_s> <dur_s>
//   asr-probe shift  <wav> <start_s> <window_s> <shift_s> <n>
#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Wav {
  std::vector<float> pcm;  // mono, native rate
  uint32_t rate = 16000;
};

bool LoadWav(const char* path, Wav* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "open %s failed\n", path); return false; }
  char riff[12];
  f.read(riff, 12);
  if (!f || std::memcmp(riff, "RIFF", 4) || std::memcmp(riff + 8, "WAVE", 4)) {
    std::fprintf(stderr, "not a RIFF/WAVE file\n");
    return false;
  }
  bool have_fmt = false, have_data = false;
  uint16_t channels = 1, bits = 16;
  std::vector<int16_t> raw;
  while (f) {
    char id[4];
    uint32_t size = 0;
    f.read(id, 4);
    f.read(reinterpret_cast<char*>(&size), 4);
    if (!f) break;
    const bool is_fmt = !std::memcmp(id, "fmt ", 4);
    const bool is_data = !std::memcmp(id, "data", 4);
    if (is_fmt) {
      uint16_t fmt = 1;
      uint32_t byte_rate = 0;
      uint16_t block = 0;
      f.read(reinterpret_cast<char*>(&fmt), 2);
      f.read(reinterpret_cast<char*>(&channels), 2);
      f.read(reinterpret_cast<char*>(&out->rate), 4);
      f.read(reinterpret_cast<char*>(&byte_rate), 4);
      f.read(reinterpret_cast<char*>(&block), 2);
      f.read(reinterpret_cast<char*>(&bits), 2);
      if (size > 16) f.seekg(size - 16, std::ios::cur);
      have_fmt = true;
    } else if (is_data) {
      const uint32_t bytes = size ? size : static_cast<uint32_t>(f.eof() ? 0 : 0);
      // size==0 (streaming headers) unsupported; rely on well-formed files.
      const size_t samples = size / 2;
      raw.resize(samples);
      f.read(reinterpret_cast<char*>(raw.data()), size);
      have_data = true;
      break;
    } else {
      f.seekg(size + (size & 1), std::ios::cur);
    }
  }
  if (!have_fmt || !have_data || bits != 16) {
    std::fprintf(stderr, "unsupported wav (fmt=%d data=%d bits=%u)\n",
                 have_fmt, have_data, bits);
    return false;
  }
  const size_t step = channels;
  out->pcm.reserve(raw.size() / step);
  for (size_t i = 0; i + step <= raw.size(); i += step)
    out->pcm.push_back(static_cast<float>(raw[i]) / 32768.0f);
  return true;
}

using InitFn = void* (*)(int, const char*);
using UninitFn = void (*)(void*);
using RunFileFn = int (*)(void*, const char*, const char*, char**);
using RunPCMFn = int (*)(void*, float*, int, int, const char*, char**);
using StreamInitFn = int (*)(void*);
using StreamFeedFn = int (*)(void*, float*, int, int);
using StreamResultFn = int (*)(void*, const char**);
using StreamFinishFn = int (*)(void*);
using StreamResetFn = int (*)(void*);
using FreeFn = void (*)(char*);

struct Api {
  InitFn init = nullptr;
  UninitFn uninit = nullptr;
  RunFileFn run_file = nullptr;
  RunPCMFn run_pcm = nullptr;
  StreamInitFn stream_init = nullptr;
  StreamFeedFn stream_feed = nullptr;
  StreamResultFn stream_result = nullptr;
  StreamFinishFn stream_finish = nullptr;
  StreamResetFn stream_reset = nullptr;
  FreeFn free_result = nullptr;
};

bool LoadApi(const char* library, Api* api) {
  void* lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!lib) { std::fprintf(stderr, "dlopen failed: %s\n", dlerror()); return false; }
  api->init = reinterpret_cast<InitFn>(dlsym(lib, "AX_ASR_Init"));
  api->uninit = reinterpret_cast<UninitFn>(dlsym(lib, "AX_ASR_Uninit"));
  api->run_file = reinterpret_cast<RunFileFn>(dlsym(lib, "AX_ASR_RunFile"));
  api->run_pcm = reinterpret_cast<RunPCMFn>(dlsym(lib, "AX_ASR_RunPCM"));
  api->stream_init = reinterpret_cast<StreamInitFn>(dlsym(lib, "AX_ASR_StreamInit"));
  api->stream_feed = reinterpret_cast<StreamFeedFn>(dlsym(lib, "AX_ASR_StreamFeed"));
  api->stream_result = reinterpret_cast<StreamResultFn>(dlsym(lib, "AX_ASR_StreamResult"));
  api->stream_finish = reinterpret_cast<StreamFinishFn>(dlsym(lib, "AX_ASR_StreamFinish"));
  api->stream_reset = reinterpret_cast<StreamResetFn>(dlsym(lib, "AX_ASR_StreamReset"));
  api->free_result = reinterpret_cast<FreeFn>(dlsym(lib, "AX_ASR_Free"));
  for (void** slot = reinterpret_cast<void**>(api);
       slot < reinterpret_cast<void**>(api) + 1; ++slot) {
    if (!*slot) {
      std::fprintf(stderr, "missing symbol in %s\n", library);
      return false;
    }
  }
  // The loop above only checks the first slot; verify the critical ones.
  if (!api->init || !api->uninit || !api->run_file || !api->run_pcm ||
      !api->stream_init || !api->stream_feed || !api->stream_result) {
    std::fprintf(stderr, "missing required symbols in %s\n", library);
    return false;
  }
  return true;
}

double Ms(std::chrono::steady_clock::time_point t0,
          std::chrono::steady_clock::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: asr-probe stream <wav> [chunk_ms]\n"
                 "       asr-probe runpcm <wav> <start_s> <dur_s>\n"
                 "       asr-probe shift  <wav> <start_s> <window_s> <shift_s> <n>\n");
    return 2;
  }
  const std::string mode = argv[1];
  const char* wav_path = argv[2];
  const char* library = "driver/asr/sensevoice/lib/libax_asr_api.so";
  const char* model_dir = "models-for-asr-chain/asr";
  const char* language = "zh";

  Api api;
  if (!LoadApi(library, &api)) return 1;
  const auto t0 = std::chrono::steady_clock::now();
  void* handle = api.init(4 /*AX_SENSEVOICE*/, model_dir);
  const auto t1 = std::chrono::steady_clock::now();
  if (!handle) { std::fprintf(stderr, "AX_ASR_Init failed\n"); return 1; }
  std::fprintf(stderr, "[probe] model loaded in %.0f ms\n", Ms(t0, t1));

  Wav wav;
  if (!LoadWav(wav_path, &wav)) return 1;
  std::fprintf(stderr, "[probe] %.2f s @ %u Hz\n",
               wav.pcm.size() / static_cast<double>(wav.rate), wav.rate);

  const size_t sr = wav.rate;
  auto pcm_at = [&](double start_s, double dur_s, std::vector<float>* out) {
    const size_t begin = static_cast<size_t>(start_s * sr);
    const size_t end = std::min(wav.pcm.size(),
                                static_cast<size_t>((start_s + dur_s) * sr));
    out->assign(wav.pcm.begin() + std::min(begin, wav.pcm.size()),
                wav.pcm.begin() + std::max(end, begin));
  };

  if (mode == "stream") {
    int chunk_ms = argc > 3 ? std::atoi(argv[3]) : 500;
    if (api.stream_init) {
      const int rc = api.stream_init(handle);
      std::fprintf(stderr, "[probe] StreamInit rc=%d\n", rc);
      if (rc != 0) { api.uninit(handle); return 1; }
    }
    const size_t chunk = static_cast<size_t>(chunk_ms) * sr / 1000;
    double total_feed_ms = 0, max_feed_ms = 0;
    std::string last;
    for (size_t off = 0; off < wav.pcm.size(); off += chunk) {
      const size_t n = std::min(chunk, wav.pcm.size() - off);
      std::fprintf(stderr, "[probe] feeding off=%zu n=%zu\n", off, n);
      const auto s0 = std::chrono::steady_clock::now();
      const int rc = api.stream_feed(handle, wav.pcm.data() + off,
                                      static_cast<int>(n),
                                      static_cast<int>(sr));
      const auto s1 = std::chrono::steady_clock::now();
      std::fprintf(stderr, "[probe] feed done rc=%d\n", rc);
      const double feed_ms = Ms(s0, s1);
      total_feed_ms += feed_ms;
      if (feed_ms > max_feed_ms) max_feed_ms = feed_ms;
      const char* partial = nullptr;
      std::string text;
      if (api.stream_result) {
        const int rrc = api.stream_result(handle, &partial);
        if (rrc == 0 && partial) text = partial;
      }
      const bool changed = text != last;
      std::printf("t=%6.2fs feed=%7.2fms rc=%d%s audio=%.2fs\n%s%s\n",
                  off / static_cast<double>(sr), feed_ms, rc,
                  changed ? " *" : "", changed ? text.c_str() : "");
      last = text;
    }
    if (api.stream_finish) {
      const char* partial = nullptr;
      const int rc = api.stream_finish(handle);
      if (api.stream_result(handle, &partial) == 0 && partial)
        std::printf("[finish rc=%d]\n%s\n", rc, partial);
    }
    const double chunks = static_cast<double>((wav.pcm.size() + chunk - 1) / chunk);
    std::fprintf(stderr, "[probe] feed avg %.2f ms/chunk, max %.2f ms (chunk %d ms)\n",
                 total_feed_ms / std::max(1.0, chunks), max_feed_ms, chunk_ms);
  } else if (mode == "runpcm") {
    if (argc < 5) { std::fprintf(stderr, "runpcm needs start_s dur_s\n"); return 2; }
    const double start = std::atof(argv[3]);
    const double dur = std::atof(argv[4]);
    std::vector<float> pcm;
    pcm_at(start, dur, &pcm);
    char* result = nullptr;
    const auto s0 = std::chrono::steady_clock::now();
    const int rc = api.run_pcm(handle, pcm.data(), static_cast<int>(pcm.size()),
                               static_cast<int>(sr), language, &result);
    const auto s1 = std::chrono::steady_clock::now();
    std::printf("runpcm [%.2f, %.2f) rc=%d %.1f ms\n", start, start + dur, rc, Ms(s0, s1));
    if (result) { std::printf("%s\n", result); api.free_result(result); }
  } else if (mode == "curve") {
    // RunPCM latency vs clip duration: the per-tick budget for the fixed-
    // cadence overlapping-slice design (P1).
    if (argc < 4) { std::fprintf(stderr, "curve needs start_s\n"); return 2; }
    const double start = std::atof(argv[3]);
    for (double dur = 1.0; dur <= 10.0; dur += 1.0) {
      std::vector<float> pcm;
      pcm_at(start, dur, &pcm);
      char* result = nullptr;
      const auto s0 = std::chrono::steady_clock::now();
      const int rc = api.run_pcm(handle, pcm.data(), static_cast<int>(pcm.size()),
                                 static_cast<int>(sr), language, &result);
      const auto s1 = std::chrono::steady_clock::now();
      std::printf("dur=%4.1fs rc=%d %7.1f ms  %s\n",
                  dur, rc, Ms(s0, s1), result ? result : "");
      if (result) api.free_result(result);
    }
  } else if (mode == "shift") {
    if (argc < 7) { std::fprintf(stderr,
        "shift needs start_s window_s shift_s n\n"); return 2; }
    const double start = std::atof(argv[3]);
    const double window = std::atof(argv[4]);
    const double shift = std::atof(argv[5]);
    const int n = std::atoi(argv[6]);
    for (int i = 0; i < n; ++i) {
      const double at = start + i * shift;
      std::vector<float> pcm;
      pcm_at(at, window, &pcm);
      char* result = nullptr;
      const auto s0 = std::chrono::steady_clock::now();
      const int rc = api.run_pcm(handle, pcm.data(), static_cast<int>(pcm.size()),
                                 static_cast<int>(sr), language, &result);
      const auto s1 = std::chrono::steady_clock::now();
      std::printf("shift%+d [%6.2f, %6.2f) rc=%d %6.1f ms  %s\n",
                  i, at, at + window, rc, Ms(s0, s1), result ? result : "");
      if (result) api.free_result(result);
    }
  } else {
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    api.uninit(handle);
    return 2;
  }
  api.uninit(handle);
  return 0;
}
