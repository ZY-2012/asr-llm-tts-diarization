#ifndef EXAMPLES_CONFIGUI_APM_CONFIG_H_
#define EXAMPLES_CONFIGUI_APM_CONFIG_H_

#include <string>

namespace apm_example {

struct AppConfig {
  struct Audio {
    std::string playback_device = "hw:1,3";
    int playback_sample_rate_hz = 48000;
    int playback_channels = 2;
    std::string capture_device = "hw:1,2";
    int capture_sample_rate_hz = 48000;
    int capture_channels = 2;
  } audio;

  struct Buffers {
    int period_ms = 10;
    int alsa_buffer_ms = 40;
    int application_buffer_ms = 1000;
  } buffers;

  struct Apm {
    struct Aec {
      bool enabled = true;
      std::string mode = "aec3";  // "aec3" or "mobile".
      bool multi_channel_render = true;
      bool multi_channel_capture = true;
    } aec;

    struct Delay {
      // Added to the playback and capture ALSA FIFO delays.
      int base_ms = 0;
    } delay;

    struct NoiseSuppression {
      bool enabled = true;
      std::string level = "high";  // low, moderate, high, very_high.
    } ns;

    struct GainController1 {
      bool enabled = false;
      std::string mode = "fixed_digital";
      int target_level_dbfs = 3;
      int compression_gain_db = 9;
      bool limiter = true;
    } agc1;

    struct GainController2 {
      bool enabled = true;
      float fixed_gain_db = 0.0f;
      bool adaptive_digital = false;
      float adaptive_headroom_db = 5.0f;
      float adaptive_max_gain_db = 50.0f;
      float adaptive_initial_gain_db = 15.0f;
      float adaptive_max_gain_change_db_per_second = 6.0f;
      float adaptive_max_output_noise_level_dbfs = -50.0f;
    } agc2;

    struct CaptureLevelAdjustment {
      bool enabled = true;
      // Linear amplitude factors. The UI displays and edits these in dB.
      float pre_gain_factor = 1.0f;
      float post_gain_factor = 1.0f;
    } gain_adjustment;

    bool high_pass_filter_enabled = true;
  } apm;

  // Post-APM AX650 chain.  Keeping these fields in AppConfig lets the C++
  // ALSA backend and the Python orchestration page consume one YAML document.
  struct FastEnhance {
    bool enabled = true;
    std::string model_path = "models-for-asr-chain/ns/fastenhance/fastenhance_48k.axmodel";
    std::string library;
    int npu_core = 0;
    bool monitor_enabled = false;
  } fastenhance;

  struct Vad {
    bool enabled = true;
    std::string model_path = "models-for-asr-chain/vad/ten-vad/ten-vad.axmodel";
    std::string library;
    int npu_core = 0;
    // Python TEN-VAD worker input buffering; accepted here because this
    // shared config is also parsed by run-alsa-apm.
    int input_buffer_ms = 1000;
    int hop_ms = 16;
    float threshold = 0.3f;
    int min_speech_ms = 100;
    // A segment starts checking trailing silence only after this much audio.
    float min_segment_seconds = 3.0f;
    // Safety fallback: force a save when a segment grows this long.
    float max_segment_seconds = 30.0f;
    float trailing_silence_seconds = 3.0f;
    // Streaming first-word fast path (0 = off).  When > 0, the accumulator
    // emits a short "preview" segment as soon as this much voiced audio has
    // accumulated, so the ASR can produce the opening text at ~preview_seconds
    // without waiting for the speaker label (which lands ~10 s later, after
    // the diarization window commits).  The preview is text-only (no speaker);
    // the normal segment flow still attaches the speaker later.
    float preview_seconds = 0.0f;
    // Long-monologue splitting (0 = off).  Once a segment has accumulated this
    // much voiced audio, any silence of >= 0.2 s closes it immediately instead
    // of waiting for the full trailing_silence; max_segment_seconds stays the
    // hard cap.  Splits then land on natural pauses (no mid-word cuts for the
    // ASR) and long monologues stream to the ASR piece by piece.
    float pause_split_seconds = 0.0f;
    // Fixed-cadence streaming text (streaming-slicer).  When tick > 0 the
    // slicer transcribes a pause-delimited "epoch" of enhanced audio every
    // tick seconds on a fixed beat (independent of VAD segment closure) and
    // confirms text via LocalAgreement-2 across consecutive hypotheses, so
    // fluent monologues keep producing text instead of stalling until the
    // speaker pauses.  pause = epoch-closing silence length; epoch_max = hard
    // re-anchor for nonstop monologues.  Events: stext (tick) / sfinal
    // (epoch close).  The old preview path should stay off when this is on
    // (duplicate fast text otherwise).
    float stream_tick_seconds = 0.0f;
    float stream_pause_seconds = 0.2f;
    float stream_epoch_max_seconds = 12.0f;
  } vad;

  struct Diarization {
    bool enabled = false;
    bool logging_enabled = false;
    // "segment": run the resident worker per VAD segment (lowest latency).
    // "session": for file inputs, diarize the whole source once at startup and
    // map the resulting speaker timeline onto VAD segments (best quality).
    // "online": like "session" transport-wise (streaming chunks for realtime
    // inputs, whole file for batch), but the worker assigns cluster ids with
    // diart-style online clustering: joint Hungarian matching, speech-gated
    // speaker creation and centroid updates, and a speaker-count cap.
    std::string mode = "segment";
    // Resident pyannote community-1 worker; the model directory must contain
    // the eight axmodels plus fbank/frontend/ffn/powerset/embtail/plda bin
    // host assets.
    std::string runner = "build/diar-worker";
    std::string model_path = "models-for-asr-chain/pyannote";
    std::string speaker_table_path = "/tmp/ax-audio-sdk/speakers.json";
    // Cosine threshold for matching 256-D community-1 centroids against the
    // persistent speaker table.
    float speaker_match_threshold = 0.55f;
    // Segmentation sliding stride; the window duration is 10 s.
    float step_sec = 2.0f;
    // OpenMP threads used by the host-side FP32 stages.
    int omp_threads = 4;
    // Streaming session mode: chunk length sent to the resident worker for
    // realtime inputs (ALSA, or file input in realtime emulation mode).
    float session_chunk_sec = 120.0f;
    // Streaming session mode: overlapping-window commit step (diart-style
    // decoupling of clustering context from commit granularity).  0 keeps
    // the legacy behaviour (one commit per full chunk).  With 0 < step <
    // chunk the worker clusters a `session_chunk_sec` window ending at every
    // step boundary but only commits the newest step-sized core region, so
    // label latency drops to ~step + inference while the clustering context
    // stays a full chunk long.
    float session_step_sec = 0.0f;
    // Cold-start ramp-up (0 = off, first commit waits a full chunk).  With
    // 0 < session_first_window_sec < session_chunk_sec the very first commit
    // clusters only that much audio, so the session's opening labels appear
    // after ~first_window + inference instead of a whole chunk; every later
    // commit keeps the full `session_chunk_sec` context, which is why only
    // the opening region is clustered from a short window.
    float session_first_window_sec = 0.0f;
    // How long a segment may wait for the session timeline to cover it before
    // falling back to per-segment diarization (0 = never wait).
    float session_max_wait_sec = 180.0f;
    // "online" mode only: seconds of speech a cluster must carry before it
    // may create a new speaker id (rho_update / delta_new gate; stops "um"/
    // "right" backchannel fragments from spawning phantom speakers).
    float online_min_new_speech_sec = 2.0f;
    // "online" mode only: seconds of speech a matched cluster needs before it
    // updates the speaker's running centroid.
    float online_update_min_sec = 1.0f;
    // "online" mode only: hard cap on global speakers in the table.
    int online_max_speakers = 20;
    // diart-style incremental clustering (streaming session/online mode).
    // Instead of buffering a full `session_chunk_sec` window and running
    // batch AHC+PLDA+VBx, each 10 s sub-window is submitted to the resident
    // worker one at a time and merged into a running global speaker set, so
    // the first label emits from sub-window 0 (~10 s) instead of ~132 s.
    // Use with session_chunk_sec == session_step_sec == 10 (the engine's
    // fixed 10 s analysis window).
    bool incremental = false;
    // Match on the PLDA discriminant projection (256 -> 128) instead of raw
    // 256-D cosine.  Recommended for 3+ speakers (delta_new 0.55 works across
    // scenes); raw cosine is better for the 2-speaker case.
    bool incremental_plda = false;
    // New-speaker cosine-distance gate: a local speaker becomes a new global
    // speaker only when its nearest running centroid is farther than this.
    float incremental_delta_new = 0.5f;
    // Centroid-update / new-speaker gate: a local speaker must be active for
    // at least this fraction of a sub-window to create or update a centroid.
    float incremental_rho_update = 0.4f;
  } diarization;

  struct Asr {
    bool enabled = true;
    std::string model_path = "models-for-asr-chain/asr/sensevoice";
    std::string model_type = "sensevoice";
    std::string language = "zh";
    std::string library;
    int npu_core = 1;
    int worker_threads = 1;
  } asr;

  struct Llm {
    bool enabled = false;
    std::string model_path = "models-for-asr-chain/llm/qwen3-1.7b";
    std::string runner = "build/qwen3-worker";
    std::string system_prompt = "你是一个乐于助人的助手。";
    bool translation_mode = false;
    std::string zh_to_en_prompt = "translate the following text into English：";
    std::string en_to_zh_prompt = "以下内容翻译成中文：";
    bool dynamic_load = false;
    int dynamic_load_pool_size = 2;
    bool memory_guard = true;
    int memory_guard_floor_mb = 128;
    int max_tokens = 128;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    std::string thinking_mode = "disabled";
    bool reset_context = true;
    bool stream_tokens = true;
    int queue_capacity = 0;
    std::string output_file = "/tmp/ax-audio-sdk/llm.txt";
    std::string npu_lock_file = "/tmp/ax-audio-sdk-npu2.lock";
  } llm;

  struct HojoTts {
    bool enabled = false;
    std::string runner = "build/hojo-tts-resident";
    std::string model_path = "models-for-asr-chain/tts/hojo/models";
    // Hojo's model package fixes English voices to 0..12 and Chinese voices
    // to 13..14. Keep separate selections so language routing is unambiguous.
    int english_voice = 9;
    int chinese_voice = 13;
    int max_new_tokens = 1600;
    int queue_capacity = 0;
    bool keep_wav = false;
    std::string output_dir = "/tmp/ax-audio-sdk/hojo-output";
    float playback_gain = 1.0f;
    int npu_core = 1;
  } hojo_tts;

  struct Output {
    std::string temp_dir = "/tmp/ax-audio-sdk/segments";
    std::string text_file = "/tmp/ax-audio-sdk/transcript.txt";
    std::string mp3_dir = "/tmp/ax-audio-sdk/mp3";
    int max_mp3_storage_mb = 512;
    std::string mp3_encoder = "ffmpeg";
  } output;

  struct Runtime {
    std::string input = "alsa";
    std::string input_wav;
    // File input is either paced to the WAV duration or consumed as quickly
    // as downstream processing permits.
    std::string file_input_mode = "realtime";  // "realtime" or "batch".
    std::string apm_binary = "./run-alsa-apm";
    std::string log_level = "info";
    bool playback_enabled = true;
  } runtime;
};

// Loads the complete shared mapping used by audio-pipeline. No external YAML
// library is required on the target. Returns false and fills error on failure.
bool LoadAppConfig(const std::string& path,
                   AppConfig* config,
                   std::string* error);

// Loads only fields consumed by the ALSA/WebRTC APM process. Other sections
// in the shared sdk-config.yaml are intentionally ignored.
bool LoadApmConfig(const std::string& path,
                   AppConfig* config,
                   std::string* error);

bool ValidateAppConfig(const AppConfig& config, std::string* error);
bool ValidateApmConfig(const AppConfig& config, std::string* error);

}  // namespace apm_example

#endif  // EXAMPLES_CONFIGUI_APM_CONFIG_H_
