// community-1 speaker diarization engine (board-side, C++).
//
// Vendored from https://github.com/ZY-2012/pyannote-speaker-diarization.axera
// (MIT License, see ../LICENSE).  Local changes for the ASR/LLM/TTS chain:
//   - Run() can return per-cluster centroids (256-D raw embeddings) so the
//     worker can keep a persistent speaker table across segments, and
//   - Run() can return non-overlapping sample ranges ("parts") for ASR
//     slicing instead of only RTTM segments.
//
// Mirrors the Python SDK (community1_sdk/pipeline.py) exactly:
//   windows: 10 s duration, `step_sec` stride
//   sincnet: host im2col -> seg_a (MatMul+Abs+MaxPool) -> host inst-norm+leaky
//            -> seg_b1 -> host norm+leaky -> seg_b2 -> host norm+leaky
//   LSTM: 4 layers, 64-step NPU cells, FP32 state carry, fwd+bwd per call
//   FFN + powerset decoding on host
//   embeddings: fbank (host) -> emb conv stack (NPU) -> weighted TSTP + seg_1 (host)
//   clustering: filter -> AHC(centroid, 0.6) -> PLDA -> VBx(Fa .07 Fb .8)
//               -> constrained assignment -> reconstruct -> RTTM
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "EngineWrapper.hpp"
#include "fbank.hpp"
#include "params.hpp"
#include "wav_reader.hpp"

namespace c1 {

constexpr int SAMPLE_RATE = 16000;
constexpr int WINDOW_SAMPLES = 160000;
constexpr int SEG_FRAMES = 589;
constexpr float FRAME_STEP = 0.016875f;
constexpr float FRAME_DUR = 0.0619375f;
constexpr int HID = 128;
constexpr int CELL_CHUNK = 96;
constexpr int EMB_DIM = 256;
constexpr int MIN_NUM_FRAMES = 2;        // ceil(589 * 400 / 160000)
constexpr float MIN_ACTIVE_RATIO = 0.2f;

// Shortest ASR part kept after frame-level speaker reconstruction.  Shorter
// runs are merged into the preceding part so segmentation flicker cannot turn
// one utterance into a burst of tiny SenseVoice calls.
constexpr std::size_t MIN_PART_SAMPLES = 6400;   // 0.4 s @ 16 kHz

static inline int closest_frame(float t) {
    return (int)std::lrint((t - 0.5f * FRAME_DUR) / FRAME_STEP);
}

// ---------- small helpers ----------
static inline float leaky(float x) { return x > 0 ? x : 0.01f * x; }

// instance norm over (1, C, T) laid out as C rows of T
static inline void inst_norm(const float* x, int C, int T,
                             const std::vector<float>& w, const std::vector<float>& b,
                             float* y) {
    for (int c = 0; c < C; ++c) {
        const float* row = x + c * T;
        double mean = 0;
        for (int t = 0; t < T; ++t) mean += row[t];
        mean /= T;
        double var = 0;
        for (int t = 0; t < T; ++t) { double d = row[t] - mean; var += d * d; }
        var = var / T + 1e-5;
        double sd = std::sqrt(var);
        for (int t = 0; t < T; ++t)
            y[c * T + t] = (float)((row[t] - mean) / sd * w[c] + b[c]);
    }
}

static inline void l2norm_rows(std::vector<float>& x, int rows, int dim) {
    for (int i = 0; i < rows; ++i) {
        float* r = x.data() + i * dim;
        double n = 0;
        for (int d = 0; d < dim; ++d) n += (double)r[d] * r[d];
        n = std::sqrt(n) + 1e-30;
        for (int d = 0; d < dim; ++d) r[d] = (float)(r[d] / n);
    }
}

struct Segment { float start, dur; int spk; };

// Contiguous sample range assigned to one local cluster.  begin/end are
// indices into the analysed WAV; the ranges form a partition of [0, N).
struct Part { std::size_t begin, end; int cluster; };

class DiarEngine {
 public:
    bool Init(const std::string& model_dir) {
        P.Load(model_dir);
        auto init = [&](EngineWrapper& w, const std::string& f) {
            return w.Init((model_dir + "/" + f).c_str()) == 0;
        };
        if (!init(seg_a, "segmentation_sincnet.axmodel")) return false;
        if (!init(seg_b1, "segmentation_convs1.axmodel")) return false;
        if (!init(seg_b2, "segmentation_convs2.axmodel")) return false;
        for (int l = 0; l < 4; ++l)
            if (!init(cells[l], "lstm_cell_l" + std::to_string(l) + ".axmodel")) return false;
        if (!init(emb, "embedding_convs.axmodel")) return false;
        return true;
    }

    // Runs the whole-file pipeline.  `rttm` receives the raw speaker segments
    // (seconds) as before.  Optional outputs:
    //   parts       - non-overlapping sample ranges covering [0, samples)
    //   centroids   - num_clusters * EMB_DIM raw embedding centroids
    //   num_clusters- number of kept clusters (centroid count)
    void Run(const WavData& wav, float step_sec,
             std::vector<Segment>& rttm,
             std::vector<Part>* parts = nullptr,
             std::vector<float>* centroids = nullptr,
             int* num_clusters = nullptr);

 private:
    Params P;
    EngineWrapper seg_a, seg_b1, seg_b2, cells[4], emb;
    float step_sec_ = 2.0f;

    // scratch buffers (reused across windows)
    std::vector<float> buf_unfold;    // (15975, 251)
    std::vector<float> buf_a;         // (80, 5325)
    std::vector<float> buf_b1;        // (60, 1773)
    std::vector<float> buf_b2;        // (60, 589)
    std::vector<float> buf_lstm;      // (589, 256) per layer
    std::vector<float> buf_cell_x;    // (64, 2*in) max (64, 512)
    std::vector<float> buf_fbank;     // (998, 80)

    void SegmentWindow(const float* win, uint8_t* masks /*589*3*/);
    void Embeddings(const WavData& wav, const std::vector<uint8_t>& seg_data, int C,
                    std::vector<float>& embs /*C*3*256*/);
    void Cluster(std::vector<float>& embs, const std::vector<uint8_t>& seg_data, int C,
                 std::vector<int>& hard_clusters, int& K,
                 std::vector<float>* centroids_out = nullptr);
};

}  // namespace c1
