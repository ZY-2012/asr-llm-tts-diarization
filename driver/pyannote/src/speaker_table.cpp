#include "speaker_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;

namespace diar {
namespace {

float Cosine(const std::vector<float>& a, const float* b, int dim) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (int i = 0; i < dim; ++i) {
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return (float)(dot / (std::sqrt(aa * bb) + 1e-12));
}

}  // namespace

void SpeakerTable::Load(const std::string& path) {
    speakers_.clear();
    next_id_ = 1;
    std::ifstream in(path);
    if (!in) return;
    std::string line;
    while (std::getline(in, line)) {
        const auto p = line.find("\"speaker_id\"");
        if (p == std::string::npos) continue;
        const auto q = line.find('"', line.find(':', p) + 1);
        const auto r = line.find('"', q + 1);
        if (q == std::string::npos || r == std::string::npos) continue;
        Speaker s;
        s.id = line.substr(q + 1, r - q - 1);
        const auto c = line.find("\"sample_count\"");
        if (c != std::string::npos)
            s.count = std::atoll(line.c_str() + line.find(':', c) + 1);
        const auto seen = line.find("\"last_seen_time\"");
        if (seen != std::string::npos)
            s.last_seen = std::atoll(line.c_str() + line.find(':', seen) + 1);
        const auto a = line.find('[', line.find("\"embedding_center\""));
        const auto b = line.find(']', a);
        if (a != std::string::npos && b != std::string::npos) {
            std::stringstream values(line.substr(a + 1, b - a - 1));
            std::string token;
            while (std::getline(values, token, ',')) {
                try { s.center.push_back(std::stof(token)); } catch (...) {}
            }
        }
        speakers_.push_back(std::move(s));
    }
    for (auto& s : speakers_) {
        if (!s.center.empty()) s.samples.push_back(s.center);
    }
    for (const auto& s : speakers_) {
        const auto p = s.id.find_last_of('-');
        if (p != std::string::npos)
            next_id_ = std::max(next_id_, std::atoi(s.id.c_str() + p + 1) + 1);
    }
}

bool SpeakerTable::Save(const std::string& path) const {
    fs::path p(path);
    std::error_code ec;
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path(), ec);
    const fs::path tmp = p.string() + ".tmp";
    std::ofstream out(tmp);
    if (!out) return false;
    out << "{\n  \"speakers\": [\n";
    for (std::size_t i = 0; i < speakers_.size(); ++i) {
        const auto& s = speakers_[i];
        out << "    {\"speaker_id\":\"" << s.id
            << "\",\"sample_count\":" << s.count
            << ",\"last_seen_time\":" << s.last_seen
            << ",\"embedding_center\":[";
        for (std::size_t j = 0; j < s.center.size(); ++j)
            out << (j ? "," : "") << std::setprecision(8) << s.center[j];
        out << "]}" << (i + 1 == speakers_.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    out.close();
    fs::rename(tmp, p, ec);
    if (ec) { fs::remove(tmp, ec); return false; }
    return true;
}

std::vector<std::string> SpeakerTable::Assign(
    const std::vector<float>& centers, int dim,
    const std::vector<long long>& weights, float threshold, long long now) {
    const int clusters = static_cast<int>(weights.size());
    std::vector<std::string> ids(clusters);
    if (clusters <= 0 || dim <= 0 ||
        centers.size() < static_cast<std::size_t>(clusters) * dim)
        return ids;

    std::vector<int> order(clusters);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return weights[a] > weights[b];
    });

    std::vector<char> taken(speakers_.size(), 0);
    constexpr std::size_t kMaxSamples = 8;
    for (int index : order) {
        const float* center = centers.data() + static_cast<std::size_t>(index) * dim;
        int best = -1;
        float best_score = threshold;
        // Match against every recent centroid of each speaker: a single
        // running mean is too brittle for short, noisy segments and causes
        // the same voice to spawn new speaker ids over a long meeting.
        for (std::size_t i = 0; i < speakers_.size(); ++i) {
            if (taken[i]) continue;
            const auto& samples = speakers_[i].samples;
            if (samples.empty()) {
                if (speakers_[i].center.size() != static_cast<std::size_t>(dim)) continue;
                const float score = Cosine(speakers_[i].center, center, dim);
                if (score >= best_score) { best_score = score; best = (int)i; }
                continue;
            }
            for (const auto& sample : samples) {
                if (sample.size() != static_cast<std::size_t>(dim)) continue;
                const float score = Cosine(sample, center, dim);
                if (score >= best_score) { best_score = score; best = (int)i; }
            }
        }
        const long long weight = std::max<long long>(1, weights[index]);
        std::vector<float> observation(center, center + dim);
        if (best < 0) {
            Speaker speaker;
            speaker.id = "speaker-" + std::to_string(next_id_++);
            speaker.center = observation;
            speaker.samples.push_back(observation);
            speaker.count = weight;
            speaker.last_seen = now;
            speakers_.push_back(std::move(speaker));
            best = static_cast<int>(speakers_.size()) - 1;
            taken.resize(speakers_.size(), 0);
            std::fprintf(stderr, "[diar] created new %s weight=%lld (no cosine >= %.2f)\n",
                         speakers_[best].id.c_str(), weight, threshold);
        } else {
            Speaker& speaker = speakers_[best];
            const long long previous = std::max<long long>(1, speaker.count);
            for (int d = 0; d < dim; ++d)
                speaker.center[d] = (speaker.center[d] * (float)previous +
                                     center[d] * (float)weight) /
                                    (float)(previous + weight);
            speaker.samples.push_back(std::move(observation));
            if (speaker.samples.size() > kMaxSamples) speaker.samples.erase(speaker.samples.begin());
            speaker.count = previous + weight;
            std::fprintf(stderr, "[diar] matched %s cosine=%.3f weight=%lld history=%lld\n",
                         speaker.id.c_str(), best_score, weight, previous);
        }
        speakers_[best].last_seen = now;
        taken[best] = 1;
        ids[index] = speakers_[best].id;
    }
    return ids;
}

}  // namespace diar
