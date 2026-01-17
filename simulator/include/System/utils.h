#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <memory>
#include "bf16/bf16_basic_ops.h"


namespace System {

// Helper to generate random float vector
inline std::vector<float> generate_random_vector(size_t size, float min_val = -0.5f, float max_val = 0.5f) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
    return vec;
}

// Helper to generate random int4 vector (stored as int8_t for convenience, range [-8, 7])
inline std::vector<uint16_t> generate_random_int4_vector(size_t size) {
    std::vector<uint16_t> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-8, 7);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = static_cast<uint16_t>(dis(gen));
    }
    return vec;
}


// Helper to generate random int4 Q and K matrices
inline void generate_random_int4_qk(size_t num_heads, size_t seq_len, size_t head_dim,
                                   std::vector<std::vector<uint16_t>>& q,
                                   std::vector<std::vector<std::vector<uint16_t>>>& k) {
    q.resize(num_heads);
    k.resize(num_heads);

    for (size_t h = 0; h < num_heads; ++h) {
        q[h] = System::generate_random_int4_vector(head_dim);
        k[h].resize(head_dim);
        for (size_t i = 0; i < head_dim; ++i) {
            k[h][i] = System::generate_random_int4_vector(seq_len);
        }

        // Convert to Simulator Format (uint16_t low 4 bits)
        for (size_t i = 0; i < head_dim; ++i) {
            q[h][i] = static_cast<uint16_t>(q[h][i] & 0xF);
        }
        for (size_t i = 0; i < head_dim; ++i) {
            for (size_t s = 0; s < seq_len; ++s) {
                k[h][i][s] = static_cast<uint16_t>(k[h][i][s] & 0xF);
            }
        }
    }
}

inline void generate_random_bf16_qkv(size_t num_heads, size_t seq_len, size_t head_dim,
                                   std::vector<std::vector<uint16_t>>& q,
                                   std::vector<std::vector<std::vector<uint16_t>>>& k,
                                   std::vector<std::vector<std::vector<uint16_t>>>& v) {
    q.resize(num_heads);
    k.resize(num_heads);
    v.resize(num_heads);

    for (size_t h = 0; h < num_heads; ++h) {
        // Generate Q
        std::vector<float> q_float = System::generate_random_vector(head_dim);
        q[h].resize(head_dim);
        for (size_t i = 0; i < head_dim; ++i) {
            q[h][i] = bf16::float_to_bf16(q_float[i]);
        }

        // Generate K
        k[h].resize(head_dim);
        for (size_t i = 0; i < head_dim; ++i) {
            std::vector<float> k_float = System::generate_random_vector(seq_len);
            k[h][i].resize(seq_len);
            for (size_t s = 0; s < seq_len; ++s) {
                k[h][i][s] = bf16::float_to_bf16(k_float[s]);
            }
        }

        // Generate V
        v[h].resize(seq_len);
        for (size_t s = 0; s < seq_len; ++s) {
            std::vector<float> v_float = System::generate_random_vector(head_dim);
            v[h][s].resize(head_dim);
            for (size_t i = 0; i < head_dim; ++i) {
                v[h][s][i] = bf16::float_to_bf16(v_float[i]);
            }
        }
    }
}

inline void generate_random_bf16_score(size_t num_heads, size_t seq_len,
                                   std::vector<std::vector<uint16_t>>& score) {
    score.resize(num_heads);
    for (size_t h = 0; h < num_heads; ++h) {
        std::vector<float> score_float = System::generate_random_vector(seq_len, -8.0f, 8.0f);
        score[h].resize(seq_len);
        for (size_t s = 0; s < seq_len; ++s) {
            score[h][s] = bf16::float_to_bf16(score_float[s]);
        }
    }
}

} // namespace System