#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// =============================================================================
// Config & Structs
// =============================================================================
struct GPT2Config {
    int n_layer, n_head, n_embd, vocab_size, max_seq_len;
};

struct LayerWeights {
    float *ln_1_w, *ln_1_b;
    float *attn_w, *attn_b;          // c_attn: [3C, C] and [3C]
    float *attn_proj_w, *attn_proj_b; // c_proj: [C, C] and [C]
    float *ln_2_w, *ln_2_b;
    float *mlp_fc_w, *mlp_fc_b;      // c_fc: [4C, C] and [4C]
    float *mlp_proj_w, *mlp_proj_b;  // c_proj: [C, 4C] and [C]
};

struct GPT2 {
    GPT2Config config;
    float* params_buf;

    // Pointers into params_buf
    float *wte, *wpe;
    std::vector<LayerWeights> layers;
    float *ln_f_w, *ln_f_b;

    // Scratch buffers
    float *x, *x2, *qkv, *attn_out;
    float *attn_sc, *q_h, *k_h, *v_h, *y_h;
    float *mlp_buf, *logits;
};

// =============================================================================
// Vocab (for decode)
// =============================================================================
struct Vocab {
    int n_vocab;
    std::vector<std::string> tokens; // tokens[id] = raw byte string
};

Vocab load_vocab(const char* path) {
    Vocab v;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fread(&v.n_vocab, sizeof(int), 1, f);
    v.tokens.resize(v.n_vocab);
    for (int i = 0; i < v.n_vocab; i++) {
        int len;
        fread(&len, sizeof(int), 1, f);
        v.tokens[i].resize(len);
        fread(&v.tokens[i][0], 1, len, f);
    }
    fclose(f);
    printf("[Vocab] Loaded %d tokens\n", v.n_vocab);
    return v;
}

std::string decode(const Vocab& v, const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) {
        if (id >= 0 && id < v.n_vocab) out += v.tokens[id];
    }
    return out;
}

// =============================================================================
// Tensor Operations
// =============================================================================

// C[M,N] = A[M,K] @ B^T[K,N]  where B is stored as [N,K]
inline void matmul_bt(float* C, const float* A, const float* B,
                      int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i*K + k] * B[j*K + k];
            C[i*N + j] = s;
        }
}

// C[M,N] = A[M,K] @ B[K,N]
inline void matmul(float* C, const float* A, const float* B,
                   int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i*K + k] * B[k*N + j];
            C[i*N + j] = s;
        }
}

// out = x @ w^T + b  (nn.Linear)
inline void linear(float* out, const float* x, const float* w, const float* b,
                   int T, int in_d, int out_d) {
    matmul_bt(out, x, w, T, in_d, out_d);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < out_d; j++)
            out[t*out_d + j] += b[j];
}

void layer_norm(float* out, const float* x, const float* w, const float* b,
                int T, int C) {
    for (int t = 0; t < T; t++) {
        const float* xp = x + t*C;
        float* op = out + t*C;
        float mean = 0.0f;
        for (int i = 0; i < C; i++) mean += xp[i];
        mean /= C;
        float var = 0.0f;
        for (int i = 0; i < C; i++) { float d = xp[i] - mean; var += d*d; }
        var /= C;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int i = 0; i < C; i++)
            op[i] = (xp[i] - mean) * inv * w[i] + b[i];
    }
}

void softmax(float* x, int size) {
    float mx = x[0];
    for (int i = 1; i < size; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

void gelu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v*v*v)));
    }
}

// =============================================================================
// GPT-2 Forward Pass
// =============================================================================

void attention(GPT2& m, int T, int l) {
    int C = m.config.n_embd;
    int nh = m.config.n_head;
    int D = C / nh;
    LayerWeights& w = m.layers[l];

    layer_norm(m.x2, m.x, w.ln_1_w, w.ln_1_b, T, C);
    linear(m.qkv, m.x2, w.attn_w, w.attn_b, T, C, 3*C);

    for (int h = 0; h < nh; h++) {
        // Extract q, k, v for head h
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++) {
                m.q_h[t*D+d] = m.qkv[t*3*C + h*D + d];
                m.k_h[t*D+d] = m.qkv[t*3*C + C + h*D + d];
                m.v_h[t*D+d] = m.qkv[t*3*C + 2*C + h*D + d];
            }

        // att = q @ k^T / sqrt(D)
        matmul_bt(m.attn_sc, m.q_h, m.k_h, T, D, T);
        float scale = 1.0f / sqrtf((float)D);
        for (int i = 0; i < T*T; i++) m.attn_sc[i] *= scale;

        // Causal mask + softmax
        for (int t = 0; t < T; t++) {
            for (int t2 = t+1; t2 < T; t2++)
                m.attn_sc[t*T + t2] = -1e9f;
            softmax(&m.attn_sc[t*T], T);
        }

        // y = att @ v
        matmul(m.y_h, m.attn_sc, m.v_h, T, T, D);

        // Copy to attn_out
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                m.attn_out[t*C + h*D + d] = m.y_h[t*D + d];
    }

    // Output projection + residual
    linear(m.x2, m.attn_out, w.attn_proj_w, w.attn_proj_b, T, C, C);
    for (int i = 0; i < T*C; i++) m.x[i] += m.x2[i];
}

void mlp_forward(GPT2& m, int T, int l) {
    int C = m.config.n_embd;
    LayerWeights& w = m.layers[l];

    layer_norm(m.x2, m.x, w.ln_2_w, w.ln_2_b, T, C);
    linear(m.mlp_buf, m.x2, w.mlp_fc_w, w.mlp_fc_b, T, C, 4*C);
    gelu(m.mlp_buf, T * 4 * C);
    linear(m.x2, m.mlp_buf, w.mlp_proj_w, w.mlp_proj_b, T, 4*C, C);
    for (int i = 0; i < T*C; i++) m.x[i] += m.x2[i];
}

float* gpt2_forward(GPT2& m, const int* ids, int T) {
    int C = m.config.n_embd;
    int V = m.config.vocab_size;

    // Embeddings: tok + pos
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            m.x[t*C + c] = m.wte[ids[t]*C + c] + m.wpe[t*C + c];

    // Transformer blocks
    for (int l = 0; l < m.config.n_layer; l++) {
        attention(m, T, l);
        mlp_forward(m, T, l);
    }

    // Final LN (only last position) + LM Head
    layer_norm(m.x2, &m.x[(T-1)*C], m.ln_f_w, m.ln_f_b, 1, C);
    matmul_bt(m.logits, m.x2, m.wte, 1, C, V);

    return m.logits;
}

// =============================================================================
// Sampling
// =============================================================================
int sample_argmax(const float* logits, int V) {
    int best = 0;
    for (int i = 1; i < V; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

int sample_temp(float* logits, int V, float temp) {
    if (temp <= 0.0f) return sample_argmax(logits, V);
    for (int i = 0; i < V; i++) logits[i] /= temp;
    softmax(logits, V);
    // Random sampling
    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < V; i++) {
        cum += logits[i];
        if (cum >= r) return i;
    }
    return V - 1;
}

// =============================================================================
// Load Model
// =============================================================================
GPT2 load_gpt2(const char* path) {
    GPT2 m;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    int hdr[5];
    fread(hdr, sizeof(int), 5, f);
    m.config = {hdr[0], hdr[1], hdr[2], hdr[3], hdr[4]};

    int L = m.config.n_layer, C = m.config.n_embd;
    int V = m.config.vocab_size, S = m.config.max_seq_len;
    printf("[GPT2] %d layers, %d heads, %d dim, %d vocab, %d max_seq\n",
           L, m.config.n_head, C, V, S);

    // Total parameter count
    size_t np = (size_t)V*C + (size_t)S*C;  // embeddings
    for (int l = 0; l < L; l++)
        np += 2*C + (size_t)3*C*C + 3*C + (size_t)C*C + C +
              2*C + (size_t)4*C*C + 4*C + (size_t)4*C*C + C;
    np += 2*C;  // final LN
    printf("[GPT2] %.1fM params (%.0f MB)\n", np/1e6, np*4.0/1024/1024);

    m.params_buf = (float*)malloc(np * sizeof(float));
    fread(m.params_buf, sizeof(float), np, f);
    fclose(f);

    // Wire pointers
    float* p = m.params_buf;
    m.wte = p; p += (size_t)V*C;
    m.wpe = p; p += (size_t)S*C;
    m.layers.resize(L);
    for (int l = 0; l < L; l++) {
        auto& w = m.layers[l];
        w.ln_1_w = p; p += C;  w.ln_1_b = p; p += C;
        w.attn_w = p; p += (size_t)3*C*C;  w.attn_b = p; p += 3*C;
        w.attn_proj_w = p; p += (size_t)C*C;  w.attn_proj_b = p; p += C;
        w.ln_2_w = p; p += C;  w.ln_2_b = p; p += C;
        w.mlp_fc_w = p; p += (size_t)4*C*C;  w.mlp_fc_b = p; p += 4*C;
        w.mlp_proj_w = p; p += (size_t)4*C*C;  w.mlp_proj_b = p; p += C;
    }
    m.ln_f_w = p; p += C;  m.ln_f_b = p; p += C;

    // Scratch buffers
    int D = C / m.config.n_head;
    m.x        = (float*)calloc((size_t)S*C, sizeof(float));
    m.x2       = (float*)calloc((size_t)S*C, sizeof(float));
    m.qkv      = (float*)calloc((size_t)S*3*C, sizeof(float));
    m.attn_out = (float*)calloc((size_t)S*C, sizeof(float));
    m.attn_sc  = (float*)calloc((size_t)S*S, sizeof(float));
    m.q_h      = (float*)calloc((size_t)S*D, sizeof(float));
    m.k_h      = (float*)calloc((size_t)S*D, sizeof(float));
    m.v_h      = (float*)calloc((size_t)S*D, sizeof(float));
    m.y_h      = (float*)calloc((size_t)S*D, sizeof(float));
    m.mlp_buf  = (float*)calloc((size_t)S*4*C, sizeof(float));
    m.logits   = (float*)calloc(V, sizeof(float));

    printf("[GPT2] Model loaded!\n");
    return m;
}

void free_gpt2(GPT2& m) {
    free(m.params_buf);
    free(m.x); free(m.x2); free(m.qkv); free(m.attn_out);
    free(m.attn_sc); free(m.q_h); free(m.k_h); free(m.v_h); free(m.y_h);
    free(m.mlp_buf); free(m.logits);
}
