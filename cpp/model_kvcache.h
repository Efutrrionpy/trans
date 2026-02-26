#pragma once
// =============================================================================
// model_kvcache.h — SIMD GPT-2 with KV Cache (Apple Silicon)
// =============================================================================
// Key optimization over model_simd.h:
//   After the initial "prefill" of all input tokens, each subsequent
//   generation step processes ONLY 1 new token instead of the entire
//   sequence. K and V from every layer are cached across steps.
//
//   Complexity per step: O(T × C²)  →  O(C² + T × C)
//   where T = current sequence length, C = embedding dim
// =============================================================================

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// === Structs =================================================================
struct GPT2Config {
  int n_layer, n_head, n_embd, vocab_size, max_seq_len;
};

struct LayerWeights {
  float *ln_1_w, *ln_1_b;
  float *attn_w, *attn_b;
  float *attn_proj_w, *attn_proj_b;
  float *ln_2_w, *ln_2_b;
  float *mlp_fc_w, *mlp_fc_b;
  float *mlp_proj_w, *mlp_proj_b;
};

struct GPT2 {
  GPT2Config config;
  float *params_buf;
  float *wte, *wpe;
  std::vector<LayerWeights> layers;
  float *ln_f_w, *ln_f_b;

  // Activation buffers (sized for max batch)
  float *x, *x2, *qkv, *attn_out;
  float *attn_sc, *q_h, *k_h, *v_h, *y_h;
  float *mlp_buf, *logits;

  // KV Cache: per-layer, stores [max_seq_len, n_embd]
  std::vector<float *> k_cache; // k_cache[layer][pos * n_embd + ...]
  std::vector<float *> v_cache;
  int cache_len; // number of positions already cached
};

struct Vocab {
  int n_vocab;
  std::vector<std::string> tokens;
};

Vocab load_vocab(const char *path) {
  Vocab v;
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    exit(1);
  }
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

std::string decode(const Vocab &v, const std::vector<int> &ids) {
  std::string out;
  for (int id : ids)
    if (id >= 0 && id < v.n_vocab)
      out += v.tokens[id];
  return out;
}

// =============================================================================
// Tensor Ops (same as model_simd.h — Accelerate BLAS + NEON)
// =============================================================================
inline void matmul_bt(float *C, const float *A, const float *B, int M, int K,
                      int N) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B,
              K, 0.0f, C, N);
}

inline void matmul(float *C, const float *A, const float *B, int M, int K,
                   int N) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B,
              N, 0.0f, C, N);
}

inline void linear(float *out, const float *x, const float *w, const float *b,
                   int T, int in_d, int out_d) {
  matmul_bt(out, x, w, T, in_d, out_d);
  for (int t = 0; t < T; t++) {
    float *op = out + t * out_d;
    int j = 0;
    for (; j + 4 <= out_d; j += 4) {
      float32x4_t v = vld1q_f32(op + j);
      float32x4_t bv = vld1q_f32(b + j);
      vst1q_f32(op + j, vaddq_f32(v, bv));
    }
    for (; j < out_d; j++)
      op[j] += b[j];
  }
}

void layer_norm(float *out, const float *x, const float *w, const float *b,
                int T, int C) {
  for (int t = 0; t < T; t++) {
    const float *xp = x + t * C;
    float *op = out + t * C;
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= C; i += 4)
      sum4 = vaddq_f32(sum4, vld1q_f32(xp + i));
    float mean = vaddvq_f32(sum4);
    for (; i < C; i++)
      mean += xp[i];
    mean /= C;

    float32x4_t mean4 = vdupq_n_f32(mean);
    float32x4_t var4 = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 4 <= C; i += 4) {
      float32x4_t d = vsubq_f32(vld1q_f32(xp + i), mean4);
      var4 = vmlaq_f32(var4, d, d);
    }
    float var = vaddvq_f32(var4);
    for (; i < C; i++) {
      float d = xp[i] - mean;
      var += d * d;
    }
    var /= C;

    float inv = 1.0f / sqrtf(var + 1e-5f);
    float32x4_t inv4 = vdupq_n_f32(inv);
    i = 0;
    for (; i + 4 <= C; i += 4) {
      float32x4_t xn = vmulq_f32(vsubq_f32(vld1q_f32(xp + i), mean4), inv4);
      float32x4_t wv = vld1q_f32(w + i);
      float32x4_t bv = vld1q_f32(b + i);
      vst1q_f32(op + i, vmlaq_f32(bv, xn, wv));
    }
    for (; i < C; i++)
      op[i] = (xp[i] - mean) * inv * w[i] + b[i];
  }
}

void softmax(float *x, int size) {
  float mx = x[0];
  int i = 1;
  if (size >= 4) {
    float32x4_t mx4 = vld1q_f32(x);
    for (i = 4; i + 4 <= size; i += 4)
      mx4 = vmaxq_f32(mx4, vld1q_f32(x + i));
    mx = vmaxvq_f32(mx4);
    for (; i < size; i++)
      if (x[i] > mx)
        mx = x[i];
  } else {
    for (; i < size; i++)
      if (x[i] > mx)
        mx = x[i];
  }
  float32x4_t mx4v = vdupq_n_f32(mx);
  i = 0;
  for (; i + 4 <= size; i += 4)
    vst1q_f32(x + i, vsubq_f32(vld1q_f32(x + i), mx4v));
  for (; i < size; i++)
    x[i] -= mx;
  int n = size;
  vvexpf(x, x, &n);
  float32x4_t sum4 = vdupq_n_f32(0.0f);
  i = 0;
  for (; i + 4 <= size; i += 4)
    sum4 = vaddq_f32(sum4, vld1q_f32(x + i));
  float sum = vaddvq_f32(sum4);
  for (; i < size; i++)
    sum += x[i];
  float inv = 1.0f / sum;
  vDSP_vsmul(x, 1, &inv, x, 1, size);
}

void gelu(float *x, int size) {
  float *tmp = (float *)malloc(size * sizeof(float));
  float *orig = (float *)malloc(size * sizeof(float));
  memcpy(orig, x, size * sizeof(float));
  const float32x4_t c1 = vdupq_n_f32(0.7978845608f);
  const float32x4_t c2 = vdupq_n_f32(0.044715f);
  int i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t v = vld1q_f32(x + i);
    float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);
    float32x4_t inner = vmlaq_f32(v, c2, v3);
    vst1q_f32(tmp + i, vmulq_f32(c1, inner));
  }
  for (; i < size; i++) {
    float v = x[i];
    tmp[i] = 0.7978845608f * (v + 0.044715f * v * v * v);
  }
  int n = size;
  vvtanhf(tmp, tmp, &n);
  const float32x4_t half = vdupq_n_f32(0.5f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t v = vld1q_f32(orig + i);
    float32x4_t t = vld1q_f32(tmp + i);
    vst1q_f32(x + i, vmulq_f32(half, vmulq_f32(v, vaddq_f32(one, t))));
  }
  for (; i < size; i++)
    x[i] = 0.5f * orig[i] * (1.0f + tmp[i]);
  free(tmp);
  free(orig);
}

// =============================================================================
// NEON residual add helper
// =============================================================================
inline void residual_add(float *dst, const float *src, int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
  for (; i < n; i++)
    dst[i] += src[i];
}

// =============================================================================
// Attention with KV Cache
// =============================================================================
//   T      = number of NEW tokens to process (T = seq_len for prefill, 1 for
//   decode) offset = starting position in the sequence (0 for prefill,
//   cache_len for decode) total  = total sequence positions to attend over =
//   offset + T
void attention_kv(GPT2 &m, int T, int offset, int l) {
  int C = m.config.n_embd, nh = m.config.n_head, D = C / nh;
  int total = offset + T; // total positions with cached + new
  LayerWeights &w = m.layers[l];

  // 1. LayerNorm + QKV projection for the T new positions
  layer_norm(m.x2, m.x, w.ln_1_w, w.ln_1_b, T, C);
  linear(m.qkv, m.x2, w.attn_w, w.attn_b, T, C, 3 * C);

  // 2. Store new K, V into cache at positions [offset, offset+T)
  float *kc = m.k_cache[l];
  float *vc = m.v_cache[l];
  for (int t = 0; t < T; t++) {
    memcpy(kc + (offset + t) * C, m.qkv + t * 3 * C + C,
           C * sizeof(float)); // K
    memcpy(vc + (offset + t) * C, m.qkv + t * 3 * C + 2 * C,
           C * sizeof(float)); // V
  }

  // 3. Multi-head attention: Q attends over ALL cached K,V [0..total)
  for (int h = 0; h < nh; h++) {
    // Extract Q for the T new positions (from qkv buffer)
    for (int t = 0; t < T; t++)
      for (int d = 0; d < D; d++)
        m.q_h[t * D + d] = m.qkv[t * 3 * C + h * D + d];

    // Gather K and V for ALL cached positions [0..total) for this head
    for (int t2 = 0; t2 < total; t2++)
      for (int d = 0; d < D; d++) {
        m.k_h[t2 * D + d] = kc[t2 * C + h * D + d];
        m.v_h[t2 * D + d] = vc[t2 * C + h * D + d];
      }

    // Q[T, D] @ K[total, D]^T = scores[T, total]
    matmul_bt(m.attn_sc, m.q_h, m.k_h, T, D, total);
    float scale = 1.0f / sqrtf((float)D);
    vDSP_vsmul(m.attn_sc, 1, &scale, m.attn_sc, 1, T * total);

    // Causal mask: position (offset+t) can only attend to positions <=
    // (offset+t)
    for (int t = 0; t < T; t++) {
      int query_pos = offset + t;
      for (int t2 = query_pos + 1; t2 < total; t2++)
        m.attn_sc[t * total + t2] = -1e9f;
      softmax(&m.attn_sc[t * total], total);
    }

    // scores[T, total] @ V[total, D] = y[T, D]
    matmul(m.y_h, m.attn_sc, m.v_h, T, total, D);

    // Scatter back to attn_out
    for (int t = 0; t < T; t++)
      for (int d = 0; d < D; d++)
        m.attn_out[t * C + h * D + d] = m.y_h[t * D + d];
  }

  // 4. Output projection + residual
  linear(m.x2, m.attn_out, w.attn_proj_w, w.attn_proj_b, T, C, C);
  residual_add(m.x, m.x2, T * C);
}

// =============================================================================
// MLP (same logic, just takes T count)
// =============================================================================
void mlp_forward(GPT2 &m, int T, int l) {
  int C = m.config.n_embd;
  LayerWeights &w = m.layers[l];
  layer_norm(m.x2, m.x, w.ln_2_w, w.ln_2_b, T, C);
  linear(m.mlp_buf, m.x2, w.mlp_fc_w, w.mlp_fc_b, T, C, 4 * C);
  gelu(m.mlp_buf, T * 4 * C);
  linear(m.x2, m.mlp_buf, w.mlp_proj_w, w.mlp_proj_b, T, 4 * C, C);
  residual_add(m.x, m.x2, T * C);
}

// =============================================================================
// GPT-2 Forward Pass with KV Cache
// =============================================================================
// ids: array of ALL token IDs so far (length T_total)
// The function automatically determines prefill vs decode:
//   - If cache_len == 0: PREFILL — process all T_total tokens
//   - If cache_len > 0:  DECODE  — only process new tokens (T_total -
//   cache_len)
float *gpt2_forward(GPT2 &m, const int *ids, int T_total) {
  int C = m.config.n_embd, V = m.config.vocab_size;

  int offset = m.cache_len; // positions already cached
  int T = T_total -
          offset; // new tokens to process (T_total for prefill, 1 for decode)

  // Embedding: only for the T new tokens at positions [offset, offset+T)
  for (int t = 0; t < T; t++) {
    int pos = offset + t;
    int tok = ids[pos];
    int i = 0;
    for (; i + 4 <= C; i += 4)
      vst1q_f32(m.x + t * C + i, vaddq_f32(vld1q_f32(m.wte + tok * C + i),
                                           vld1q_f32(m.wpe + pos * C + i)));
    for (; i < C; i++)
      m.x[t * C + i] = m.wte[tok * C + i] + m.wpe[pos * C + i];
  }

  // Transformer blocks
  for (int l = 0; l < m.config.n_layer; l++) {
    attention_kv(m, T, offset, l);
    mlp_forward(m, T, l);
  }

  // Final layer norm on the last position's embedding only
  layer_norm(m.x2, &m.x[(T - 1) * C], m.ln_f_w, m.ln_f_b, 1, C);

  // LM head: logits = x2 @ wte^T
  matmul_bt(m.logits, m.x2, m.wte, 1, C, V);

  // Update cache length
  m.cache_len = T_total;

  return m.logits;
}

// === Sampling ================================================================
int sample_argmax(const float *logits, int V) {
  int best = 0;
  for (int i = 1; i < V; i++)
    if (logits[i] > logits[best])
      best = i;
  return best;
}

int sample_temp(float *logits, int V, float temp) {
  if (temp <= 0.0f)
    return sample_argmax(logits, V);
  float inv = 1.0f / temp;
  vDSP_vsmul(logits, 1, &inv, logits, 1, V);
  softmax(logits, V);
  float r = (float)rand() / RAND_MAX;
  float cum = 0.0f;
  for (int i = 0; i < V; i++) {
    cum += logits[i];
    if (cum >= r)
      return i;
  }
  return V - 1;
}

// === Load Model ==============================================================
GPT2 load_gpt2(const char *path) {
  GPT2 m;
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    exit(1);
  }

  int hdr[5];
  fread(hdr, sizeof(int), 5, f);
  m.config = {hdr[0], hdr[1], hdr[2], hdr[3], hdr[4]};
  int L = m.config.n_layer, C = m.config.n_embd;
  int V = m.config.vocab_size, S = m.config.max_seq_len;
  printf("[GPT2-KV] %d layers, %d heads, %d dim, %d vocab\n", L,
         m.config.n_head, C, V);

  size_t np = (size_t)V * C + (size_t)S * C;
  for (int l = 0; l < L; l++)
    np += 2 * C + (size_t)3 * C * C + 3 * C + (size_t)C * C + C + 2 * C +
          (size_t)4 * C * C + 4 * C + (size_t)4 * C * C + C;
  np += 2 * C;
  printf("[GPT2-KV] %.1fM params (%.0f MB)\n", np / 1e6,
         np * 4.0 / 1024 / 1024);

  m.params_buf = (float *)malloc(np * sizeof(float));
  fread(m.params_buf, sizeof(float), np, f);
  fclose(f);

  float *p = m.params_buf;
  m.wte = p;
  p += (size_t)V * C;
  m.wpe = p;
  p += (size_t)S * C;
  m.layers.resize(L);
  for (int l = 0; l < L; l++) {
    auto &w = m.layers[l];
    w.ln_1_w = p;
    p += C;
    w.ln_1_b = p;
    p += C;
    w.attn_w = p;
    p += (size_t)3 * C * C;
    w.attn_b = p;
    p += 3 * C;
    w.attn_proj_w = p;
    p += (size_t)C * C;
    w.attn_proj_b = p;
    p += C;
    w.ln_2_w = p;
    p += C;
    w.ln_2_b = p;
    p += C;
    w.mlp_fc_w = p;
    p += (size_t)4 * C * C;
    w.mlp_fc_b = p;
    p += 4 * C;
    w.mlp_proj_w = p;
    p += (size_t)4 * C * C;
    w.mlp_proj_b = p;
    p += C;
  }
  m.ln_f_w = p;
  p += C;
  m.ln_f_b = p;
  p += C;

  int D = C / m.config.n_head;
  m.x = (float *)calloc((size_t)S * C, sizeof(float));
  m.x2 = (float *)calloc((size_t)S * C, sizeof(float));
  m.qkv = (float *)calloc((size_t)S * 3 * C, sizeof(float));
  m.attn_out = (float *)calloc((size_t)S * C, sizeof(float));
  m.attn_sc = (float *)calloc((size_t)S * S, sizeof(float));
  m.q_h = (float *)calloc((size_t)S * D, sizeof(float));
  m.k_h = (float *)calloc((size_t)S * D, sizeof(float));
  m.v_h = (float *)calloc((size_t)S * D, sizeof(float));
  m.y_h = (float *)calloc((size_t)S * D, sizeof(float));
  m.mlp_buf = (float *)calloc((size_t)S * 4 * C, sizeof(float));
  m.logits = (float *)calloc(V, sizeof(float));

  // Allocate KV cache: one [S, C] buffer per layer
  m.k_cache.resize(L);
  m.v_cache.resize(L);
  for (int l = 0; l < L; l++) {
    m.k_cache[l] = (float *)calloc((size_t)S * C, sizeof(float));
    m.v_cache[l] = (float *)calloc((size_t)S * C, sizeof(float));
  }
  m.cache_len = 0;

  size_t cache_mb = 2 * L * S * C * sizeof(float) / 1024 / 1024;
  printf("[GPT2-KV] KV cache: %zu MB (%d layers × %d positions × %d dim × 2)\n",
         cache_mb, L, S, C);
  printf("[GPT2-KV] Model loaded! (SIMD + KV Cache)\n");
  return m;
}

void free_gpt2(GPT2 &m) {
  free(m.params_buf);
  free(m.x);
  free(m.x2);
  free(m.qkv);
  free(m.attn_out);
  free(m.attn_sc);
  free(m.q_h);
  free(m.k_h);
  free(m.v_h);
  free(m.y_h);
  free(m.mlp_buf);
  free(m.logits);
  for (auto *p : m.k_cache)
    free(p);
  for (auto *p : m.v_cache)
    free(p);
}
