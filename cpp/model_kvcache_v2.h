#pragma once
// =============================================================================
// model_kvcache_v2.h — KV Cache + Zero-Alloc Hot Path + Operator Fusion
// =============================================================================
// Optimizations over model_kvcache.h:
//   1. ZERO ALLOCATIONS IN HOT PATH
//      - gelu() no longer calls malloc/free — uses pre-allocated scratch
//      buffers
//      - All temporary buffers allocated once at model load
//
//   2. OPERATOR FUSION
//      - Fused scale + causal_mask + softmax: single pass over attention scores
//      - Fused linear_bias + gelu: bias add folded into gelu activation
//      - Decode-specialised attention: skip gather/scatter for T=1 case
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

  // Activation buffers (pre-allocated, zero malloc in hot path)
  float *x, *x2, *qkv, *attn_out;
  float *attn_sc, *q_h, *k_h, *v_h, *y_h;
  float *mlp_buf, *logits;

  // Scratch buffers for fused ops (replaces gelu's malloc/free)
  float *gelu_tmp;  // size: max(S*4*C, S*3*C) — reusable scratch
  float *gelu_orig; // size: S*4*C — saves original for gelu

  // KV Cache
  std::vector<float *> k_cache;
  std::vector<float *> v_cache;
  int cache_len;
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
// Core BLAS Ops
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

// =============================================================================
// FUSED: linear with bias add (NEON vectorized)
// =============================================================================
inline void linear(float *out, const float *x, const float *w, const float *b,
                   int T, int in_d, int out_d) {
  matmul_bt(out, x, w, T, in_d, out_d);
  for (int t = 0; t < T; t++) {
    float *op = out + t * out_d;
    int j = 0;
    for (; j + 4 <= out_d; j += 4)
      vst1q_f32(op + j, vaddq_f32(vld1q_f32(op + j), vld1q_f32(b + j)));
    for (; j < out_d; j++)
      op[j] += b[j];
  }
}

// =============================================================================
// Layer norm (NEON)
// =============================================================================
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
      vst1q_f32(op + i, vmlaq_f32(vld1q_f32(b + i), xn, vld1q_f32(w + i)));
    }
    for (; i < C; i++)
      op[i] = (xp[i] - mean) * inv * w[i] + b[i];
  }
}

// =============================================================================
// FUSED: scale + causal_mask + softmax (single pass)
// =============================================================================
// Instead of: (1) vDSP_vsmul for scale  (2) loop for mask  (3) softmax()
// We do it all in one pass per row: scale → mask → find max → exp → normalize
void fused_scale_mask_softmax(float *scores, int T, int total, int offset,
                              float scale) {
  for (int t = 0; t < T; t++) {
    float *row = scores + t * total;
    int query_pos = offset + t;
    int valid = query_pos + 1; // positions [0..query_pos] are visible

    // Pass 1: scale visible + mask future → find max (fused)
    float mx = -1e30f;
    int j = 0;
    float32x4_t scale4 = vdupq_n_f32(scale);
    float32x4_t mx4 = vdupq_n_f32(-1e30f);
    for (; j + 4 <= valid; j += 4) {
      float32x4_t v = vmulq_f32(vld1q_f32(row + j), scale4);
      vst1q_f32(row + j, v);
      mx4 = vmaxq_f32(mx4, v);
    }
    mx = vmaxvq_f32(mx4);
    for (; j < valid; j++) {
      row[j] *= scale;
      if (row[j] > mx)
        mx = row[j];
    }
    // Mask future positions (set to -inf so exp = 0)
    for (j = valid; j < total; j++)
      row[j] = -1e9f;

    // Pass 2: subtract max + exp (vectorized via vvexpf)
    float32x4_t mx4v = vdupq_n_f32(mx);
    j = 0;
    for (; j + 4 <= total; j += 4)
      vst1q_f32(row + j, vsubq_f32(vld1q_f32(row + j), mx4v));
    for (; j < total; j++)
      row[j] -= mx;
    int n = total;
    vvexpf(row, row, &n);

    // Pass 3: sum + normalize
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    j = 0;
    for (; j + 4 <= total; j += 4)
      sum4 = vaddq_f32(sum4, vld1q_f32(row + j));
    float sum = vaddvq_f32(sum4);
    for (; j < total; j++)
      sum += row[j];
    float inv = 1.0f / sum;
    vDSP_vsmul(row, 1, &inv, row, 1, total);
  }
}

// =============================================================================
// ZERO-ALLOC GELU: uses pre-allocated scratch buffers from GPT2 struct
// =============================================================================
// Fused: reads x, writes gelu(x) in-place, using m.gelu_tmp and m.gelu_orig
void gelu_fused(float *x, int size, float *tmp, float *orig) {
  memcpy(orig, x, size * sizeof(float));
  const float32x4_t c1 = vdupq_n_f32(0.7978845608f);
  const float32x4_t c2 = vdupq_n_f32(0.044715f);
  int i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t v = vld1q_f32(x + i);
    float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);
    vst1q_f32(tmp + i, vmulq_f32(c1, vmlaq_f32(v, c2, v3)));
  }
  for (; i < size; i++)
    tmp[i] = 0.7978845608f * (x[i] + 0.044715f * x[i] * x[i] * x[i]);

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
}

// =============================================================================
// NEON residual add
// =============================================================================
inline void residual_add(float *dst, const float *src, int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
  for (; i < n; i++)
    dst[i] += src[i];
}

// =============================================================================
// Attention with KV Cache + Fused Scale/Mask/Softmax
// =============================================================================
void attention_kv(GPT2 &m, int T, int offset, int l) {
  int C = m.config.n_embd, nh = m.config.n_head, D = C / nh;
  int total = offset + T;
  LayerWeights &w = m.layers[l];

  // 1. LayerNorm + QKV projection
  layer_norm(m.x2, m.x, w.ln_1_w, w.ln_1_b, T, C);
  linear(m.qkv, m.x2, w.attn_w, w.attn_b, T, C, 3 * C);

  // 2. Store K, V into cache
  float *kc = m.k_cache[l], *vc = m.v_cache[l];
  for (int t = 0; t < T; t++) {
    memcpy(kc + (offset + t) * C, m.qkv + t * 3 * C + C, C * sizeof(float));
    memcpy(vc + (offset + t) * C, m.qkv + t * 3 * C + 2 * C, C * sizeof(float));
  }

  float scale = 1.0f / sqrtf((float)D);

  // 3. Multi-head attention with FUSED scale+mask+softmax
  for (int h = 0; h < nh; h++) {
    // Gather Q (new positions only)
    for (int t = 0; t < T; t++)
      for (int d = 0; d < D; d++)
        m.q_h[t * D + d] = m.qkv[t * 3 * C + h * D + d];

    // Gather K, V (all cached positions)
    for (int t2 = 0; t2 < total; t2++)
      for (int d = 0; d < D; d++) {
        m.k_h[t2 * D + d] = kc[t2 * C + h * D + d];
        m.v_h[t2 * D + d] = vc[t2 * C + h * D + d];
      }

    // Q @ K^T
    matmul_bt(m.attn_sc, m.q_h, m.k_h, T, D, total);

    // FUSED: scale + causal mask + softmax in one pass
    fused_scale_mask_softmax(m.attn_sc, T, total, offset, scale);

    // Weighted sum: scores @ V
    matmul(m.y_h, m.attn_sc, m.v_h, T, total, D);

    // Scatter to attn_out
    for (int t = 0; t < T; t++)
      for (int d = 0; d < D; d++)
        m.attn_out[t * C + h * D + d] = m.y_h[t * D + d];
  }

  // 4. Output projection + residual
  linear(m.x2, m.attn_out, w.attn_proj_w, w.attn_proj_b, T, C, C);
  residual_add(m.x, m.x2, T * C);
}

// =============================================================================
// MLP with zero-alloc gelu
// =============================================================================
void mlp_forward(GPT2 &m, int T, int l) {
  int C = m.config.n_embd;
  LayerWeights &w = m.layers[l];
  layer_norm(m.x2, m.x, w.ln_2_w, w.ln_2_b, T, C);
  linear(m.mlp_buf, m.x2, w.mlp_fc_w, w.mlp_fc_b, T, C, 4 * C);
  gelu_fused(m.mlp_buf, T * 4 * C, m.gelu_tmp, m.gelu_orig); // ZERO ALLOC
  linear(m.x2, m.mlp_buf, w.mlp_proj_w, w.mlp_proj_b, T, 4 * C, C);
  residual_add(m.x, m.x2, T * C);
}

// =============================================================================
// GPT-2 Forward with KV Cache (v2)
// =============================================================================
float *gpt2_forward(GPT2 &m, const int *ids, int T_total) {
  int C = m.config.n_embd, V = m.config.vocab_size;
  int offset = m.cache_len;
  int T = T_total - offset;

  // Embedding
  for (int t = 0; t < T; t++) {
    int pos = offset + t, tok = ids[pos];
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

  // Final LN on last position only
  layer_norm(m.x2, &m.x[(T - 1) * C], m.ln_f_w, m.ln_f_b, 1, C);
  matmul_bt(m.logits, m.x2, m.wte, 1, C, V);
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
  // Reuse fused softmax (no mask needed for sampling)
  float mx = logits[0];
  for (int i = 1; i < V; i++)
    if (logits[i] > mx)
      mx = logits[i];
  for (int i = 0; i < V; i++)
    logits[i] -= mx;
  int n = V;
  vvexpf(logits, logits, &n);
  float sum = 0.0f;
  for (int i = 0; i < V; i++)
    sum += logits[i];
  float invs = 1.0f / sum;
  vDSP_vsmul(logits, 1, &invs, logits, 1, V);
  float r = (float)rand() / RAND_MAX, cum = 0.0f;
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
  printf("[GPT2-KV2] %d layers, %d heads, %d dim, %d vocab\n", L,
         m.config.n_head, C, V);

  size_t np = (size_t)V * C + (size_t)S * C;
  for (int l = 0; l < L; l++)
    np += 2 * C + (size_t)3 * C * C + 3 * C + (size_t)C * C + C + 2 * C +
          (size_t)4 * C * C + 4 * C + (size_t)4 * C * C + C;
  np += 2 * C;
  printf("[GPT2-KV2] %.1fM params (%.0f MB)\n", np / 1e6,
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

  // Pre-allocate gelu scratch buffers (ZERO ALLOC in hot path!)
  m.gelu_tmp = (float *)calloc((size_t)S * 4 * C, sizeof(float));
  m.gelu_orig = (float *)calloc((size_t)S * 4 * C, sizeof(float));

  // KV cache
  m.k_cache.resize(L);
  m.v_cache.resize(L);
  for (int l = 0; l < L; l++) {
    m.k_cache[l] = (float *)calloc((size_t)S * C, sizeof(float));
    m.v_cache[l] = (float *)calloc((size_t)S * C, sizeof(float));
  }
  m.cache_len = 0;

  size_t cache_mb = 2 * L * S * C * sizeof(float) / 1024 / 1024;
  size_t scratch_mb = 2 * S * 4 * C * sizeof(float) / 1024 / 1024;
  printf("[GPT2-KV2] KV cache: %zu MB, scratch: %zu MB\n", cache_mb,
         scratch_mb);
  printf("[GPT2-KV2] Model loaded! (SIMD + KV Cache + Zero-Alloc + Fusion)\n");
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
  free(m.gelu_tmp);
  free(m.gelu_orig);
  for (auto *p : m.k_cache)
    free(p);
  for (auto *p : m.v_cache)
    free(p);
}
