#pragma once
// =============================================================================
// model_q8_kv.h — INT8 Quantized GPT-2 with KV Cache (Apple Silicon)
// =============================================================================
// Weight matrices stored as int8 + float32 per-row scales.
// Dequantized on-the-fly into a pre-allocated FP32 scratch buffer, then
// fed to cblas_sgemm. Biases and LayerNorm params remain FP32.
//
// Memory: ~120MB (INT8 weights) + ~10MB (scratch) vs ~475MB (full FP32)
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

// Quantized weight matrix: int8 data + float32 per-row scales
struct Q8Weight {
  int8_t *data;  // [out_dim, in_dim]  — row-major int8
  float *scales; // [out_dim]          — per-row scale
  int out_dim, in_dim;
};

struct LayerWeights {
  float *ln_1_w, *ln_1_b;
  Q8Weight attn_w;
  float *attn_b;
  Q8Weight attn_proj_w;
  float *attn_proj_b;
  float *ln_2_w, *ln_2_b;
  Q8Weight mlp_fc_w;
  float *mlp_fc_b;
  Q8Weight mlp_proj_w;
  float *mlp_proj_b;
};

struct GPT2 {
  GPT2Config config;

  // Raw buffers
  int8_t *q8_buf; // all int8 weights packed
  float *f32_buf; // all float32 data (scales, biases, LN, embeddings)

  Q8Weight wte_q; // quantized token embeddings
  Q8Weight wpe_q; // quantized position embeddings
  std::vector<LayerWeights> layers;
  float *ln_f_w, *ln_f_b;

  // Activation buffers
  float *x, *x2, *qkv, *attn_out;
  float *attn_sc, *q_h, *k_h, *v_h, *y_h;
  float *mlp_buf, *logits;
  float *gelu_tmp, *gelu_orig;

  // Dequantization scratch: large enough for biggest weight matrix
  float *dequant_scratch; // max(V*C, 4C*C) float32

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
// Dequantization: INT8 → FP32 with per-row scales (NEON accelerated)
// =============================================================================
inline void dequantize_rows(float *out, const int8_t *q, const float *scales,
                            int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    float s = scales[r];
    float32x4_t s4 = vdupq_n_f32(s);
    const int8_t *qr = q + r * cols;
    float *or_ = out + r * cols;
    int j = 0;
    for (; j + 8 <= cols; j += 8) {
      // Load 8 int8 → 2 × 4 float32
      int8x8_t q8 = vld1_s8(qr + j);
      int16x8_t q16 = vmovl_s8(q8);
      int32x4_t lo = vmovl_s16(vget_low_s16(q16));
      int32x4_t hi = vmovl_s16(vget_high_s16(q16));
      vst1q_f32(or_ + j, vmulq_f32(vcvtq_f32_s32(lo), s4));
      vst1q_f32(or_ + j + 4, vmulq_f32(vcvtq_f32_s32(hi), s4));
    }
    for (; j < cols; j++)
      or_[j] = (float)qr[j] * s;
  }
}

// Dequantize specific rows (for embedding lookup)
inline void dequantize_row(float *out, const Q8Weight &w, int row) {
  float s = w.scales[row];
  float32x4_t s4 = vdupq_n_f32(s);
  const int8_t *qr = w.data + row * w.in_dim;
  int j = 0;
  for (; j + 8 <= w.in_dim; j += 8) {
    int8x8_t q8 = vld1_s8(qr + j);
    int16x8_t q16 = vmovl_s8(q8);
    int32x4_t lo = vmovl_s16(vget_low_s16(q16));
    int32x4_t hi = vmovl_s16(vget_high_s16(q16));
    vst1q_f32(out + j, vmulq_f32(vcvtq_f32_s32(lo), s4));
    vst1q_f32(out + j + 4, vmulq_f32(vcvtq_f32_s32(hi), s4));
  }
  for (; j < w.in_dim; j++)
    out[j] = (float)qr[j] * s;
}

// =============================================================================
// Core BLAS + NEON ops (same as kvcache_v2)
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

// Quantized linear: dequantize weight → cblas_sgemm → add bias
inline void q8_linear(float *out, const float *x, const Q8Weight &w,
                      const float *bias, int T, float *scratch) {
  dequantize_rows(scratch, w.data, w.scales, w.out_dim, w.in_dim);
  matmul_bt(out, x, scratch, T, w.in_dim, w.out_dim);
  // Add bias (NEON)
  for (int t = 0; t < T; t++) {
    float *op = out + t * w.out_dim;
    int j = 0;
    for (; j + 4 <= w.out_dim; j += 4)
      vst1q_f32(op + j, vaddq_f32(vld1q_f32(op + j), vld1q_f32(bias + j)));
    for (; j < w.out_dim; j++)
      op[j] += bias[j];
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
      vst1q_f32(op + i, vmlaq_f32(vld1q_f32(b + i), xn, vld1q_f32(w + i)));
    }
    for (; i < C; i++)
      op[i] = (xp[i] - mean) * inv * w[i] + b[i];
  }
}

void fused_scale_mask_softmax(float *scores, int T, int total, int offset,
                              float scale) {
  for (int t = 0; t < T; t++) {
    float *row = scores + t * total;
    int valid = offset + t + 1;
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
    for (j = valid; j < total; j++)
      row[j] = -1e9f;

    float32x4_t mx4v = vdupq_n_f32(mx);
    j = 0;
    for (; j + 4 <= total; j += 4)
      vst1q_f32(row + j, vsubq_f32(vld1q_f32(row + j), mx4v));
    for (; j < total; j++)
      row[j] -= mx;
    int n = total;
    vvexpf(row, row, &n);

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

inline void residual_add(float *dst, const float *src, int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
  for (; i < n; i++)
    dst[i] += src[i];
}

// =============================================================================
// Attention with KV Cache (Q8 weights)
// =============================================================================
void attention_kv(GPT2 &m, int T, int offset, int l) {
  int C = m.config.n_embd, nh = m.config.n_head, D = C / nh;
  int total = offset + T;
  LayerWeights &w = m.layers[l];

  layer_norm(m.x2, m.x, w.ln_1_w, w.ln_1_b, T, C);
  q8_linear(m.qkv, m.x2, w.attn_w, w.attn_b, T, m.dequant_scratch);

  float *kc = m.k_cache[l], *vc = m.v_cache[l];
  for (int t = 0; t < T; t++) {
    memcpy(kc + (offset + t) * C, m.qkv + t * 3 * C + C, C * sizeof(float));
    memcpy(vc + (offset + t) * C, m.qkv + t * 3 * C + 2 * C, C * sizeof(float));
  }

  float scale = 1.0f / sqrtf((float)D);

  for (int h = 0; h < nh; h++) {
    for (int t = 0; t < T; t++)
      for (int d = 0; d < D; d++)
        m.q_h[t * D + d] = m.qkv[t * 3 * C + h * D + d];

    for (int t2 = 0; t2 < total; t2++)
      for (int d = 0; d < D; d++) {
        m.k_h[t2 * D + d] = kc[t2 * C + h * D + d];
        m.v_h[t2 * D + d] = vc[t2 * C + h * D + d];
      }

    matmul_bt(m.attn_sc, m.q_h, m.k_h, T, D, total);
    fused_scale_mask_softmax(m.attn_sc, T, total, offset, scale);
    matmul(m.y_h, m.attn_sc, m.v_h, T, total, D);

    for (int t = 0; t < T; t++)
      for (int d = 0; d < D; d++)
        m.attn_out[t * C + h * D + d] = m.y_h[t * D + d];
  }

  q8_linear(m.x2, m.attn_out, w.attn_proj_w, w.attn_proj_b, T,
            m.dequant_scratch);
  residual_add(m.x, m.x2, T * C);
}

// =============================================================================
// MLP (Q8 weights)
// =============================================================================
void mlp_forward(GPT2 &m, int T, int l) {
  int C = m.config.n_embd;
  LayerWeights &w = m.layers[l];
  layer_norm(m.x2, m.x, w.ln_2_w, w.ln_2_b, T, C);
  q8_linear(m.mlp_buf, m.x2, w.mlp_fc_w, w.mlp_fc_b, T, m.dequant_scratch);
  gelu_fused(m.mlp_buf, T * 4 * C, m.gelu_tmp, m.gelu_orig);
  q8_linear(m.x2, m.mlp_buf, w.mlp_proj_w, w.mlp_proj_b, T, m.dequant_scratch);
  residual_add(m.x, m.x2, T * C);
}

// =============================================================================
// Forward pass
// =============================================================================
float *gpt2_forward(GPT2 &m, const int *ids, int T_total) {
  int C = m.config.n_embd, V = m.config.vocab_size;
  int offset = m.cache_len;
  int T = T_total - offset;

  // Embedding: dequantize wte[tok] and wpe[pos] on the fly
  for (int t = 0; t < T; t++) {
    int pos = offset + t, tok = ids[pos];
    float *xp = m.x + t * C;
    // Dequantize token embedding
    dequantize_row(xp, m.wte_q, tok);
    // Dequantize + add position embedding
    float s = m.wpe_q.scales[pos];
    float32x4_t s4 = vdupq_n_f32(s);
    const int8_t *pq = m.wpe_q.data + pos * C;
    int i = 0;
    for (; i + 8 <= C; i += 8) {
      int8x8_t q8 = vld1_s8(pq + i);
      int16x8_t q16 = vmovl_s8(q8);
      int32x4_t lo = vmovl_s16(vget_low_s16(q16));
      int32x4_t hi = vmovl_s16(vget_high_s16(q16));
      float32x4_t plo = vmulq_f32(vcvtq_f32_s32(lo), s4);
      float32x4_t phi = vmulq_f32(vcvtq_f32_s32(hi), s4);
      vst1q_f32(xp + i, vaddq_f32(vld1q_f32(xp + i), plo));
      vst1q_f32(xp + i + 4, vaddq_f32(vld1q_f32(xp + i + 4), phi));
    }
    for (; i < C; i++)
      xp[i] += (float)pq[i] * s;
  }

  for (int l = 0; l < m.config.n_layer; l++) {
    attention_kv(m, T, offset, l);
    mlp_forward(m, T, l);
  }

  // Final LN
  layer_norm(m.x2, &m.x[(T - 1) * C], m.ln_f_w, m.ln_f_b, 1, C);

  // LM head: logits = x2 @ wte^T (dequantize wte into scratch)
  dequantize_rows(m.dequant_scratch, m.wte_q.data, m.wte_q.scales,
                  m.wte_q.out_dim, m.wte_q.in_dim);
  matmul_bt(m.logits, m.x2, m.dequant_scratch, 1, C, V);

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

// =============================================================================
// Load Quantized Model
// =============================================================================
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
  printf("[GPT2-Q8] %d layers, %d heads, %d dim, %d vocab\n", L,
         m.config.n_head, C, V);

  // --- Compute total sizes ---
  // Quantized: wte(V,C), wpe(S,C), per layer: attn_w(3C,C), attn_proj(C,C),
  //            mlp_fc(4C,C), mlp_proj(C,4C)
  // FP32: scales + biases + LN params

  // Helper to read a Q8 weight from file
  auto read_q8 = [&](int out_dim, int in_dim) -> Q8Weight {
    Q8Weight w;
    w.out_dim = out_dim;
    w.in_dim = in_dim;
    w.scales = (float *)malloc(out_dim * sizeof(float));
    w.data = (int8_t *)malloc((size_t)out_dim * in_dim);
    fread(w.scales, sizeof(float), out_dim, f);
    fread(w.data, 1, (size_t)out_dim * in_dim, f);
    return w;
  };

  auto read_f32 = [&](int count) -> float * {
    float *buf = (float *)malloc(count * sizeof(float));
    fread(buf, sizeof(float), count, f);
    return buf;
  };

  // Read weights in same order as model_q8.bin
  m.wte_q = read_q8(V, C);
  m.wpe_q = read_q8(S, C);

  m.layers.resize(L);
  for (int l = 0; l < L; l++) {
    auto &w = m.layers[l];
    w.ln_1_w = read_f32(C);
    w.ln_1_b = read_f32(C);
    w.attn_w = read_q8(3 * C, C);
    w.attn_b = read_f32(3 * C);
    w.attn_proj_w = read_q8(C, C);
    w.attn_proj_b = read_f32(C);
    w.ln_2_w = read_f32(C);
    w.ln_2_b = read_f32(C);
    w.mlp_fc_w = read_q8(4 * C, C);
    w.mlp_fc_b = read_f32(4 * C);
    w.mlp_proj_w = read_q8(C, 4 * C);
    w.mlp_proj_b = read_f32(C);
  }
  m.ln_f_w = read_f32(C);
  m.ln_f_b = read_f32(C);
  fclose(f);

  // Count sizes for reporting
  size_t q8_bytes = (size_t)V * C + (size_t)S * C;
  for (int l = 0; l < L; l++)
    q8_bytes += (size_t)3 * C * C + (size_t)C * C + (size_t)4 * C * C +
                (size_t)C * 4 * C;
  size_t scale_bytes = (V + S) * sizeof(float);
  for (int l = 0; l < L; l++)
    scale_bytes += (3 * C + C + 4 * C + C) * sizeof(float);
  printf("[GPT2-Q8] INT8 weights: %.1f MB, scales: %.1f MB\n", q8_bytes / 1e6,
         scale_bytes / 1e6);

  // Activation buffers
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
  m.gelu_tmp = (float *)calloc((size_t)S * 4 * C, sizeof(float));
  m.gelu_orig = (float *)calloc((size_t)S * 4 * C, sizeof(float));

  // Dequantization scratch: largest matrix is wte (V × C) or mlp (4C × C)
  size_t max_matrix = std::max((size_t)V * C, (size_t)4 * C * C);
  m.dequant_scratch = (float *)calloc(max_matrix, sizeof(float));
  printf("[GPT2-Q8] Dequant scratch: %.1f MB\n",
         max_matrix * sizeof(float) / 1e6);

  // KV cache
  m.k_cache.resize(L);
  m.v_cache.resize(L);
  for (int l = 0; l < L; l++) {
    m.k_cache[l] = (float *)calloc((size_t)S * C, sizeof(float));
    m.v_cache[l] = (float *)calloc((size_t)S * C, sizeof(float));
  }
  m.cache_len = 0;

  printf("[GPT2-Q8] Model loaded! (INT8 + KV Cache + SIMD)\n");
  return m;
}

void free_gpt2(GPT2 &m) {
  // Free Q8 weights
  free(m.wte_q.data);
  free(m.wte_q.scales);
  free(m.wpe_q.data);
  free(m.wpe_q.scales);
  for (auto &w : m.layers) {
    free(w.ln_1_w);
    free(w.ln_1_b);
    free(w.attn_w.data);
    free(w.attn_w.scales);
    free(w.attn_b);
    free(w.attn_proj_w.data);
    free(w.attn_proj_w.scales);
    free(w.attn_proj_b);
    free(w.ln_2_w);
    free(w.ln_2_b);
    free(w.mlp_fc_w.data);
    free(w.mlp_fc_w.scales);
    free(w.mlp_fc_b);
    free(w.mlp_proj_w.data);
    free(w.mlp_proj_w.scales);
    free(w.mlp_proj_b);
  }
  free(m.ln_f_w);
  free(m.ln_f_b);
  // Free activations
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
  free(m.dequant_scratch);
  for (auto *p : m.k_cache)
    free(p);
  for (auto *p : m.v_cache)
    free(p);
}
