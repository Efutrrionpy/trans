"""
Quantize model.bin → model_q8.bin (INT8 per-channel, absmax)

Weight matrices: int8 + float32 per-row scale
Biases & LayerNorm: float32 (unchanged)

Usage: python quantize_weights.py [--input weights/model.bin] [--output weights/model_q8.bin]
"""

import struct
import numpy as np
import argparse
import os


def quantize_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row absmax quantization: w ≈ q * scale (per row).
    w:      float32 [out_features, in_features]
    Returns (q: int8, scales: float32[out_features])
    """
    row_max = np.max(np.abs(w), axis=1)             # [out_features]
    row_max = np.where(row_max == 0, 1.0, row_max)  # avoid div-by-zero
    scales = row_max / 127.0                         # [out_features]
    q = np.round(w / scales[:, None]).astype(np.int8)
    return q, scales.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="weights/model.bin")
    parser.add_argument("--output", default="weights/model_q8.bin")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        hdr = struct.unpack("5i", f.read(20))
        n_layer, n_head, n_embd, vocab_size, max_seq_len = hdr
        C = n_embd
        V = vocab_size
        S = max_seq_len

        def read_f32(count):
            return np.frombuffer(f.read(count * 4), dtype=np.float32).copy()

        wte = read_f32(V * C).reshape(V, C)
        wpe = read_f32(S * C).reshape(S, C)

        layers = []
        for _ in range(n_layer):
            layer = {
                "ln_1_w": read_f32(C),
                "ln_1_b": read_f32(C),
                "attn_w": read_f32(3*C*C).reshape(3*C, C),
                "attn_b": read_f32(3*C),
                "attn_proj_w": read_f32(C*C).reshape(C, C),
                "attn_proj_b": read_f32(C),
                "ln_2_w": read_f32(C),
                "ln_2_b": read_f32(C),
                "mlp_fc_w": read_f32(4*C*C).reshape(4*C, C),
                "mlp_fc_b": read_f32(4*C),
                "mlp_proj_w": read_f32(4*C*C).reshape(C, 4*C),
                "mlp_proj_b": read_f32(C),
            }
            layers.append(layer)

        ln_f_w = read_f32(C)
        ln_f_b = read_f32(C)

    # --- Quantize weight matrices ---
    print(f"Quantizing {args.input}...")
    print(f"  Config: {n_layer} layers, {n_head} heads, {C} dim, {V} vocab")

    # Track quantization error
    total_params = 0
    total_error = 0.0

    def quantize_and_report(name, w):
        nonlocal total_params, total_error
        q, scales = quantize_per_channel(w)
        # Measure quantization error
        w_deq = q.astype(np.float32) * scales[:, None]
        mse = np.mean((w - w_deq) ** 2)
        max_err = np.max(np.abs(w - w_deq))
        total_params += w.size
        total_error += np.sum((w - w_deq) ** 2)
        print(f"  {name:20s}: {w.shape} → int8, MSE={mse:.2e}, max_err={max_err:.4f}")
        return q, scales

    wte_q, wte_s = quantize_and_report("wte", wte)
    wpe_q, wpe_s = quantize_and_report("wpe", wpe)

    layer_q = []
    for i, layer in enumerate(layers):
        lq = {}
        for key in ["attn_w", "attn_proj_w", "mlp_fc_w", "mlp_proj_w"]:
            lq[key + "_q"], lq[key + "_s"] = quantize_and_report(
                f"layer[{i}].{key}", layer[key])
        # Keep biases and LN params as-is
        for key in ["ln_1_w", "ln_1_b", "attn_b", "attn_proj_b",
                     "ln_2_w", "ln_2_b", "mlp_fc_b", "mlp_proj_b"]:
            lq[key] = layer[key]
        layer_q.append(lq)

    rmse = np.sqrt(total_error / total_params)
    print(f"\n  Overall RMSE: {rmse:.6f} ({total_params/1e6:.1f}M quantized params)")

    # --- Write model_q8.bin ---
    # Format:
    #   Header: 5 × int32 (same as model.bin)
    #   For quantized tensors: scales(float32 × out_dim) + data(int8 × out_dim × in_dim)
    #   For FP32 tensors: data(float32 × dim)
    # Order matches model.bin exactly.

    with open(args.output, "wb") as f:
        f.write(struct.pack("5i", *hdr))

        def write_q8(q, scales):
            f.write(scales.tobytes())           # float32 scales
            f.write(q.tobytes())                # int8 data

        def write_f32(arr):
            f.write(arr.astype(np.float32).tobytes())

        write_q8(wte_q, wte_s)
        write_q8(wpe_q, wpe_s)

        for i in range(n_layer):
            lq = layer_q[i]
            write_f32(lq["ln_1_w"])
            write_f32(lq["ln_1_b"])
            write_q8(lq["attn_w_q"], lq["attn_w_s"])
            write_f32(lq["attn_b"])
            write_q8(lq["attn_proj_w_q"], lq["attn_proj_w_s"])
            write_f32(lq["attn_proj_b"])
            write_f32(lq["ln_2_w"])
            write_f32(lq["ln_2_b"])
            write_q8(lq["mlp_fc_w_q"], lq["mlp_fc_w_s"])
            write_f32(lq["mlp_fc_b"])
            write_q8(lq["mlp_proj_w_q"], lq["mlp_proj_w_s"])
            write_f32(lq["mlp_proj_b"])

        write_f32(ln_f_w)
        write_f32(ln_f_b)

    orig_size = os.path.getsize(args.input) / 1024 / 1024
    new_size = os.path.getsize(args.output) / 1024 / 1024
    ratio = orig_size / new_size
    print(f"\n  {args.input}: {orig_size:.1f} MB")
    print(f"  {args.output}: {new_size:.1f} MB")
    print(f"  Compression: {ratio:.2f}x")
    print("Done!")


if __name__ == "__main__":
    main()
