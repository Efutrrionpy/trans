"""
Shared Weight Loader: reads model.bin exported by export_weights.py.
All backends (HF, Torch Native, NumPy, C++) use this same binary format.

Binary format:
  Header:  [n_layer, n_head, n_embd, vocab_size, max_seq_len] as 5 x int32
  Weights: wte, wpe, per-layer blocks, ln_f  (all float32, row-major)
  Conv1D weights are stored TRANSPOSED as (out_features, in_features).
"""
import struct
import numpy as np


def load_weights(model_bin_path: str) -> tuple[dict, dict]:
    """
    Load GPT-2 weights from model.bin.

    Returns:
        (config, weights) where:
        - config: dict with n_layer, n_head, n_embd, vocab_size, max_seq_len
        - weights: dict matching numpy_backend's format:
            wte, wpe, blocks[{ln_1_w, ln_1_b, attn_w, ...}], ln_f_w, ln_f_b
    """
    with open(model_bin_path, "rb") as f:
        # Header
        hdr = struct.unpack("iiiii", f.read(20))
        config = {
            "n_layer": hdr[0], "n_head": hdr[1], "n_embd": hdr[2],
            "vocab_size": hdr[3], "max_seq_len": hdr[4],
        }
        C, V, S, L = config["n_embd"], config["vocab_size"], config["max_seq_len"], config["n_layer"]

        def read(shape):
            n = 1
            for s in shape:
                n *= s
            return np.frombuffer(f.read(n * 4), dtype=np.float32).copy().reshape(shape)

        weights = {
            "wte": read((V, C)),
            "wpe": read((S, C)),
            "blocks": [],
        }

        for _ in range(L):
            weights["blocks"].append({
                "ln_1_w": read((C,)),       "ln_1_b": read((C,)),
                "attn_w": read((3*C, C)),   "attn_b": read((3*C,)),
                "attn_proj_w": read((C, C)),"attn_proj_b": read((C,)),
                "ln_2_w": read((C,)),       "ln_2_b": read((C,)),
                "mlp_fc_w": read((4*C, C)), "mlp_fc_b": read((4*C,)),
                "mlp_proj_w": read((C, 4*C)), "mlp_proj_b": read((C,)),
            })

        weights["ln_f_w"] = read((C,))
        weights["ln_f_b"] = read((C,))

    print(f"[weight_loader] Loaded from {model_bin_path} "
          f"({L} layers, {C}d, {config['n_head']} heads)")
    return config, weights
