"""
Export GPT-2 weights and vocab to shared binary files.
All backends (Python + C++) read from these files.

Usage: python export_weights.py [--output_dir weights]
"""
import os
import struct
import argparse
import numpy as np


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = list(bs)
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="weights", help="Output directory")
    parser.add_argument("--model_name", default="gpt2", help="HuggingFace model name")
    args = parser.parse_args()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Export Model Weights ---
    print(f"Loading {args.model_name}...")
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    config = model.config
    sd = model.state_dict()

    model_path = os.path.join(args.output_dir, "model.bin")
    with open(model_path, "wb") as f:
        f.write(struct.pack("iiiii",
            config.n_layer, config.n_head, config.n_embd,
            config.vocab_size, config.n_positions))

        def write(key, transpose=False):
            t = sd[key].cpu().numpy().astype(np.float32)
            if transpose:
                t = t.T
            f.write(np.ascontiguousarray(t).tobytes())

        write("transformer.wte.weight")
        write("transformer.wpe.weight")
        for i in range(config.n_layer):
            p = f"transformer.h.{i}"
            write(f"{p}.ln_1.weight");       write(f"{p}.ln_1.bias")
            write(f"{p}.attn.c_attn.weight", True); write(f"{p}.attn.c_attn.bias")
            write(f"{p}.attn.c_proj.weight", True);  write(f"{p}.attn.c_proj.bias")
            write(f"{p}.ln_2.weight");       write(f"{p}.ln_2.bias")
            write(f"{p}.mlp.c_fc.weight", True);     write(f"{p}.mlp.c_fc.bias")
            write(f"{p}.mlp.c_proj.weight", True);   write(f"{p}.mlp.c_proj.bias")
        write("transformer.ln_f.weight")
        write("transformer.ln_f.bias")

    print(f"Model: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")

    # --- Export Vocab ---
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    vocab = tokenizer.encoder
    inv_vocab = {v: k for k, v in vocab.items()}

    vocab_path = os.path.join(args.output_dir, "vocab.bin")
    with open(vocab_path, "wb") as f:
        f.write(struct.pack("i", len(vocab)))
        for i in range(len(vocab)):
            token_str = inv_vocab.get(i, "")
            raw = bytes([byte_decoder[c] for c in token_str])
            f.write(struct.pack("i", len(raw)))
            f.write(raw)

    print(f"Vocab: {vocab_path} ({len(vocab)} tokens)")
    print(f"\nDone! All backends can now load from: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
