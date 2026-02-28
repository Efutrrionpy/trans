import argparse
import sys
import os
import numpy as np

# Ensure the current directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.interfaces import TokenizerInterface, TransformerEngineInterface
from core.sampler import sample_next_token

# =============================================================================
# Registry: All Available Backends
# =============================================================================
AVAILABLE_BACKENDS = ["hf", "torch_native", "numpy"]

def get_backend(backend_name: str, weights_path: str = "weights/model.bin") -> tuple[TokenizerInterface, TransformerEngineInterface]:
    if backend_name == "hf":
        from backends.hf_backend import HfTokenizer, HfEngine
        return HfTokenizer(), HfEngine(weights_path=weights_path)
    elif backend_name == "torch_native":
        from backends.torch_native_backend import TorchNativeTokenizer, TorchNativeEngine
        return TorchNativeTokenizer(), TorchNativeEngine(weights_path=weights_path)
    elif backend_name == "numpy":
        from backends.numpy_backend import NumpyTokenizer, NumpyEngine
        return NumpyTokenizer(), NumpyEngine(weights_path=weights_path)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

# =============================================================================
# Single Backend: Generation Loop
# =============================================================================
def generate(engine, tokenizer, input_ids, max_length, temperature):
    """Run generation and return (generated_ids, all_logits)."""
    generated_ids = list(input_ids)
    all_logits = []

    for _ in range(max_length):
        logits = engine.forward(generated_ids)
        all_logits.append(logits)

        next_token = sample_next_token(logits, temperature=temperature)
        generated_ids.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return generated_ids, all_logits

# =============================================================================
# Compare Mode
# =============================================================================
def run_compare(prompt, max_length):
    """Load all backends (Python + C++), run greedy decoding, compare outputs and speed."""
    import time
    import subprocess

    print("=" * 70)
    print("  FULL BENCHMARK — All Backends (temperature=0, greedy)")
    print("=" * 70)
    print(f"  Prompt: {prompt}")
    print(f"  Max New Tokens: {max_length}")
    print("=" * 70)

    # --- 1. Run Python backends ---
    results = {}  # name -> {"text": str, "time": float, "tokens": int}

    # Get input IDs from the first backend's tokenizer
    first_tok, _ = get_backend(AVAILABLE_BACKENDS[0])
    input_ids = first_tok.encode(prompt)
    ids_str = " ".join(str(i) for i in input_ids)
    print(f"\n  Input IDs: {input_ids}")

    for name in AVAILABLE_BACKENDS:
        print(f"\n>>> Loading backend: {name}")
        tokenizer, engine = get_backend(name)
        print(f">>> Running {name}...")
        t0 = time.perf_counter()
        gen_ids, _ = generate(engine, tokenizer, input_ids, max_length, temperature=0)
        elapsed = time.perf_counter() - t0
        text = tokenizer.decode(gen_ids)
        n_tok = len(gen_ids) - len(input_ids)
        results[name] = {"text": text, "time": elapsed, "tokens": n_tok}
        print(f"    Output: {text}")
        print(f"    {n_tok} tokens in {elapsed:.3f}s ({n_tok/elapsed:.1f} tok/sec)")

    # --- 2. Run C++ backends via subprocess ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(script_dir, "cpp")
    weights_dir = os.path.join(script_dir, "weights")
    model_bin = os.path.join(weights_dir, "model.bin")
    vocab_bin = os.path.join(weights_dir, "vocab.bin")

    cpp_backends = [
        #("cpp_naive", os.path.join(cpp_dir, "gpt2"), model_bin),
        ("cpp_simd",  os.path.join(cpp_dir, "gpt2_fast"), model_bin),
        ("cpp_kv",    os.path.join(cpp_dir, "gpt2_kv"),   model_bin),
        ("cpp_kv2",   os.path.join(cpp_dir, "gpt2_kv2"),  model_bin),
        ("cpp_q8",    os.path.join(cpp_dir, "gpt2_q8"),
                      os.path.join(weights_dir, "model_q8.bin")),
    ]

    for name, exe_path, cmodel in cpp_backends:
        if not os.path.exists(exe_path):
            print(f"\n>>> Skipping {name}: {exe_path} not found (run 'cd cpp && make')")
            continue
        if not os.path.exists(cmodel):
            print(f"\n>>> Skipping {name}: {cmodel} not found")
            continue

        print(f"\n>>> Running {name}...")
        cmd = [
            exe_path,
            "--ids", ids_str,
            "--max_length", str(max_length),
            "--temperature", "0",
            "--model", cmodel,
            "--vocab", vocab_bin,
            "--machine",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            # Parse MACHINE_IDS and MACHINE_TIME lines
            cpp_ids = None
            cpp_time = None
            for line in proc.stdout.strip().split("\n"):
                if line.startswith("MACHINE_IDS:"):
                    cpp_ids = list(map(int, line[len("MACHINE_IDS:"):].strip().split()))
                elif line.startswith("MACHINE_TIME:"):
                    cpp_time = float(line[len("MACHINE_TIME:"):].strip())

            if cpp_ids is not None and cpp_time is not None:
                text = first_tok.decode(cpp_ids)
                n_tok = len(cpp_ids) - len(input_ids)
                results[name] = {"text": text, "time": cpp_time, "tokens": n_tok}
                print(f"    Output: {text}")
                print(f"    {n_tok} tokens in {cpp_time:.3f}s ({n_tok/cpp_time:.1f} tok/sec)")
            else:
                print(f"    ⚠️ Failed to parse output")
                if proc.stderr:
                    print(f"    stderr: {proc.stderr[:200]}")
        except Exception as e:
            print(f"    ⚠️ Error: {e}")

    # --- 3. Summary ---
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    ref_text = results[AVAILABLE_BACKENDS[0]]["text"]
    all_match = True

    print(f"\n  {'Backend':<16} {'tok/sec':>10} {'Time (s)':>10} {'Match':>7}  Output")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*7}  {'-'*30}")

    for name in list(results.keys()):
        r = results[name]
        speed = r["tokens"] / r["time"] if r["time"] > 0 else 0
        match = r["text"] == ref_text
        if not match:
            all_match = False
        icon = "✅" if match else "❌"
        # Truncate output for display
        out_short = r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"]
        print(f"  {name:<16} {speed:>10.1f} {r['time']:>10.3f} {icon:>7}  {out_short}")

    print()
    if all_match:
        print("  ✅ ALL BACKENDS PRODUCE IDENTICAL OUTPUT")
    else:
        print("  ⚠️  SOME DIFFERENCES DETECTED")
    print("=" * 70)

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Modular Transformer Inference Engine")
    parser.add_argument("--backend", type=str, default="hf", help="Backend to use: hf, torch_native")
    parser.add_argument("--prompt", type=str, default="The quick brown fox jumps over the lazy dog", help="Input prompt")
    parser.add_argument("--max_length", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--compare", action="store_true", help="Run all backends and compare outputs")

    args = parser.parse_args()

    if args.compare:
        run_compare(args.prompt, args.max_length)
        return

    # --- Normal single-backend mode ---
    print(f"Using Backend: {args.backend}")
    print(f"Prompt: {args.prompt}")

    tokenizer, engine = get_backend(args.backend)
    input_ids = tokenizer.encode(args.prompt)
    print(f"Input IDs: {input_ids}")

    print("\n--- Generating ---")
    generated_ids, _ = generate(engine, tokenizer, input_ids, args.max_length, args.temperature)

    # Stream is not used in refactored generate(), so print final result
    final_text = tokenizer.decode(generated_ids)
    print(f"{final_text}")
    print("--- Done ---")

if __name__ == "__main__":
    main()
 