"""
NumPy Backend: Hand-Crafted GPT-2 (Inference Only)
"""

import math
import numpy as np
from typing import List

from core.interfaces import TokenizerInterface, TransformerEngineInterface


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:

    x_stable = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x: np.ndarray) -> np.ndarray:

    return 0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:

    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return weight * x_norm + bias

def mlp(x: np.ndarray,
        c_fc_w: np.ndarray, c_fc_b: np.ndarray,
        c_proj_w: np.ndarray, c_proj_b: np.ndarray) -> np.ndarray:

    x = linear(x, c_fc_w, c_fc_b)
    x = gelu(x)
    x = linear(x, c_proj_w, c_proj_b)
    return x


def causal_self_attention(
        x: np.ndarray,
        c_attn_w: np.ndarray, c_attn_b: np.ndarray,
        c_proj_w: np.ndarray, c_proj_b: np.ndarray,
        n_head: int) -> np.ndarray:

    B, T, C = x.shape
    head_dim = C // n_head

    qkv = linear(x, c_attn_w, c_attn_b)
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, n_head, head_dim)
    q = q.transpose(0, 2, 1, 3)
    k, v = k.reshape(B, T, n_head, head_dim), v.reshape(B, T, n_head, head_dim)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    mask = np.tril(np.ones((T, T)))
    att = np.where(mask == 0, -np.inf, att)
    att = softmax(att, axis=-1)
    y = att @ v
    y = y.transpose(0, 2, 1, 3)
    y = y.reshape(B, T, C)
    y = linear(y, c_proj_w, c_proj_b)
    return y

def transformer_block(x: np.ndarray, block_weights: dict, n_head: int) -> np.ndarray:

    # Pre-LN + Attention + Residual
    x = x + causal_self_attention(
        layer_norm(x, block_weights["ln_1_w"], block_weights["ln_1_b"]),
        block_weights["attn_w"], block_weights["attn_b"],
        block_weights["attn_proj_w"], block_weights["attn_proj_b"],
        n_head
    )
    # Pre-LN + MLP + Residual
    x = x + mlp(
        layer_norm(x, block_weights["ln_2_w"], block_weights["ln_2_b"]),
        block_weights["mlp_fc_w"], block_weights["mlp_fc_b"],
        block_weights["mlp_proj_w"], block_weights["mlp_proj_b"]
    )
    return x


def gpt2_forward(input_ids: np.ndarray, weights: dict, n_head: int) -> np.ndarray:
    """
    GPT-2 完整的 Forward Pass.
    Token IDs → Embeddings → 12 × Block → Final LN → LM Head → Logits
    """
    B, T = input_ids.shape

    # Embeddings
    tok_emb = weights["wte"][input_ids]                           # (B, T, C)
    pos_emb = weights["wpe"][np.arange(T)]                        # (T, C)
    x = tok_emb + pos_emb                                         # (B, T, C)

    # Transformer Blocks
    for block_w in weights["blocks"]:
        x = transformer_block(x, block_w, n_head)

    # Final LayerNorm
    x = layer_norm(x, weights["ln_f_w"], weights["ln_f_b"])       # (B, T, C)

    # LM Head (weight tying: 重用 wte)
    logits = x @ weights["wte"].T                                 # (B, T, vocab_size)

    return logits

class NumpyTokenizer(TokenizerInterface):
    """Wraps GPT2Tokenizer (BPE tokenization is not the focus)."""
    def __init__(self, model_name: str = "gpt2"):
        from transformers import GPT2Tokenizer
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id


class NumpyEngine(TransformerEngineInterface):
    """Wraps the pure-NumPy GPT-2 model."""
    def __init__(self, weights_path: str = "weights/model.bin"):
        from core.weight_loader import load_weights
        self.config, self.weights = load_weights(weights_path)
        print(f"[NumpyEngine] Ready (pure NumPy, no GPU)")

    def forward(self, input_ids: List[int]) -> np.ndarray:
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            ids = np.array([input_ids], dtype=np.int64)
            logits = gpt2_forward(ids, self.weights, self.config["n_head"])
            return logits[0, -1, :]

