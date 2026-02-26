"""
PyTorch Native Backend: Hand-Crafted GPT-2 (Inference Only)
===========================================================
Every matrix operation is written explicitly using PyTorch primitives.
No `transformers` library code runs during inference.

GPT-2 Small Config:
    vocab_size  = 50257
    n_embd      = 768
    n_head      = 12
    n_layer     = 12
    max_seq_len = 1024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from core.interfaces import TokenizerInterface, TransformerEngineInterface

class ManualLayerNorm(nn.Module):
    """
    Layer Normalization.
    公式: y = (x - mean) / sqrt(var + eps) * weight + bias
    """
    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    步驟:
        1. x -> Linear -> [Q, K, V]  (一次算出三個)
        2. Q, K, V reshape 成 (B, n_head, T, head_dim)
        3. Attention = softmax( (Q @ K^T) / sqrt(d_k) + causal_mask )
        4. Output = Attention @ V
        5. Output -> Linear -> projected output
    """
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Q, K, V 合併成一個 Linear (效率更高)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Sequence Length, Embedding Dim

        # --- Step 1: Compute Q, K, V ---
        qkv = self.c_attn(x)                                  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)               # 3 x (B, T, C)

        # --- Step 2: Reshape to multi-head ---
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- Step 3: Scaled Dot-Product Attention ---
        # att = (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, n_head, T, T)

        # Causal Mask: 上三角全部設為 -inf，讓 softmax 後變成 0
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax
        att = F.softmax(att, dim=-1)                           # (B, n_head, T, T)

        # --- Step 4: Weighted sum of Values ---
        y = att @ v                                            # (B, n_head, T, head_dim)

        # --- Step 5: Concatenate heads and project ---
        y = y.transpose(1, 2).contiguous().view(B, T, C)      # (B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    """
    Feed-Forward Network.
    公式: x -> Linear(768->3072) -> GELU -> Linear(3072->768)
    GPT-2 uses approximate GELU (tanh approximation).
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)                              # (B, T, 4*C)
        x = F.gelu(x, approximate='tanh')             # GELU activation
        x = self.c_proj(x)                             # (B, T, C)
        return x

class TransformerBlock(nn.Module):
    """
    一個完整的 Transformer Block (Pre-LayerNorm 架構, GPT-2 Style).
    
    流程:
        x  ->  LN1  ->  Attention  ->  + residual  ->  LN2  ->  MLP  ->  + residual
    """
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln_1 = ManualLayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = ManualLayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + Residual Connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ManualGPT2(nn.Module):
    """
    Complete GPT-2 Model (Inference Only).
    
    Architecture:
        Token IDs -> Token Embedding + Position Embedding
                  -> 12 x TransformerBlock
                  -> Final LayerNorm
                  -> LM Head (tied with Token Embedding)
                  -> Logits
    """
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12, max_seq_len=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)       # Token Embeddings
        self.wpe = nn.Embedding(max_seq_len, n_embd)      # Position Embeddings
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = ManualLayerNorm(n_embd)                # Final LayerNorm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) tensor of token IDs
        Returns:
            logits: (B, T, vocab_size) tensor of logits
        """
        B, T = input_ids.size()
        device = input_ids.device

        # --- Embeddings ---
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        tok_emb = self.wte(input_ids)   # (B, T, n_embd)
        pos_emb = self.wpe(pos)         # (1, T, n_embd)
        x = tok_emb + pos_emb          # (B, T, n_embd)

        # --- Transformer Blocks ---
        for block in self.blocks:
            x = block(x)

        # --- Final LayerNorm ---
        x = self.ln_f(x)               # (B, T, n_embd)

        # --- LM Head (Weight Tying: reuse wte.weight) ---
        logits = x @ self.wte.weight.T  # (B, T, vocab_size)

        return logits

    @classmethod
    def from_weights(cls, weights_path: str = "weights/model.bin"):
        """
        Load weights from shared model.bin (same file used by all backends).
        Converts numpy arrays from weight_loader into PyTorch state_dict.
        """
        from core.weight_loader import load_weights

        print(f"[ManualGPT2] Loading weights from '{weights_path}'...")
        config, weights = load_weights(weights_path)

        model = cls(
            vocab_size=config["vocab_size"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            n_layer=config["n_layer"],
            max_seq_len=config["max_seq_len"],
        )

        # Convert numpy weight dict → PyTorch state_dict
        sd = {}
        sd["wte.weight"] = torch.from_numpy(weights["wte"])
        sd["wpe.weight"] = torch.from_numpy(weights["wpe"])
        sd["ln_f.weight"] = torch.from_numpy(weights["ln_f_w"])
        sd["ln_f.bias"] = torch.from_numpy(weights["ln_f_b"])

        for i, bw in enumerate(weights["blocks"]):
            p = f"blocks.{i}"
            sd[f"{p}.ln_1.weight"] = torch.from_numpy(bw["ln_1_w"])
            sd[f"{p}.ln_1.bias"]   = torch.from_numpy(bw["ln_1_b"])
            sd[f"{p}.attn.c_attn.weight"] = torch.from_numpy(bw["attn_w"])
            sd[f"{p}.attn.c_attn.bias"]   = torch.from_numpy(bw["attn_b"])
            sd[f"{p}.attn.c_proj.weight"] = torch.from_numpy(bw["attn_proj_w"])
            sd[f"{p}.attn.c_proj.bias"]   = torch.from_numpy(bw["attn_proj_b"])
            sd[f"{p}.ln_2.weight"] = torch.from_numpy(bw["ln_2_w"])
            sd[f"{p}.ln_2.bias"]   = torch.from_numpy(bw["ln_2_b"])
            sd[f"{p}.mlp.c_fc.weight"]   = torch.from_numpy(bw["mlp_fc_w"])
            sd[f"{p}.mlp.c_fc.bias"]     = torch.from_numpy(bw["mlp_fc_b"])
            sd[f"{p}.mlp.c_proj.weight"] = torch.from_numpy(bw["mlp_proj_w"])
            sd[f"{p}.mlp.c_proj.bias"]   = torch.from_numpy(bw["mlp_proj_b"])

        model.load_state_dict(sd)
        print(f"[ManualGPT2] All weights loaded from model.bin!")
        return model

class TorchNativeTokenizer(TokenizerInterface):
    """Wraps GPT2Tokenizer. BPE tokenization is not the focus of this project."""
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


class TorchNativeEngine(TransformerEngineInterface):
    """Wraps ManualGPT2, implements TransformerEngineInterface."""
    def __init__(self, weights_path: str = "weights/model.bin"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[TorchNativeEngine] Using device: {self.device}")

        self.model = ManualGPT2.from_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, input_ids: List[int]) -> np.ndarray:
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)

        return logits[0, -1, :].cpu().numpy()

