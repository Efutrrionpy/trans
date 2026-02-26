from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import numpy as np
from typing import List
from core.interfaces import TokenizerInterface, TransformerEngineInterface

class HfTokenizer(TokenizerInterface):
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

class HfEngine(TransformerEngineInterface):
    def __init__(self, weights_path: str = "weights/model.bin"):
        from core.weight_loader import load_weights
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[HfEngine] Using device: {self.device}")

        config, weights = load_weights(weights_path)

        # Build HF GPT2Config from our config dict
        hf_config = GPT2Config(
            vocab_size=config["vocab_size"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            n_layer=config["n_layer"],
            n_positions=config["max_seq_len"],
        )

        # Create empty model, then load weights
        self.model = GPT2LMHeadModel(hf_config)

        # Convert numpy weight dict → HF state_dict
        # model.bin stores Conv1D weights as (out, in), HF expects (in, out)
        sd = {}
        sd["transformer.wte.weight"] = torch.from_numpy(weights["wte"])
        sd["transformer.wpe.weight"] = torch.from_numpy(weights["wpe"])
        sd["transformer.ln_f.weight"] = torch.from_numpy(weights["ln_f_w"])
        sd["transformer.ln_f.bias"]   = torch.from_numpy(weights["ln_f_b"])
        sd["lm_head.weight"] = sd["transformer.wte.weight"]  # weight tying

        for i, bw in enumerate(weights["blocks"]):
            p = f"transformer.h.{i}"
            sd[f"{p}.ln_1.weight"] = torch.from_numpy(bw["ln_1_w"])
            sd[f"{p}.ln_1.bias"]   = torch.from_numpy(bw["ln_1_b"])
            # Conv1D weights: transpose BACK from (out, in) to (in, out)
            sd[f"{p}.attn.c_attn.weight"] = torch.from_numpy(bw["attn_w"].T.copy())
            sd[f"{p}.attn.c_attn.bias"]   = torch.from_numpy(bw["attn_b"])
            sd[f"{p}.attn.c_proj.weight"] = torch.from_numpy(bw["attn_proj_w"].T.copy())
            sd[f"{p}.attn.c_proj.bias"]   = torch.from_numpy(bw["attn_proj_b"])
            sd[f"{p}.ln_2.weight"] = torch.from_numpy(bw["ln_2_w"])
            sd[f"{p}.ln_2.bias"]   = torch.from_numpy(bw["ln_2_b"])
            sd[f"{p}.mlp.c_fc.weight"]   = torch.from_numpy(bw["mlp_fc_w"].T.copy())
            sd[f"{p}.mlp.c_fc.bias"]     = torch.from_numpy(bw["mlp_fc_b"])
            sd[f"{p}.mlp.c_proj.weight"] = torch.from_numpy(bw["mlp_proj_w"].T.copy())
            sd[f"{p}.mlp.c_proj.bias"]   = torch.from_numpy(bw["mlp_proj_b"])

        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()
        print(f"[HfEngine] Loaded from model.bin!")

    def forward(self, input_ids: List[int]) -> np.ndarray:
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return outputs.logits[0, -1, :].cpu().numpy()

