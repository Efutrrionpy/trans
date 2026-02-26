from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class TokenizerInterface(ABC):
    """
    Abstract Base Class for Tokenizers.
    All backends (HF, Manual) must implement this interface.
    """
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert a string to a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        pass
    
    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        pass

class TransformerEngineInterface(ABC):
    """
    Abstract Base Class for Transformer Models.
    All backends (HF, Torch Native, NumPy Manual) must implement this interface.
    """
    @abstractmethod
    def forward(self, input_ids: List[int]) -> np.ndarray:
        """
        Run one forward pass.
        
        Args:
            input_ids: List of token IDs representing the current context.
            
        Returns:
            logits: A numpy array of shape (vocab_size,) representing the logits 
                    for the next token prediction.
        """
        pass
