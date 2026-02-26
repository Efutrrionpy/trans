import numpy as np

def sample_next_token(logits: np.ndarray, temperature: float = 1.0, top_k: int = 0) -> int:
    """
    Samples the next token from the logits.
    
    Args:
        logits: A numpy array of shape (vocab_size,).
        temperature: Controls randomness. Lower is more deterministic.
        top_k: Limit sampling to the top k most likely tokens. 0 means disable.
        
    Returns:
        The ID of the next token.
    """
    # 1. Apply Temperature
    if temperature == 0:
        return np.argmax(logits)
    
    logits = logits / temperature
    
    # 2. Subtract max for numerical stability (softmax trick)
    # exp(x - max) / sum(exp(x - max))
    logits = logits - np.max(logits)
    
    # 3. Apply Softmax
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    # 4. Top-K filtering (Simple implementation)
    if top_k > 0:
        # Get indices of the top k probabilities
        top_k_indices = np.argsort(probs)[-top_k:]
        # Mask out everything else
        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        probs[mask] = 0
        # Re-normalize
        probs = probs / np.sum(probs)
        
    # 5. Sample from the distribution
    next_token = np.random.choice(len(probs), p=probs)
    return int(next_token)
