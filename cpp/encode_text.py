"""Helper: tokenize text using GPT-2 tokenizer for C++ inference."""
import sys
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello world"
ids = tokenizer.encode(text)
print(" ".join(str(i) for i in ids))
