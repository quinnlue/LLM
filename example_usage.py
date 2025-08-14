"""
Example usage of the GPT-1 project with the new import structure.
"""

# Method 1: Import the entire package
import gpt1

# Now you can access components directly
model = gpt1.Model(
    vocab_size=50000,
    d_model=768,
    max_seq_len=512,
    pad_idx=0,
    n_heads=12,
    transformer_depth=12,
    checkpoint_interval_seconds=3600,
    train_dir="data/train",
    validation_dir="data/validation", 
    checkpoint_dir="checkpoints",
    epochs=10,
    mini_batch_per_step=4
)

# Method 2: Import specific modules
from gpt1.training import Model
from gpt1.preprocess import DataLoader, Dataset
from gpt1.tokenizer import tokenizer

# Method 3: Import specific classes
from gpt1 import Model, DataLoader, Dataset, tokenizer

print("GPT-1 project imported successfully!")
print(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")
