import sys
import os

import dlx as dlx
from dlx import xp
from gpt1.training.model import Model
from dlx.nn.optim import AdamW
import numpy as np


class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.is_training = False  # Set to eval mode
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer, vocab_size, d_model, max_seq_len, 
                       pad_idx, n_heads, transformer_depth, mlp_ratio=4):
        """
        Load model from checkpoint directory.
        
        Args:
            checkpoint_path: Path to checkpoint directory containing model/*.npy files
            tokenizer: Tokenizer object (from tokenizers import Tokenizer)
            vocab_size, d_model, max_seq_len, pad_idx, n_heads, transformer_depth: Model architecture parameters
            mlp_ratio: MLP expansion ratio (default: 4)
        """
        # Create model with same architecture
        model = Model(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            n_heads=n_heads,
            transformer_depth=transformer_depth,
            checkpoint_interval_seconds=0,  # Not used for inference
            train_dir="",  # Not used for inference
            validation_dir="",  # Not used for inference
            checkpoint_dir="",  # Not used for inference
            epochs=0,  # Not used for inference
            mini_batch_per_step=1,  # Not used for inference
            mlp_ratio=mlp_ratio
        )

        print(model)

        print(model.num_parameters)
        for name, param in model.parameters().items():
            print(name, param.shape)

        print(f"Loading model from {checkpoint_path}")

        optim = AdamW(model.parameters(), precision=(np.float32, np.float32))
        optim.load_state(checkpoint_path)
        
        return cls(model, tokenizer)
    
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Generate text continuation from a prompt.
        
        Args:
            prompt: String input text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated text as string
        """
        # Encode prompt
        encoded = self.tokenizer.encode(prompt)
        idx = xp.array([encoded.ids], dtype=xp.int32)
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # Get predictions for the last token
            logits = self.model.forward(idx)  # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]  # (1, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Get logits as numpy array
            logits_np = xp.asnumpy(logits.data[0])
            
            # Apply top-k if specified
            if top_k is not None:
                top_k_idx = np.argpartition(logits_np, -top_k)[-top_k:]
                mask = np.full_like(logits_np, -float('inf'))
                mask[top_k_idx] = logits_np[top_k_idx]
                logits_np = mask
            
            # Sample from distribution
            probs = np.exp(logits_np - np.max(logits_np))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            
            # Append to sequence
            next_token_array = xp.array([[next_token]], dtype=xp.int32)
            idx = xp.concatenate([idx, next_token_array], axis=1)
            
            # Stop if we hit EOS token (if your tokenizer has one)
            # Uncomment and adjust if you have an EOS token
            # if next_token == eos_token_id:
            #     break
        
        # Decode back to text
        generated_ids = xp.asnumpy(idx[0]).tolist()
        return self.tokenizer.decode(generated_ids)


if __name__ == "__main__":
    from tokenizers import Tokenizer
    
    # Example usage
    CHECKPOINT_PATH = "checkpoints/pretraining"
    TOKENIZER_PATH = "tokenizer/tokenizer.json"  # Adjust to your tokenizer path
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    VOCAB_SIZE = len(tokenizer.get_vocab())
    D_MODEL = 1024
    N_HEADS = 16
    MAX_SEQ_LEN = 512
    PAD_IDX = 0
    DEPTH = 12
    
    print("Loading model from checkpoint...")
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
        n_heads=N_HEADS,
        transformer_depth=DEPTH
    )
    
    print("\nGenerating text...")
    prompt = "Once upon a time"
    generated = engine.generate(prompt, max_new_tokens=50, temperature=0.8)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")