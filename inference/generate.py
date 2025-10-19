import sys
import os

import dlx as dlx
from dlx import xp
from dlx.nn.tensor import Tensor
from gpt1.training.model import Model
from dlx.nn.optim import AdamW
import numpy as np


class InferenceEngine:
    def __init__(self, model: Model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.is_training = False  # Set to eval mode
        self.is_cuda = xp.__name__ == "cupy"
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer, vocab_size, d_model, max_seq_len, 
                       pad_idx, n_heads, transformer_depth, mlp_ratio=4, lora=False, lora_r=8, lora_alpha=8):
        """
        Load model from checkpoint directory.
        
        Args:
            checkpoint_path: Path to checkpoint directory containing model/*.npy files
            tokenizer: Tokenizer object (from tokenizers import Tokenizer)
            vocab_size, d_model, max_seq_len, pad_idx, n_heads, transformer_depth: Model architecture parameters
            mlp_ratio: MLP expansion ratio (default: 4)
        """
        model = Model(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            n_heads=n_heads,
            transformer_depth=transformer_depth,
            checkpoint_interval_seconds=0,
            train_dir="",
            validation_dir="",
            checkpoint_dir="",
            epochs=0,
            mini_batch_per_step=1,
            mlp_ratio=mlp_ratio,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        optim = AdamW(model.parameters(), precision=(np.float32, np.float32))
        optim.load_state(checkpoint_path)
        
        return cls(model, tokenizer)
    
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=None, stream=False):
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


        # kv_cache is of shape (trasnformer_depth, batch_size, max_seq_len, d_model)
        kv_shape = (self.model.transformer_depth, 1, self.model.max_seq_len, self.model.d_model)
        kv_cache = {
            "k": Tensor(np.zeros(kv_shape)),
            "v": Tensor(np.zeros(kv_shape))
        }


        for i in range(len(idx[0])):
            current_position = i
            # print(f"Print kv cache after {i} tokens (position {current_position}) (in the prompt part of idx)")
            # print(kv_cache)
            logits = self.model.forward(idx[:,:i+1], kv_cache, current_position)


        for _ in range(10):
            # print(f"Print kv cache after {i} tokens (position {current_position}) (in the generated part of idx)")
            # print(kv_cache)
            logits = self.model.forward(idx, kv_cache, current_position)
            logits = logits[:, -1, :]
            logits = logits / temperature

            logits_np = xp.asnumpy(logits.data[0]) if self.is_cuda else logits.data[0]

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
            if stream:
                print(self.tokenizer.decode(xp.asnumpy(next_token_array[0]).tolist()), end="", flush=True)
            idx = xp.concatenate([idx, next_token_array], axis=1)

            if next_token == self.tokenizer.token_to_id("[EOS]"):
                print("[EOS]", flush=True)
                break

            

            current_position += 1
    
        return self.tokenizer.decode(xp.asnumpy(idx[0]).tolist()) if self.is_cuda else self.tokenizer.decode(idx[0].data[0].tolist())

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
    MLP_RATIO = 4

    print("Loading model from checkpoint...")
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
        n_heads=N_HEADS,
        transformer_depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        lora=True,
        lora_r=8,
        lora_alpha=8
    )
    
    print("\nGenerating text...")
    prompt = "Once upon a time"
    generated = engine.generate(prompt, max_new_tokens=50, temperature=0.8)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")