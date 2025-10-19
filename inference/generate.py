import sys
import os
import heapq

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
        self.model.is_training = False
        self.is_cuda = xp.__name__ == "cupy"
        self.prompter_id = tokenizer.token_to_id("<prompter>")
        self.assistant_id = tokenizer.token_to_id("<assistant>")
        self.eos_id = tokenizer.token_to_id("<eos>")

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
    
    def _sample_token(self, logits_np, temperature=1.0, top_k=None, top_p=None, repeat_penalty=None, repeated_mask=None):
        """
        Sample a token from logits using various sampling strategies.
        
        Args:
            logits_np: Logits array (numpy)
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling
            repeat_penalty: If set, penalize repeated tokens
            repeated_mask: Boolean mask of repeated tokens
            
        Returns:
            Selected token ID
        """
        # Apply temperature scaling
        logits_np = logits_np / temperature
        
        # Apply repeat penalty if specified
        if repeat_penalty is not None and repeated_mask is not None:
            logits_np[repeated_mask & (logits_np > 0)] /= repeat_penalty
            logits_np[repeated_mask & (logits_np < 0)] *= repeat_penalty
        
        # Apply top-k filtering if specified
        if top_k is not None:
            top_k_idx = np.argpartition(logits_np, -top_k)[-top_k:]
            mask = np.full_like(logits_np, -float('inf'))
            mask[top_k_idx] = logits_np[top_k_idx]
            logits_np = mask
        
        # Convert to probabilities
        probs = np.exp(logits_np - np.max(logits_np))
        probs = probs / np.sum(probs)
        
        # Apply top-p (nucleus sampling) if specified
        if top_p is not None:
            heap = []
            for token_id, prob in enumerate(probs):
                if prob > 0:
                    heapq.heappush(heap, (-prob, token_id))
            
            # Select tokens until cumulative probability >= top_p
            selected_tokens = []
            cumulative_prob = 0.0
            
            while heap and cumulative_prob < top_p:
                neg_prob, token_id = heapq.heappop(heap)
                prob = -neg_prob
                selected_tokens.append(token_id)
                cumulative_prob += prob
            
            # Mask out non-selected tokens
            mask = np.full_like(probs, 0.0)
            mask[selected_tokens] = probs[selected_tokens]
            probs = mask
            probs = probs / np.sum(probs)
        
        # Sample from distribution
        next_token = np.random.choice(len(probs), p=probs)
        return next_token
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 500, 
        temperature: float = 1.0, 
        top_k: int = 80, 
        top_p: float | None = None,
        stream: bool = False,
        repeat_penalty: float | None = None
    ) -> str:
        """
        Generate text continuation from a prompt.
        
        Args:
            prompt: String input text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling (select tokens with cumulative probability >= top_p)
            stream: If True, stream the generated text
            repeat_penalty: If set, penalize the generation of repeated tokens
        
        Returns:
            Generated text as string
        """
        # Encode prompt
        encoded = [self.prompter_id] + self.tokenizer.encode(prompt) + [self.assistant_id]


        if repeat_penalty is not None:
            repeated_mask = np.zeros((self.model.vocab_size,), dtype=np.bool_)
            repeated_mask[encoded.ids] = True

        idx = xp.array([encoded.ids], dtype=xp.int32)


        # kv_cache is of shape (trasnformer_depth, batch_size, max_seq_len, d_model)
        kv_shape = (self.model.transformer_depth, 1, self.model.max_seq_len, self.model.d_model)
        kv_cache = {
            "k": Tensor(np.zeros(kv_shape)),
            "v": Tensor(np.zeros(kv_shape))
        }


        for i in range(len(idx[0])):
            current_position = i
            logits = self.model.forward(idx[:,:i+1], kv_cache, current_position)


        for _ in range(max_new_tokens):
            logits = self.model.forward(idx, kv_cache, current_position)
            logits = logits[:, -1, :]
            
            # Convert to numpy for sampling
            logits_np = xp.asnumpy(logits.data[0]) if self.is_cuda else logits.data[0]
            
            # Sample token using the extracted sampling logic
            next_token = self._sample_token(
                logits_np=logits_np,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                repeated_mask=repeated_mask
            )
            
            # Update repeated mask for repeat penalty
            if repeat_penalty is not None:
                repeated_mask[next_token] = True

            # Append to sequence
            next_token_array = xp.array([[next_token]], dtype=xp.int32)
            if stream:
                print(self.tokenizer.decode(xp.asnumpy(next_token_array[0]).tolist()), end="", flush=True)
            idx = xp.concatenate([idx, next_token_array], axis=1)

            if next_token == self.tokenizer.token_to_id("[EOS]"):
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
    generated = engine.generate(prompt, max_new_tokens=50, temperature=0.8, top_p=0.9)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")