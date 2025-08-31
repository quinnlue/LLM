import time
import os
import torch
import torch.nn as nn

def calculate_steps_per_sec(current_step: int, start_time: float) -> float:
    elapsed_time = time.perf_counter() - start_time
    return current_step / elapsed_time if elapsed_time > 0 else 0.0


def resize_token_embeddings(model, new_vocab_size: int, pad_idx: int = 0) -> None:
    """
    Resize token embeddings and language model head when vocabulary size changes.
    
    Args:
        model: The model to resize
        new_vocab_size: New vocabulary size
        pad_idx: Padding token index for initialization
    """
    current_vocab_size = model.token_emb.num_embeddings
    
    if new_vocab_size == current_vocab_size:
        return  # No resizing needed
    
    print(f"[INFO] Resizing vocab from {current_vocab_size} to {new_vocab_size}")
    
    # Get current weights
    old_token_emb_weight = model.token_emb.weight.data
    old_lm_head_weight = model.lm_head.weight.data
    old_lm_head_bias = model.lm_head.bias.data if model.lm_head.bias is not None else None
    
    # Create new embedding layer
    # Ensure the new layers are created on the same device (and dtype) as the existing model parameters
    base_param: torch.Tensor = model.token_emb.weight
    device = base_param.device
    dtype = base_param.dtype

    new_token_emb = (
        nn.Embedding(new_vocab_size, model.d_model, padding_idx=pad_idx)
        .to(device=device, dtype=dtype)
    )
    new_lm_head = nn.Linear(model.d_model, new_vocab_size).to(device=device, dtype=dtype)
    
    # Initialize new weights with normal distribution (same as model's _init_weights)
    nn.init.normal_(new_token_emb.weight, mean=0.0, std=0.02)
    nn.init.xavier_uniform_(new_lm_head.weight)
    if new_lm_head.bias is not None:
        nn.init.zeros_(new_lm_head.bias)
    
    # Copy over the old weights for existing vocabulary
    min_vocab_size = min(current_vocab_size, new_vocab_size)
    new_token_emb.weight.data[:min_vocab_size] = old_token_emb_weight[:min_vocab_size]
    new_lm_head.weight.data[:min_vocab_size] = old_lm_head_weight[:min_vocab_size]
    if old_lm_head_bias is not None and new_lm_head.bias is not None:
        new_lm_head.bias.data[:min_vocab_size] = old_lm_head_bias[:min_vocab_size]
    
    # Replace the model's layers
    model.token_emb = new_token_emb
    model.lm_head = new_lm_head
    model.vocab_size = new_vocab_size
    
    print(f"[INFO] Successfully resized token embeddings and lm_head")


def load_latest_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    checkpoint_dir,
    *,
    strict: bool = True,
    load_optim_state: bool = True,
) -> None:
    if not os.path.isdir(checkpoint_dir):
        print(f"[INFO] No checkpoint directory found at {checkpoint_dir}")
        return  
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not ckpts:
        print(f"[INFO] No checkpoints found in {checkpoint_dir}")
        return
    latest = max(ckpts, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    cp_path = os.path.join(checkpoint_dir, latest)
    state = torch.load(cp_path, map_location=device)
    
    # Check for vocabulary size mismatch and handle it
    model_state = state["model"]
    current_vocab_size = model.vocab_size
    
    # Check if token_emb.weight exists in checkpoint and has different size
    if "token_emb.weight" in model_state:
        checkpoint_vocab_size = model_state["token_emb.weight"].shape[0]
        if checkpoint_vocab_size != current_vocab_size:
            print(f"[INFO] Vocabulary size mismatch detected:")
            print(f"[INFO] Checkpoint vocab size: {checkpoint_vocab_size}")
            print(f"[INFO] Current model vocab size: {current_vocab_size}")
            print(f"[INFO] Resizing token embeddings...")
            
            # Resize current model to match checkpoint vocab size temporarily
            original_vocab_size = current_vocab_size
            resize_token_embeddings(model, checkpoint_vocab_size, model.pad_idx)
            
            # Load the checkpoint state
            try:
                model.load_state_dict(model_state, strict=strict)
                print(f"[INFO] Successfully loaded checkpoint with vocab size {checkpoint_vocab_size}")
                
                # Now resize to the target vocab size
                resize_token_embeddings(model, original_vocab_size, model.pad_idx)
                print(f"[INFO] Resized model to target vocab size {original_vocab_size}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint after resizing: {e}")
                # Restore original vocab size if loading failed
                resize_token_embeddings(model, original_vocab_size, model.pad_idx)
                return
        else:
            # No vocab size mismatch, load normally
            model.load_state_dict(model_state, strict=strict)
    else:
        # No token embeddings in checkpoint, load normally
        model.load_state_dict(model_state, strict=strict)
    
    # ------------------------------------------------------------------ #
    # Optionally restore optimizer / scheduler / scaler state.           #
    # This is useful for full-model training, but can be disabled        #
    # (e.g. when fine-tuning only a subset of parameters like LoRA).     #
    # ------------------------------------------------------------------ #
    if load_optim_state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except ValueError as exc:
            print(f"[WARN] Skipping optimizer state: {exc}")
        else:
            if "scheduler" in state:
                try:
                    scheduler.load_state_dict(state["scheduler"])
                except Exception as exc:  # scheduler state is nice-to-have
                    print(f"[WARN] Skipping scheduler state: {exc}")
            if "scaler" in state:
                try:
                    scaler.load_state_dict(state["scaler"])
                except Exception as exc:
                    print(f"[WARN] Skipping GradScaler state: {exc}")

    print(f"[INFO] Resumed from checkpoint {latest}")