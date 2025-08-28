import time
import os
import torch

def calculate_steps_per_sec(current_step: int, start_time: float) -> float:
    elapsed_time = time.perf_counter() - start_time
    return current_step / elapsed_time if elapsed_time > 0 else 0.0


def load_latest_checkpoint(model, optimizer, scheduler, scaler, device, checkpoint_dir) -> None:
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
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    print(f"[INFO] Resumed from checkpoint {latest}")