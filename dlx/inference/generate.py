"""
Simple text-generation script that re-uses the TransformerLM architecture
defined in src/training/torch_train.py.

Usage
-----
python -m src.inference.generate \
    --checkpoint checkpoints/<your_ckpt>.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 120 \
    --temperature 0.8 \
    --top_k 40
"""

import argparse
import os
import sys
import torch

# ─── project deps ──────────────────────────────────────────────────────────────
from src.training.torch_train import (
    TransformerLM, VOCAB_SIZE, D_MODEL, N_HEADS,
    DEPTH, MAX_SEQ_LEN, PAD_IDX, EOS_IDX, DEVICE as _TRAIN_DEVICE,  # renamed
)
from src.tokenizer.tokenizer import tokenizer

# ────────────────────────── helpers ────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device) -> TransformerLM:
    """Instantiate the model and load weights from *ckpt_path*."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """
    Greedy / top-k sampling generation.

    • prompt             : input string
    • max_new_tokens     : how many tokens to append
    • temperature        : softmax temperature (1.0 = none)
    • top_k              : if set, restrict sampling to top-k logits
    """
    # Tokenise prompt → (1, S)
    prompt_ids = tokenizer.encode(prompt).ids[: MAX_SEQ_LEN - 1]
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        if ids.shape[1] >= MAX_SEQ_LEN:
            break

        # Forward pass
        logits = model(ids)[:, -1, :] / temperature  # (1, vocab)

        # Optional top-k filtering
        if top_k is not None:
            top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(1, top_indices, top_values)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        ids = torch.cat([ids, next_id], dim=1)

        # Stop when EOS token generated
        if next_id.item() == EOS_IDX:
            break

    return tokenizer.decode(ids.squeeze(0).tolist())


# ────────────────────────── CLI ────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="TransformerLM inference script")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--prompt", required=True, help="Seed text")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Where to run inference (default: auto)",
    )
    args = parser.parse_args()

    # ─── Select device ────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_model(args.checkpoint, device=device)
    try:
        out_text = generate(
            model,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            print("[warning] CUDA OOM → retrying on CPU …")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            model = load_model(args.checkpoint, device=device)
            out_text = generate(
                model,
                prompt=args.prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        else:
            raise
    print(out_text)


if __name__ == "__main__":
    # If the user supplied CLI args, run the normal path; otherwise fall back to a quick demo.
    if len(sys.argv) > 1:
        main()
    else:
        # ─── Quick demo ─────────────────────────────────────────────────────────
        ckpt_dir = "checkpoints"
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if not ckpt_files:
            raise FileNotFoundError(
                "No '.pt' files found in 'checkpoints/'.  "
                "Either train the model first or pass --checkpoint <path>."
            )

        # Pick the most recently modified checkpoint
        latest_ckpt = max(
            ckpt_files,
            key=lambda f: os.path.getctime(os.path.join(ckpt_dir, f)),
        )
        ckpt_path = os.path.join(ckpt_dir, latest_ckpt)

        print(f"[demo] Using checkpoint → {ckpt_path}")
        # ─── Demo section ─────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(ckpt_path, device=device)

        demo_prompt = "The best way to guard"
        print(f"[demo] Prompt: {demo_prompt!r}")
        completion = generate(
            model,
            demo_prompt,
            device=device,
            max_new_tokens=64,
            temperature=0.8,
            top_k=40,
        )
        print("[demo] Completion:\n" + completion)
