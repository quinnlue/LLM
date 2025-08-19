"""
Simple text-generation script that re-uses the Transformer architecture
defined in `gpt1/torch_train/model.py`.

Usage
-----
python -m gpt1.inference.generate \
    --prompt "Once upon a time" \
    --max_new_tokens 120 \
    --temperature 0.8 \
    --top_k 40

Notes
-----
- If `--checkpoint` is not provided, the most recent file in `gpt1/checkpoints/`
  will be used automatically.
"""

import argparse
import os
import sys
from pathlib import Path
import torch

# ─── project deps ──────────────────────────────────────────────────────────────
from gpt1.torch_train.train import (
    VOCAB_SIZE, D_MODEL, N_HEADS, DEPTH, MAX_SEQ_LEN, PAD_IDX,
)
from gpt1.torch_train.model import Model as TransformerLM
from gpt1.tokenizer.tokenizer import tokenizer

s = "Once upon a time, there was a test."
ids = tokenizer.encode(s).ids
print("ids[:20] =", ids[:20])
print("roundtrip:", tokenizer.decode(ids, skip_special_tokens=True))

exit()
try:
    # Ensure we have a suitable decoder to avoid artifacts like 'Ġ'/'Ċ'
    from tokenizers import decoders as _tk_decoders  # type: ignore
    if getattr(tokenizer, "decoder", None) is None:
        # Heuristic: if the vocab contains 'Ġ' markers, prefer ByteLevel decoder
        vocab_sample = "".join(list(tokenizer.get_vocab().keys())[:2000])
        if "\u0120" in vocab_sample:  # 'Ġ'
            tokenizer.decoder = _tk_decoders.ByteLevel()
        else:
            tokenizer.decoder = _tk_decoders.BPE()
except Exception:
    pass

# ────────────────────────── helpers ────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CKPT_DIR = _BASE_DIR / "checkpoints"


def _is_git_lfs_pointer(path: str | os.PathLike) -> bool:
    try:
        with open(path, "rt", encoding="utf-8", errors="ignore") as fp:
            first_line = fp.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def _resolve_latest_checkpoint(checkpoint_dir: str | os.PathLike) -> str:
    """Return path to the most recently modified `.pt` checkpoint in dir."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpt_files = [p for p in ckpt_dir.iterdir() if p.suffix == ".pt" and p.is_file()]
    if not ckpt_files:
        raise FileNotFoundError(f"No '.pt' files found in '{ckpt_dir}'.")
    # Prefer likely real checkpoints: not LFS pointers and reasonably large
    def is_likely_real(fp: Path) -> bool:
        try:
            return (not _is_git_lfs_pointer(fp)) and (fp.stat().st_size > 1_000_000)
        except Exception:
            return False

    real_ckpts = [p for p in ckpt_files if is_likely_real(p)]
    candidates = real_ckpts if real_ckpts else [p for p in ckpt_files if not _is_git_lfs_pointer(p)] or ckpt_files
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    if _is_git_lfs_pointer(latest):
        raise FileNotFoundError(
            f"Checkpoint '{latest.name}' appears to be a Git LFS pointer. "
            "Fetch LFS files with: git lfs install && git lfs fetch --all && git lfs checkout"
        )
    return str(latest)


def load_model(ckpt_path: str, device: torch.device) -> TransformerLM:
    """Instantiate the model and load weights from *ckpt_path*."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        transformer_depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
    ).to(device)

    if _is_git_lfs_pointer(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_path}' appears to be a Git LFS pointer, not the actual file. "
            "Run: git lfs install && git lfs fetch --all && git lfs checkout"
        )
    # PyTorch 2.6 defaults weights_only=True which breaks generic checkpoints.
    # We explicitly opt into full unpickling for trusted local checkpoints.
    print(f"Loading checkpoint from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Training saves with key "model"
    state_dict = ckpt.get("model", ckpt.get("model_state", None))
    if state_dict is None:
        raise KeyError("Checkpoint missing model state under keys 'model' or 'model_state'")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.6,
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
        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is not None and next_id.item() == eos_id:
            break

    def _clean_decoded_text(text: str) -> str:
        # Handle common GPT2/Roberta BPE artifacts if tokenizer decoder isn't configured
        if "\u0120" in text:  # 'Ġ' indicates a space prefix in some vocabs
            text = text.replace("\u0120", " ")
        if "\u010a" in text:  # 'Ċ' often used for newlines
            text = text.replace("\u010a", "\n")
        # Normalize multiple spaces introduced by replacements
        text = " ".join(text.split())
        return text

    decoded = tokenizer.decode(ids.squeeze(0).tolist(), skip_special_tokens=True)
    return _clean_decoded_text(decoded)


# ────────────────────────── CLI ────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="TransformerLM inference script")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt checkpoint. If omitted, the latest in gpt1/checkpoints is used.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=str(_DEFAULT_CKPT_DIR),
        help="Directory to scan for checkpoints when --checkpoint is not provided.",
    )
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

    ckpt_path = args.checkpoint or _resolve_latest_checkpoint(args.checkpoint_dir)
    model = load_model(ckpt_path, device=device)
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
            model = load_model(ckpt_path, device=device)
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
    main()
