from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model import BitNetLlama, BitNetLlamaConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a Terminus Flagship 30B Transformer++ checkpoint")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--prompt", required=True, help="Prompt text to continue from.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--attention-backend",
        choices=["fa3", "flex", "sdpa_reference"],
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
    return torch.device(requested)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Optional[Path]]:
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if run_dir is None and checkpoint is None:
        raise ValueError("provide --run-dir or --checkpoint")
    if checkpoint is None:
        checkpoint = run_dir / "checkpoints" / "ckpt_final.pt"
    if run_dir is None:
        run_dir = infer_run_dir_from_checkpoint(checkpoint)
    return checkpoint, run_dir


def infer_run_dir_from_checkpoint(checkpoint: Path) -> Optional[Path]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name == "checkpoints" and checkpoint.parent.parent.exists():
        return checkpoint.parent.parent
    return None


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def build_model_config(run_dir: Optional[Path], checkpoint_payload: Dict[str, object]) -> BitNetLlamaConfig:
    embedded = checkpoint_payload.get("model_config")
    if isinstance(embedded, dict):
        config = BitNetLlamaConfig(**embedded)
        if run_dir is not None:
            model_config_path = run_dir / "configs" / "model_config.json"
            if model_config_path.exists() and load_json(model_config_path) != embedded:
                raise RuntimeError("run-directory model_config does not match checkpoint model_config")
        return config
    if run_dir is not None:
        model_config_path = run_dir / "configs" / "model_config.json"
        if model_config_path.exists():
            payload = load_json(model_config_path)
            return BitNetLlamaConfig(**payload)

    raise RuntimeError(
        "checkpoint has no embedded model_config; refusing to reconstruct a model from defaults"
    )


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        raise RuntimeError(f"checkpoint integrity sidecar is missing: {sidecar}")
    metadata = load_json(sidecar)
    if int(metadata.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError("checkpoint size does not match its integrity sidecar")
    if metadata.get("sha256") != file_sha256(path):
        raise RuntimeError("checkpoint SHA-256 does not match its integrity sidecar")


def resolve_tokenizer_name(
    explicit_name: Optional[str],
    run_dir: Optional[Path],
    checkpoint_payload: Dict[str, object],
) -> str:
    if explicit_name:
        return explicit_name
    if run_dir is not None:
        train_config_path = run_dir / "configs" / "train_config.json"
        if train_config_path.exists():
            train_config = load_json(train_config_path)
            tokenizer_name = train_config.get("tokenizer_name")
            if isinstance(tokenizer_name, str) and tokenizer_name:
                return tokenizer_name
    train_config = checkpoint_payload.get("train_config", {})
    tokenizer_name = train_config.get("tokenizer_name")
    if isinstance(tokenizer_name, str) and tokenizer_name:
        return tokenizer_name
    return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def resolve_tokenizer_revision(
    explicit_revision: Optional[str], checkpoint_payload: Dict[str, object]
) -> str:
    if explicit_revision:
        return explicit_revision
    train_config = checkpoint_payload.get("train_config", {})
    revision = train_config.get("tokenizer_revision") if isinstance(train_config, dict) else None
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("checkpoint is missing its pinned tokenizer_revision")
    return revision


def load_model_and_tokenizer(
    checkpoint_path: Path,
    *,
    run_dir: Optional[Path],
    tokenizer_name: Optional[str],
    tokenizer_revision: Optional[str],
    attention_backend: Optional[str],
    device: torch.device,
    dtype: str,
) -> tuple[BitNetLlama, AutoTokenizer, Dict[str, object], BitNetLlamaConfig]:
    verify_checkpoint_sidecar(checkpoint_path)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = build_model_config(run_dir, checkpoint_payload)
    if attention_backend is not None:
        model_config = BitNetLlamaConfig(**{**model_config.__dict__, "attention_backend": attention_backend})

    resolved_tokenizer_name = resolve_tokenizer_name(tokenizer_name, run_dir, checkpoint_payload)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer_name,
        revision=resolve_tokenizer_revision(tokenizer_revision, checkpoint_payload),
        use_fast=True,
    )
    tokenizer.model_max_length = int(1e30)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BitNetLlama(model_config)
    state_dict = checkpoint_payload["model"]
    state_dict = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if dtype == "bf16" and device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    return model, tokenizer, checkpoint_payload, model_config


def sample_next_token(logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scores = logits / temperature
    if top_k > 0 and top_k < scores.shape[-1]:
        values, _ = torch.topk(scores, top_k)
        cutoff = values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(sorted_mask, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: BitNetLlama,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError("prompt produced no tokens")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    if input_ids.shape[1] > model.config.max_seq_len:
        raise ValueError("prompt exceeds max_seq_len for cached Terminus Flagship 30B inference")
    if input_ids.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError(
            "prompt plus requested generation exceeds max_seq_len; cached Terminus Flagship 30B inference "
            "preserves global history and therefore does not silently evict it"
        )
    eos_token_id = tokenizer.eos_token_id
    output, cache = model.forward_cached(input_ids, return_logits=True)
    for _ in range(max_new_tokens):
        logits = output["logits"][:, -1, :].float()
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        input_ids = torch.cat([input_ids, next_token.to(device=device)], dim=1)
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break
        output, cache = model.forward_cached(next_token.to(device=device), cache=cache, return_logits=True)
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint_path, run_dir = resolve_paths(args)
    device = resolve_device(args.device)
    model, tokenizer, _, _ = load_model_and_tokenizer(
        checkpoint_path,
        run_dir=run_dir,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        attention_backend=args.attention_backend,
        device=device,
        dtype=args.dtype,
    )
    text = generate(
        model,
        tokenizer,
        args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
