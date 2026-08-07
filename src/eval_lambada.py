#!/usr/bin/env python3
"""Evaluate exact last-word prediction on the canonical LAMBADA open test set."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import BitNetLlama, BitNetLlamaConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Terminus Flagship 30B on LAMBADA open.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="EleutherAI/lambada_openai")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attention-backend", default="sdpa_reference", choices=["fa3", "flex", "sdpa_reference"])
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if dtype_name == "bf16" else torch.float16)


def load_model_and_tokenizer(
    run_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    dtype_name: str,
    attention_backend: str,
):
    model_config_path = run_dir / "configs" / "model_config.json"
    train_config_path = run_dir / "configs" / "train_config.json"
    model_config = BitNetLlamaConfig(**load_json(model_config_path))
    train_config = load_json(train_config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    state_dict = {(key[10:] if key.startswith("_orig_mod.") else key): value for key, value in state_dict.items()}
    eval_config = BitNetLlamaConfig(**{**asdict(model_config), "attention_backend": attention_backend})
    model = BitNetLlama(eval_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device=device)
    if dtype_name == "bf16" and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        train_config["tokenizer_name"],
        revision=train_config.get("tokenizer_revision"),
        use_fast=True,
    )
    tokenizer.model_max_length = int(1e30)
    return model, tokenizer, checkpoint, model_config, train_config


def prepare_examples(dataset, tokenizer, max_seq_len: int) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for index, row in enumerate(dataset):
        text = str(row["text"]).strip()
        if " " not in text:
            continue
        context, target = text.rsplit(" ", 1)
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # Tokenize the complete sentence so SentencePiece does not emit a
        # standalone whitespace token at the context/target boundary.
        if len(full_ids) < len(context_ids):
            continue
        target_ids = full_ids[len(context_ids) :]
        if not context_ids or not target_ids:
            continue
        sequence = context_ids + target_ids
        if len(sequence) > max_seq_len:
            sequence = sequence[-max_seq_len:]
            context_start = len(sequence) - len(target_ids)
            if context_start < 1:
                continue
        else:
            context_start = len(context_ids)
        prepared.append(
            {
                "index": index,
                "input_ids": sequence,
                "target_ids": target_ids,
                "context_start": context_start,
                "target_word": target,
                "text": text,
            }
        )
    return prepared


@torch.inference_mode()
def evaluate(model, examples: List[Dict[str, Any]], device: torch.device, batch_size: int, dtype_name: str, pad_id: int):
    total_nll = 0.0
    total_target_tokens = 0
    exact = 0
    token_correct = 0
    per_example: List[Dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        for row, item in enumerate(batch):
            input_ids[row, : len(item["input_ids"])] = torch.tensor(item["input_ids"], device=device)
        with autocast_context(device, dtype_name):
            logits = model(input_ids, return_logits=True)["logits"]
        for row, item in enumerate(batch):
            target = torch.tensor(item["target_ids"], dtype=torch.long, device=device)
            start_pos = int(item["context_start"])
            positions = torch.arange(start_pos - 1, start_pos - 1 + target.numel(), device=device)
            predicted = logits[row, positions].float().argmax(dim=-1)
            target_logits = logits[row, positions].float()
            log_probs = torch.log_softmax(target_logits, dim=-1)
            nll = -log_probs.gather(1, target[:, None]).sum().item()
            correct_tokens = int((predicted == target).sum().item())
            exact_match = bool(torch.equal(predicted, target))
            total_nll += nll
            total_target_tokens += int(target.numel())
            token_correct += correct_tokens
            exact += int(exact_match)
            per_example.append(
                {
                    "index": item["index"],
                    "target_word": item["target_word"],
                    "target_token_count": int(target.numel()),
                    "token_correct": correct_tokens,
                    "exact_match": exact_match,
                }
            )
    return {
        "examples": len(examples),
        "exact_correct": exact,
        "exact_accuracy": exact / len(examples),
        "exact_accuracy_percent": 100.0 * exact / len(examples),
        "token_correct": token_correct,
        "target_tokens": total_target_tokens,
        "token_accuracy": token_correct / total_target_tokens,
        "target_nll": total_nll,
        "target_average_nll": total_nll / total_target_tokens,
        "target_perplexity": math.exp(min(50.0, total_nll / total_target_tokens)),
        "per_example": per_example,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    device = torch.device(args.device)
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    if "text" not in dataset.column_names:
        raise RuntimeError(f"LAMBADA dataset must have a text column, got {dataset.column_names}")
    model, tokenizer, checkpoint, model_config, train_config = load_model_and_tokenizer(
        run_dir, checkpoint_path, device, args.dtype, args.attention_backend
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    examples = prepare_examples(dataset, tokenizer, model_config.max_seq_len)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = evaluate(model, examples, device, args.batch_size, args.dtype, int(pad_id))
    if device.type == "cuda":
        torch.cuda.synchronize()
    result["wall_seconds"] = time.perf_counter() - started
    result["benchmark"] = "lambada_openai"
    result["dataset"] = {"name": args.dataset, "split": args.dataset_split, "rows": len(dataset)}
    result["checkpoint"] = {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path), "step": checkpoint.get("step")}
    result["evaluation"] = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "batch_size": args.batch_size,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    result["training_run"] = {"path": str(run_dir), "tokenizer_name": train_config.get("tokenizer_name"), "tokenizer_revision": train_config.get("tokenizer_revision")}
    per_example = result.pop("per_example")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "per_example.jsonl").open("w") as handle:
        for row in per_example:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
