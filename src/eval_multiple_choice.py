#!/usr/bin/env python3
"""Reproducible zero-shot multiple-choice evaluation for Terminus Flagship 30B checkpoints.

Supported tasks deliberately use their labeled public evaluation splits:
ARC-Easy test, WinoGrande-XL validation (the test answers are not public), and
SciQ test.  Each candidate is scored by its conditional token log-probability.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import BitNetLlama, BitNetLlamaConfig


TASK_DEFAULTS = {
    "arc_easy": ("ai2_arc", "ARC-Easy", "test"),
    "winogrande": ("winogrande", "winogrande_xl", "validation"),
    "sciq": ("allenai/sciq", None, "test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Terminus Flagship 30B checkpoint on a public MCQ benchmark.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_DEFAULTS))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
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
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[BitNetLlama, AutoTokenizer, Dict[str, Any]]:
    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint)
    model_config_path = run_dir / "configs" / "model_config.json"
    train_config_path = run_dir / "configs" / "train_config.json"
    model_config = BitNetLlamaConfig(**load_json(model_config_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    state_dict = {(key[10:] if key.startswith("_orig_mod.") else key): value for key, value in state_dict.items()}
    eval_config = BitNetLlamaConfig(**{**asdict(model_config), "attention_backend": args.attention_backend})
    model = BitNetLlama(eval_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device=device)
    if device.type == "cuda" and args.dtype == "bf16":
        model.to(dtype=torch.bfloat16)
    elif device.type == "cuda" and args.dtype == "fp16":
        model.to(dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(load_json(train_config_path)["tokenizer_name"])
    metadata = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_config_sha256": sha256_file(model_config_path),
        "train_config_sha256": sha256_file(train_config_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
    }
    return model, tokenizer, metadata


def task_examples(task: str) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    dataset_name, config, split = TASK_DEFAULTS[task]
    dataset = load_dataset(dataset_name, config, split=split)
    output: List[Dict[str, Any]] = []
    for row in dataset:
        if task == "arc_easy":
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            prompt = "Question: " + row["question"].strip() + "\nChoices:\n" + "\n".join(
                f"{label}. {text}" for label, text in zip(labels, texts)
            ) + "\nAnswer:"
            output.append({"id": row["id"], "prompt": prompt, "choices": [" " + label for label in labels], "gold": labels.index(row["answerKey"])})
        elif task == "winogrande":
            before, after = row["sentence"].split("_", 1)
            options = [row["option1"], row["option2"]]
            # Keep the trailing space in the continuation rather than the
            # prompt.  This makes the prompt tokenization identical with and
            # without a candidate for BPE tokenizers.
            output.append({"id": str(len(output)), "prompt": before.rstrip(), "choices": [" " + option + after for option in options], "gold": int(row["answer"]) - 1})
        else:
            choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
            prompt = f"Support: {row['support'].strip()}\nQuestion: {row['question'].strip()}\nAnswer:"
            output.append({"id": str(len(output)), "prompt": prompt, "choices": [" " + choice for choice in choices], "gold": 3})
    return output, {"dataset": dataset_name, "config": config or "", "split": split, "examples": str(len(output))}


def candidate_ids(tokenizer: AutoTokenizer, prompt: str, continuation: str, max_seq_len: int) -> tuple[List[int], int, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    if not full_ids[: len(prompt_ids)] == prompt_ids:
        raise RuntimeError("tokenizer changed prompt tokens at a candidate boundary; use a stable prompt template")
    target_length = len(full_ids) - len(prompt_ids)
    if target_length < 1:
        raise RuntimeError("candidate has no scorable tokens")
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[-max_seq_len:]
        start = len(full_ids) - target_length
        if start < 1:
            raise RuntimeError("candidate is too long to score after truncation")
    else:
        start = len(prompt_ids)
    return full_ids, start, target_length


@torch.inference_mode()
def score_candidates(model: BitNetLlama, candidates: Sequence[Dict[str, Any]], device: torch.device, batch_size: int, dtype_name: str, pad_token_id: int) -> List[float]:
    scores: List[float] = []
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        max_length = max(len(item["ids"]) for item in batch)
        max_target = max(item["target_length"] for item in batch)
        inputs = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long, device=device)
        targets = torch.zeros((len(batch), max_target), dtype=torch.long, device=device)
        starts = torch.empty(len(batch), dtype=torch.long, device=device)
        lengths = torch.empty(len(batch), dtype=torch.long, device=device)
        for row, item in enumerate(batch):
            ids, start, length = item["ids"], item["start"], item["target_length"]
            inputs[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            targets[row, :length] = inputs[row, start : start + length]
            starts[row], lengths[row] = start, length
        with autocast_context(device, dtype_name):
            logits = model(inputs, return_logits=True)["logits"]
        positions = (starts[:, None] - 1 + torch.arange(max_target, device=device)[None, :]).clamp_max(max_length - 1)
        selected = logits[torch.arange(len(batch), device=device)[:, None], positions].float()
        logprobs = torch.log_softmax(selected, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        valid = torch.arange(max_target, device=device)[None, :] < lengths[:, None]
        scores.extend((logprobs * valid).sum(dim=1).cpu().tolist())
    return scores


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, tokenizer, metadata = load_model(args, device)
    examples, dataset_info = task_examples(args.task)
    candidates: List[Dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        for choice_index, choice in enumerate(example["choices"]):
            ids, start, target_length = candidate_ids(tokenizer, example["prompt"], choice, model.config.max_seq_len)
            candidates.append({"example_index": example_index, "choice_index": choice_index, "ids": ids, "start": start, "target_length": target_length})
    started = time.monotonic()
    scores = score_candidates(model, candidates, device, args.batch_size, args.dtype, tokenizer.pad_token_id or tokenizer.eos_token_id)
    grouped: List[List[Dict[str, Any]]] = [[] for _ in examples]
    for candidate, score in zip(candidates, scores):
        grouped[candidate["example_index"]].append({**candidate, "logprob": float(score)})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_correct = normalized_correct = 0
    with (output_dir / "per_example.jsonl").open("w") as handle:
        for example, choices in zip(examples, grouped):
            choices.sort(key=lambda item: item["choice_index"])
            raw_scores = [item["logprob"] for item in choices]
            normalized_scores = [item["logprob"] / item["target_length"] for item in choices]
            raw_prediction = max(range(len(choices)), key=raw_scores.__getitem__)
            normalized_prediction = max(range(len(choices)), key=normalized_scores.__getitem__)
            raw_correct += raw_prediction == example["gold"]
            normalized_correct += normalized_prediction == example["gold"]
            handle.write(json.dumps({"id": example["id"], "gold": example["gold"], "raw_prediction": raw_prediction, "normalized_prediction": normalized_prediction, "raw_scores": raw_scores, "normalized_scores": normalized_scores}) + "\n")
    result = {**metadata, **dataset_info, "task": args.task, "device": str(device), "dtype": args.dtype, "attention_backend": args.attention_backend, "batch_size": args.batch_size, "raw_accuracy": raw_correct / len(examples), "length_normalized_accuracy": normalized_correct / len(examples), "raw_correct": raw_correct, "length_normalized_correct": normalized_correct, "elapsed_seconds": time.monotonic() - started}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
