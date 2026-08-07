#!/usr/bin/env python3
"""Reproducible full HellaSwag validation evaluation for Terminus Flagship 30B."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import BitNetLlama, BitNetLlamaConfig


DEFAULT_DATA_REVISION = "a29ff8e9a04bba4bd6588223785ce105328adc57"
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/"
    f"{DEFAULT_DATA_REVISION}/data/hellaswag_val.jsonl"
)
DEFAULT_DATA_SHA256 = "0aa3b88843990f3f10a97b9575c94d7b71fb2205240ba04ae4884d9e9c992588"
FULL_VALIDATION_EXAMPLES = 10_042


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Terminus Flagship 30B checkpoint on HellaSwag validation.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing configs/.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to RUN_DIR/checkpoints/ckpt_final.pt.")
    parser.add_argument("--output-dir", required=True, help="New directory for this evaluation's artifacts.")
    parser.add_argument("--data", default=None, help="Pinned HellaSwag validation JSONL path.")
    parser.add_argument("--download-data", action="store_true", help="Download --data only when it is absent.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--attention-backend",
        default="sdpa_reference",
        choices=["fa3", "flex", "sdpa_reference"],
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Number of answer continuations per forward pass.")
    parser.add_argument("--max-examples", type=int, default=0, help="0 means all 10,042 validation examples.")
    parser.add_argument("--allow-subset", action="store_true", help="Required whenever --max-examples is nonzero.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fetch_data(path: Path, url: str, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        actual = sha256_file(partial)
        if actual != expected_sha256:
            raise RuntimeError(
                f"downloaded HellaSwag hash mismatch: expected {expected_sha256}, got {actual}"
            )
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def load_examples(path: Path, expected_sha256: str, max_examples: int) -> List[Dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"HellaSwag hash mismatch: expected {expected_sha256}, got {actual}")
    examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(examples) != FULL_VALIDATION_EXAMPLES:
        raise RuntimeError(
            f"expected {FULL_VALIDATION_EXAMPLES} HellaSwag validation examples, found {len(examples)}"
        )
    return examples[:max_examples] if max_examples else examples


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_run_config(run_dir: Path) -> tuple[BitNetLlamaConfig, Dict[str, Any], Path, Path]:
    model_config_path = run_dir / "configs" / "model_config.json"
    train_config_path = run_dir / "configs" / "train_config.json"
    if not model_config_path.is_file() or not train_config_path.is_file():
        raise FileNotFoundError(
            "--run-dir must contain configs/model_config.json and configs/train_config.json"
        )
    return (
        BitNetLlamaConfig(**load_json(model_config_path)),
        load_json(train_config_path),
        model_config_path,
        train_config_path,
    )


def load_model(
    checkpoint_path: Path,
    model_config: BitNetLlamaConfig,
    device: torch.device,
    dtype_name: str,
) -> tuple[BitNetLlama, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"{checkpoint_path} is not a Terminus Flagship 30B checkpoint with a model state dict")
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint model payload is not a state dictionary")
    normalized: Dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = key[10:] if key.startswith("_orig_mod.") else key
        if normalized_key in normalized:
            raise RuntimeError(f"checkpoint key collision after compile-prefix cleanup: {normalized_key}")
        normalized[normalized_key] = value

    model = BitNetLlama(model_config)
    model.load_state_dict(normalized, strict=True)
    model.to(device=device)
    if dtype_name == "bf16":
        model.to(dtype=torch.bfloat16)
    model.eval()
    return model, checkpoint


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def preprocess_hellaswag(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def prepare_candidates(
    examples: Sequence[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
) -> tuple[List[Dict[str, Any]], int]:
    candidates: List[Dict[str, Any]] = []
    truncated = 0
    for example_index, example in enumerate(examples):
        context = preprocess_hellaswag(
            str(example["activity_label"])
            + ": "
            + str(example["ctx_a"])
            + " "
            + str(example["ctx_b"]).capitalize()
        )
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        if not context_ids:
            raise RuntimeError(f"example {example_index} has an empty tokenized context")
        endings = example["endings"]
        if len(endings) != 4:
            raise RuntimeError(f"example {example_index} has {len(endings)} endings, expected 4")
        for choice_index, ending in enumerate(endings):
            continuation = " " + preprocess_hellaswag(str(ending))
            continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
            if not continuation_ids:
                raise RuntimeError(f"example {example_index} choice {choice_index} has an empty continuation")
            sequence = context_ids + continuation_ids
            was_truncated = len(sequence) > max_seq_len
            if was_truncated:
                sequence = sequence[-max_seq_len:]
                truncated += 1
            continuation_start = len(sequence) - len(continuation_ids)
            if continuation_start < 1:
                raise RuntimeError(
                    f"example {example_index} choice {choice_index} cannot be scored after context truncation"
                )
            candidates.append(
                {
                    "example_index": example_index,
                    "choice_index": choice_index,
                    "input_ids": sequence,
                    "continuation_start": continuation_start,
                    "continuation_length": len(continuation_ids),
                    "choice_character_length": len(continuation[1:]),
                    "truncated": was_truncated,
                }
            )
    return candidates, truncated


@torch.inference_mode()
def score_candidates(
    model: BitNetLlama,
    candidates: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    dtype_name: str,
    pad_token_id: int,
) -> List[float]:
    scores: List[float] = []
    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
        max_length = max(len(item["input_ids"]) for item in batch)
        max_continuation = max(int(item["continuation_length"]) for item in batch)
        input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long, device=device)
        targets = torch.zeros((len(batch), max_continuation), dtype=torch.long, device=device)
        starts = torch.empty(len(batch), dtype=torch.long, device=device)
        lengths = torch.empty(len(batch), dtype=torch.long, device=device)
        for row, item in enumerate(batch):
            ids = item["input_ids"]
            start = int(item["continuation_start"])
            length = int(item["continuation_length"])
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            targets[row, :length] = input_ids[row, start : start + length]
            starts[row] = start
            lengths[row] = length

        with autocast_context(device, dtype_name):
            logits = model(input_ids, return_logits=True)["logits"]
        token_positions = (starts[:, None] - 1 + torch.arange(max_continuation, device=device)[None, :]).clamp_max(
            max_length - 1
        )
        selected_logits = logits[
            torch.arange(len(batch), device=device)[:, None],
            token_positions,
        ].float()
        selected_log_probs = torch.log_softmax(selected_logits, dim=-1)
        selected_token_log_probs = selected_log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        valid = torch.arange(max_continuation, device=device)[None, :] < lengths[:, None]
        scores.extend((selected_token_log_probs * valid).sum(dim=1).cpu().tolist())
    return scores


def write_per_example_results(
    path: Path,
    examples: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    scores: Sequence[float],
) -> Dict[str, Any]:
    grouped: List[List[Dict[str, Any]]] = [[] for _ in examples]
    for candidate, score in zip(candidates, scores):
        enriched = dict(candidate)
        enriched["logprob"] = score
        grouped[int(candidate["example_index"])].append(enriched)

    correct = 0
    correct_norm = 0
    correct_token_norm = 0
    with path.open("w") as output:
        for example_index, (example, choices) in enumerate(zip(examples, grouped)):
            if len(choices) != 4:
                raise RuntimeError(f"example {example_index} was scored {len(choices)} times, expected 4")
            choices.sort(key=lambda item: int(item["choice_index"]))
            raw_scores = [float(item["logprob"]) for item in choices]
            token_normalized_scores = [
                float(item["logprob"]) / int(item["continuation_length"]) for item in choices
            ]
            normalized_scores = [
                float(item["logprob"]) / int(item["choice_character_length"]) for item in choices
            ]
            label = int(example["label"])
            prediction = max(range(4), key=lambda index: raw_scores[index])
            normalized_prediction = max(range(4), key=lambda index: normalized_scores[index])
            token_normalized_prediction = max(range(4), key=lambda index: token_normalized_scores[index])
            correct += int(prediction == label)
            correct_norm += int(normalized_prediction == label)
            correct_token_norm += int(token_normalized_prediction == label)
            output.write(
                json.dumps(
                    {
                        "example_index": example_index,
                        "source_ind": example.get("ind"),
                        "label": label,
                        "prediction": prediction,
                        "prediction_norm": normalized_prediction,
                        "prediction_token_norm": token_normalized_prediction,
                        "raw_scores": raw_scores,
                        "normalized_scores": normalized_scores,
                        "token_normalized_scores": token_normalized_scores,
                        "choice_character_lengths": [int(item["choice_character_length"]) for item in choices],
                        "continuation_token_lengths": [int(item["continuation_length"]) for item in choices],
                        "any_choice_truncated": any(bool(item["truncated"]) for item in choices),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    count = len(examples)
    return {
        "examples": count,
        "correct": correct,
        "correct_norm": correct_norm,
        "correct_token_norm": correct_token_norm,
        "acc": correct / count,
        "acc_norm": correct_norm / count,
        "acc_token_norm": correct_token_norm / count,
        "acc_percent": 100.0 * correct / count,
        "acc_norm_percent": 100.0 * correct_norm / count,
        "acc_token_norm_percent": 100.0 * correct_token_norm / count,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_examples < 0:
        raise ValueError("--max-examples cannot be negative")
    if args.max_examples > FULL_VALIDATION_EXAMPLES:
        raise ValueError(f"--max-examples cannot exceed {FULL_VALIDATION_EXAMPLES}")
    if args.max_examples and not args.allow_subset:
        raise ValueError("--max-examples requires --allow-subset")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (run_dir / "checkpoints" / "ckpt_final.pt").resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"final checkpoint is missing: {checkpoint_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = (
        Path(args.data).expanduser().resolve()
        if args.data
        else (output_dir / "hellaswag_val.jsonl").resolve()
    )
    if not data_path.exists():
        if not args.download_data:
            raise FileNotFoundError(f"HellaSwag data is missing: {data_path}; pass --download-data to fetch it")
        fetch_data(data_path, DEFAULT_DATA_URL, DEFAULT_DATA_SHA256)

    model_config, train_config, model_config_path, train_config_path = load_run_config(run_dir)
    examples = load_examples(data_path, DEFAULT_DATA_SHA256, args.max_examples)
    device = torch.device(args.device)
    tokenizer_name = train_config.get("tokenizer_name")
    tokenizer_revision = train_config.get("tokenizer_revision")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise RuntimeError("run config is missing tokenizer_name")
    tokenizer_kwargs = {"use_fast": True}
    if isinstance(tokenizer_revision, str) and tokenizer_revision:
        tokenizer_kwargs["revision"] = tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if len(tokenizer) != model_config.vocab_size:
        raise RuntimeError(
            f"tokenizer vocabulary ({len(tokenizer)}) does not match model vocab_size ({model_config.vocab_size})"
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    eval_config = BitNetLlamaConfig(**{**asdict(model_config), "attention_backend": args.attention_backend})
    model, checkpoint = load_model(checkpoint_path, eval_config, device, args.dtype)
    candidates, truncated_choices = prepare_candidates(examples, tokenizer, eval_config.max_seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    scores = score_candidates(model, candidates, device, args.batch_size, args.dtype, int(pad_token_id))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = write_per_example_results(output_dir / "per_example.jsonl", examples, candidates, scores)

    report: Dict[str, Any] = {
        "benchmark": "hellaswag",
        "split": "validation",
        "status": "complete",
        "final_comparable_score": not bool(args.max_examples),
        "subset": bool(args.max_examples),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": checkpoint.get("step"),
        },
        "training_run": {
            "path": str(run_dir),
            "model_config_path": str(model_config_path),
            "model_config_sha256": sha256_file(model_config_path),
            "train_config_path": str(train_config_path),
            "train_config_sha256": sha256_file(train_config_path),
            "tokenizer_name": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
        },
        "dataset": {
            "url": DEFAULT_DATA_URL,
            "revision": DEFAULT_DATA_REVISION,
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "full_validation_examples": FULL_VALIDATION_EXAMPLES,
            "evaluated_examples": len(examples),
        },
        "evaluation": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": args.dtype,
            "attention_backend": args.attention_backend,
            "batch_size_continuations": args.batch_size,
            "truncated_choices": truncated_choices,
            "wall_seconds": elapsed,
            "continuations_per_second": len(candidates) / elapsed,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
        "implementation": {
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "model_source_path": str((HERE / "model.py").resolve()),
            "model_source_sha256": sha256_file(HERE / "model.py"),
            "strict_checkpoint_load": True,
        },
        "metrics": metrics,
    }
    write_json(output_dir / "result.json", report)
    print(
        "[result] HellaSwag validation "
        f"acc={metrics['acc_percent']:.2f}% acc_norm={metrics['acc_norm_percent']:.2f}% "
        f"examples={metrics['examples']} wall={elapsed:.1f}s",
        flush=True,
    )
    print(f"[result] wrote {output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
