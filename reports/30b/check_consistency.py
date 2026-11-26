#!/usr/bin/env python3
"""Check published run claims against immutable raw and derived artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVIDENCE = REPO / "provenance" / "30b-run"
RUN_ID = "fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z"
TASKS = ("arc_easy", "hellaswag", "lambada", "sciq", "winogrande")


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def equal(self, actual: Any, expected: Any, label: str) -> None:
        if actual != expected:
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
        self.count += 1

    def close(self, actual: float, expected: float, label: str) -> None:
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
        self.count += 1

    def contains(self, path: Path, text: str) -> None:
        if text not in path.read_text(encoding="utf-8"):
            raise AssertionError(f"{path.relative_to(REPO)} does not contain {text!r}")
        self.count += 1


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    compressed = path.with_suffix(path.suffix + ".gz")
    selected = path if path.exists() else compressed
    opener = gzip.open if selected.suffix == ".gz" else Path.open
    with opener(selected, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row["step"])
        if step not in selected or row["time_generated"] > selected[step]["time_generated"]:
            selected[step] = row
    return [selected[step] for step in sorted(selected)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def raw_eval_metric(task: str, payload: dict[str, Any]) -> float:
    if task == "hellaswag":
        return float(payload["metrics"]["acc_token_norm"])
    return float(payload["exact_accuracy" if task == "lambada" else "raw_accuracy"])


def main() -> None:
    checks = Checks()
    checks.equal(load_json(EVIDENCE / "RUN_ID.json")["run_id"], RUN_ID, "run ID")

    budget = load_json(EVIDENCE / "configs" / "resolved_budget.json")
    metrics = load_json(EVIDENCE / "milestones" / "final" / "metrics.json")
    model = load_json(EVIDENCE / "configs" / "model_config.json")
    environment = load_json(EVIDENCE / "configs" / "environment.json")
    checks.equal(budget["resolved_cumulative_tokens"], 29_999_996_928, "budget tokens")
    checks.equal(budget["total_steps"], 261_580, "budget steps")
    checks.equal(budget["tokens_per_step"], 114_688, "tokens per step")
    checks.equal(
        budget["total_steps"] * budget["tokens_per_step"],
        budget["schedule_total_tokens"],
        "nominal schedule token product",
    )
    checks.equal(
        min(budget["schedule_total_tokens"], budget["manifest_usable_train_tokens"]),
        budget["resolved_cumulative_tokens"],
        "usable tokens after terminal partial batch",
    )
    checks.equal(
        budget["resolved_cumulative_tokens"] - (budget["total_steps"] - 1) * budget["tokens_per_step"],
        6 * model["max_seq_len"],
        "terminal partial-batch sequences",
    )
    checks.equal(
        budget["schedule_total_tokens"] - budget["resolved_cumulative_tokens"],
        22 * model["max_seq_len"],
        "terminal shortfall from a full batch",
    )
    checks.equal(metrics["step"], 261_580, "final step")
    checks.equal(metrics["tokens_seen"], 29_999_996_928, "final tokens")
    checks.equal(metrics["final"], True, "final milestone marker")
    checks.equal((EVIDENCE / "logs" / "exit_code.txt").read_text().strip(), "0", "exit code")

    for name, expected_hash in environment["source_provenance"]["source_file_sha256"].items():
        digest = hashlib.sha256((EVIDENCE / "source" / name).read_bytes()).hexdigest()
        checks.equal(digest, expected_hash, f"frozen source hash: {name}")

    head_dim = model["dim"] // model["n_heads"]
    attention = model["dim"] ** 2 + 2 * model["n_kv_heads"] * head_dim * model["dim"] + model["dim"] ** 2
    ffn = 3 * model["dim"] * model["ffn_dim"]
    bitlinear = model["n_layers"] * (attention + ffn)
    embeddings = model["vocab_size"] * model["dim"]
    norms = model["n_layers"] * (2 * model["dim"] + 2) + model["dim"]
    total = bitlinear + embeddings + norms
    checks.equal(bitlinear, 125_829_120, "BitLinear parameter total")
    checks.equal(embeddings + norms, 24_607_528, "embedding/norm parameter total")
    checks.equal(total, 150_436_648, "model parameter total")
    checks.equal(metrics["quantization_error"]["weights_scanned"], bitlinear, "final scan weight count")

    raw_train = load_jsonl(EVIDENCE / "logs" / "train.jsonl")
    raw_validation = load_jsonl(EVIDENCE / "logs" / "validation.jsonl")
    train = canonical(raw_train)
    validation = canonical(raw_validation)
    checks.equal(len(raw_train), 2_637, "raw training rows")
    checks.equal(len(train), 2_620, "canonical training rows")
    checks.equal(len(raw_validation), 131, "raw validation rows")
    checks.equal(len(validation), 130, "canonical validation rows")
    checks.equal(train[-1]["step"], 261_500, "last canonical training log step")
    best = min(validation, key=lambda row: row["val_loss"])
    last = validation[-1]
    checks.equal(best["step"], 252_000, "best validation step")
    checks.close(best["val_loss"], 2.064144030213356, "best validation loss")
    checks.equal(last["step"], 260_000, "last scheduled validation step")
    checks.close(last["val_loss"], 2.1463008262217045, "last scheduled validation loss")
    checks.close(metrics["val_loss"], 2.1653283908963203, "final validation loss")
    checks.close(metrics["val_ppl"], 8.717464182731097, "final perplexity")
    checks.close(math.exp(metrics["val_loss"]), metrics["val_ppl"], "perplexity from loss")

    quantization = metrics["quantization_error"]
    checks.close(quantization["mean_relative_mse"], 0.42584320893511174, "final relative MSE")
    checks.close(quantization["mean_abs_error"], 0.04426160319514262, "final absolute MAE")
    checks.close(quantization["mean_relative_mae"], 0.5831257307789505, "final relative MAE")
    checks.close(metrics["ternary"]["p_zero"], 0.38338369528452554, "final zero fraction")
    checks.close(metrics["ternary"]["p_neg"], 0.3079492171605428, "final negative fraction")
    checks.close(metrics["ternary"]["p_pos"], 0.30866708755493166, "final positive fraction")
    checks.close(metrics["ternary"]["normalized_entropy"], 0.9949841125807508, "final entropy")

    summary = load_json(HERE / "forensic_outputs" / "analysis_summary.json")
    checks.equal(summary["run_id"], RUN_ID, "derived summary run ID")
    checks.equal(summary["evidence_counts"]["train_raw"], len(raw_train), "summary raw training count")
    checks.equal(summary["evidence_counts"]["train_canonical"], len(train), "summary canonical training count")
    checks.equal(summary["evidence_counts"]["validation_raw"], len(raw_validation), "summary raw validation count")
    checks.equal(summary["evidence_counts"]["validation_canonical"], len(validation), "summary canonical validation count")
    checks.close(summary["validation"]["best"], best["val_loss"], "summary best validation")
    checks.equal(summary["validation"]["best_step"], best["step"], "summary best step")
    checks.close(summary["validation"]["last_scheduled"], last["val_loss"], "summary last validation")
    checks.close(summary["validation"]["final_milestone"], metrics["val_loss"], "summary final validation")
    checks.close(
        summary["quantization_temporal"]["final_exhaustive_relative_mse"],
        quantization["mean_relative_mse"],
        "summary final MSE",
    )

    benchmark_rows = {row["task"]: row for row in read_csv(HERE / "forensic_outputs" / "benchmark_comparison.csv")}
    evaluation_rows = {row["task"]: row for row in read_csv(HERE / "tables" / "evaluation_comparison.csv")}
    checks.equal(set(benchmark_rows), set(TASKS), "forensic benchmark task set")
    checks.equal(set(evaluation_rows), set(TASKS), "technical evaluation task set")
    missing_raw_eval = []
    for task in TASKS:
        benchmark = benchmark_rows[task]
        evaluation = evaluation_rows[task]
        checks.close(100 * float(benchmark["step_209264"]), float(evaluation["prior"]), f"{task} prior table agreement")
        checks.close(100 * float(benchmark["final"]), float(evaluation["final"]), f"{task} final table agreement")
        checks.close(float(benchmark["delta_percentage_points"]), float(evaluation["delta"]), f"{task} delta table agreement")
        for checkpoint, column in (("ckpt_step_209264.pt", "step_209264"), ("ckpt_final.pt", "final")):
            path = EVIDENCE / "evaluations" / checkpoint / task / "result.json"
            if path.is_file():
                checks.close(raw_eval_metric(task, load_json(path)), float(benchmark[column]), f"{task} {checkpoint} raw agreement")
            else:
                missing_raw_eval.append(path)

    readme = REPO / "README.md"
    for text in (
        "30 billion training tokens, not 30 billion parameters",
        "150,436,648",
        "29,999,996,928",
        "2.1653283908963203",
        "8.717464182731097",
        "2.064144030213356",
        "2.1463008262217045",
        "0.42584320893511174",
    ):
        checks.contains(readme, text)
    for path in (
        REPO / "PROVENANCE.md",
        REPO / "EXPERTS_GUIDANCE.md",
        HERE / "PRECISION_FORENSIC_ANALYSIS.md",
    ):
        checks.contains(path, "150,436,648")
        checks.contains(path, "2.1653283908963203")
    checks.contains(HERE / "FORENSIC_TRAINING_RUN_PAPER.md", "150,436,648")
    checks.contains(HERE / "FORENSIC_TRAINING_RUN_PAPER.md", "final milestone measured 2.1653")
    checks.contains(HERE / "30B_TRAINING_RUN_TECHNICAL_REPORT.tex", "final held-out loss was 2.1653")
    checks.contains(HERE / "generate_mse_trend.py", "FINAL_RELATIVE_MSE = 0.42584320893511174")
    checks.contains(HERE / "generate_mae_mse_trend.py", "FINAL_MAE = 0.04426160319514262")

    print(f"PASS: {checks.count} raw/derived/document consistency checks")
    if missing_raw_eval:
        print(
            f"WARNING: {len(missing_raw_eval)} raw downstream result JSON files are absent; "
            "downstream values were checked only for agreement between the two derived CSV tables."
        )
    else:
        print("PASS: downstream metrics verified against raw result JSON")


if __name__ == "__main__":
    main()
