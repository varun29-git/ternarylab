#!/usr/bin/env python3
"""Reproducible forensic analysis for the Test 17 30B-token run.

This script deliberately distinguishes the canonical trajectory from an
abandoned pre-resume branch, aligns telemetry only at its true sampling cadence,
and emits both level and first-difference correlations.  It uses only numpy and
matplotlib so it can run in the repository's lightweight local venv.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


RUN_ID = "fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z"
HERE = Path(__file__).resolve().parent
ROOT = HERE / "source_artifacts" / RUN_ID
OUT = HERE / "forensic_outputs"
FIG = HERE / "figures"
TOKENS_PER_STEP = 114_688
GLOBAL_LAYERS = {4, 9, 14, 19}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    compressed = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if not path.exists() and compressed.exists() else Path.open
    selected = compressed if opener is gzip.open else path
    with opener(selected, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonicalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest timestamp for each step, preserving the resumed branch."""
    chosen: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row["step"])
        if step not in chosen or row["time_generated"] > chosen[step]["time_generated"]:
            chosen[step] = row
    return [chosen[step] for step in sorted(chosen)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def weighted(items: Iterable[dict[str, Any]], key: str, weight: str = "total_params") -> float:
    values = list(items)
    denom = sum(float(item[weight]) for item in values)
    return sum(float(item[key]) * float(item[weight]) for item in values) / denom


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return math.nan
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / denom) if denom else math.nan


def rankdata(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return pearson(rankdata(a[mask]), rankdata(b[mask]))


def slope(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    a = a - a.mean()
    denom = float(np.dot(a, a))
    return float(np.dot(a, b - b.mean()) / denom) if denom else math.nan


def interpolate(rows: list[dict[str, Any]], key: str, steps: np.ndarray) -> np.ndarray:
    x = np.asarray([row["step"] for row in rows], dtype=float)
    y = np.asarray([row[key] for row in rows], dtype=float)
    return np.interp(steps, x, y)


def phase_for_tokens(tokens: float) -> str:
    if tokens < 5e9:
        return "phase_1"
    if tokens < 20e9:
        return "phase_2"
    return "phase_3"


def role_of(item: dict[str, Any]) -> str:
    return str(item.get("role", "unknown"))


def snapshot_features(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    attention: list[dict[str, Any]],
    dynamics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    train_by_step = {row["step"]: row for row in train}
    attention_by_step = {row["step"]: row for row in attention}
    val_interp = interpolate(val, "val_loss", np.asarray([row["step"] for row in dynamics]))
    out: list[dict[str, Any]] = []
    for index, dyn in enumerate(dynamics):
        step = int(dyn["step"])
        tr = train_by_step[step]
        att = attention_by_step[step]
        layers = dyn["effective_bitlinear"]["layers"]
        total = sum(float(item["total_params"]) for item in layers)

        def w(key: str, selected: Iterable[dict[str, Any]] = layers) -> float:
            selected = list(selected)
            denom = sum(float(item["total_params"]) for item in selected)
            return sum(float(item[key]) * float(item["total_params"]) for item in selected) / denom

        groups = dyn["groups"]
        record = {
            "step": step,
            "tokens_b": step * TOKENS_PER_STEP / 1e9,
            "phase": phase_for_tokens(step * TOKENS_PER_STEP),
            "val_loss_interp": float(val_interp[index]),
            "train_running_loss": float(tr["running_loss"]),
            "grad_norm": float(tr["grad_norm"]),
            "alpha_mean": float(tr["alpha_stats"]["mean_abs"]),
            "alpha_min": float(tr["alpha_stats"]["min"]),
            "alpha_max": float(tr["alpha_stats"]["max"]),
            "quant_rel_mse": float(tr["quantization_mean_relative_mse"]),
            "quant_abs_mae": float(tr["quantization_mean_abs_error"]),
            "flip_rate": float(tr["weight_flip_rate"]),
            "flip_rate_ema": float(tr["weight_flip_rate_ema"]),
            "lr_q": float(tr["lr"]["muon_transformer_l0_q_proj"]),
            "lr_ffn": float(tr["lr"]["muon_transformer_l0_gate_proj"]),
            "lr_adamw": float(tr["lr"]["adamw_decay"]),
            "attention_max_logit": float(att["summary"]["max_abs_logit"]),
            "attention_entropy": float(att["summary"]["mean_normalized_entropy"]),
            "attention_min_entropy": float(att["summary"]["min_normalized_entropy"]),
            "attention_local_entropy": float(att["summary"]["by_attention_kind"]["local"]["mean_normalized_entropy"]),
            "attention_global_entropy": float(att["summary"]["by_attention_kind"]["global"]["mean_normalized_entropy"]),
            "attention_global_distant_mass": float(att["summary"]["by_attention_kind"]["global"]["mean_distant_mass"]),
            "attention_mean_near_one_hot": float(np.mean([item["near_one_hot_query_fraction"] for item in att["layers"]])),
            "attention_mean_pre_q_rms": float(np.mean([item["pre_qknorm_q_vector_rms_mean"] for item in att["layers"]])),
            "attention_mean_pre_k_rms": float(np.mean([item["pre_qknorm_k_vector_rms_mean"] for item in att["layers"]])),
            "latent_weight_rms": w("latent_weight_rms"),
            "effective_weight_rms": w("effective_weight_rms"),
            "latent_update_ratio": w("latent_update_to_weight_rms"),
            "effective_update_ratio": w("effective_update_to_weight_rms"),
            "latent_update_over_alpha": w("latent_update_over_alpha_rms"),
            "effective_update_over_alpha": w("effective_update_over_alpha_rms"),
            "zero_crossing_rate": w("sample_zero_boundary_crossing_rate"),
            "boundary_band_fraction": w("sample_boundary_band_fraction_0p05_before"),
            "saturated_tail_fraction": w("sample_saturated_tail_fraction_1p5_before"),
            "effective_value_change_rate": w("effective_value_change_rate"),
            "embedding_weight_rms": float(groups["adamw_decay"]["weight_rms_before"]),
            "embedding_update_ratio": float(groups["adamw_decay"]["update_to_weight_rms"]),
            "norm_weight_rms": float(groups["adamw_norm"]["weight_rms_before"]),
            "norm_update_ratio": float(groups["adamw_norm"]["update_to_weight_rms"]),
            "memory_reserved_gb": float(tr["memory"]["reserved_gb"]),
            "tokens_per_sec": float(tr["tokens_per_sec"]),
            "bitlinear_params": total,
        }
        for role in sorted({role_of(item) for item in layers}):
            selected = [item for item in layers if role_of(item) == role]
            record[f"role_{role}_weight_rms"] = w("latent_weight_rms", selected)
            record[f"role_{role}_update_ratio"] = w("latent_update_to_weight_rms", selected)
        out.append(record)
    return out


def analyze_correlations(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = np.asarray([row["val_loss_interp"] for row in snapshots], dtype=float)
    steps = np.asarray([row["step"] for row in snapshots], dtype=float)
    phases = [row["phase"] for row in snapshots]

    def phase_detrend(values: np.ndarray) -> np.ndarray:
        residual = np.full_like(values, np.nan, dtype=float)
        for phase in sorted(set(phases)):
            mask = np.asarray([item == phase for item in phases])
            x = steps[mask]
            y = values[mask]
            design = np.column_stack([np.ones(len(x)), x])
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            residual[mask] = y - design @ coefficients
        return residual

    target_residual = phase_detrend(target)
    metrics = [
        key for key, value in snapshots[0].items()
        if isinstance(value, (int, float)) and key not in {"step", "tokens_b", "val_loss_interp", "bitlinear_params"}
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = np.asarray([row[metric] for row in snapshots], dtype=float)
        rows.append({
            "metric": metric,
            "n": len(values),
            "pearson_level_vs_val": pearson(values, target),
            "spearman_level_vs_val": spearman(values, target),
            "pearson_first_difference_vs_val_difference": pearson(np.diff(values), np.diff(target)),
            "spearman_first_difference_vs_val_difference": spearman(np.diff(values), np.diff(target)),
            "pearson_within_phase_detrended_vs_val": pearson(phase_detrend(values), target_residual),
            "lead_metric_change_vs_next_val_change": pearson(np.diff(values)[:-1], np.diff(target)[1:]),
        })
    return sorted(rows, key=lambda row: abs(row["pearson_first_difference_vs_val_difference"]), reverse=True)


def phase_statistics(val: list[dict[str, Any]], train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for phase, lower, upper in (("phase_1", 0, 5e9), ("phase_2", 5e9, 20e9), ("phase_3", 20e9, 30.1e9)):
        vr = [row for row in val if lower <= row["step"] * TOKENS_PER_STEP < upper]
        tr = [row for row in train if lower <= row["step"] * TOKENS_PER_STEP < upper]
        xs = [row["step"] * TOKENS_PER_STEP / 1e9 for row in vr]
        result.append({
            "phase": phase,
            "token_start_b": lower / 1e9,
            "token_end_b": min(upper, 30e9) / 1e9,
            "validation_points": len(vr),
            "val_first": vr[0]["val_loss"],
            "val_last": vr[-1]["val_loss"],
            "val_best": min(row["val_loss"] for row in vr),
            "val_best_step": min(vr, key=lambda row: row["val_loss"])["step"],
            "val_slope_nats_per_billion": slope(xs, [row["val_loss"] for row in vr]),
            "running_loss_first": tr[0]["running_loss"],
            "running_loss_last": tr[-1]["running_loss"],
            "grad_norm_median": statistics.median(row["grad_norm"] for row in tr),
            "throughput_median": statistics.median(row["tokens_per_sec"] for row in tr if row["step"] > 5),
        })
    return result


def resume_analysis(raw_train: list[dict[str, Any]], raw_val: list[dict[str, Any]]) -> dict[str, Any]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_train:
        by_step[int(row["step"])].append(row)
    overlap = []
    for step, records in sorted(by_step.items()):
        if len(records) != 2:
            continue
        first, second = records
        overlap.append({
            "step": step,
            "loss_abs_delta": abs(first["loss"] - second["loss"]),
            "running_loss_abs_delta": abs(first["running_loss"] - second["running_loss"]),
            "grad_norm_abs_delta": abs(first["grad_norm"] - second["grad_norm"]),
            "grad_norm_relative_delta": abs(first["grad_norm"] - second["grad_norm"]) / max(abs(first["grad_norm"]), 1e-12),
            "first_time": first["time_generated"],
            "replay_time": second["time_generated"],
        })
    nonmonotonic = [
        (left, right) for left, right in zip(raw_train, raw_train[1:])
        if int(right["step"]) <= int(left["step"])
    ]
    left, right = nonmonotonic[0]
    gap_s = (
        datetime.fromisoformat(right["time_generated"])
        - datetime.fromisoformat(left["time_generated"])
    ).total_seconds()
    start_step = int(right["step"]) - (int(right["step"]) % 100) - 64
    replay_from = 33_136
    abandoned_high = int(left["step"])
    replay_steps = abandoned_high - replay_from
    return {
        "duplicate_train_steps": sum(len(records) - 1 for records in by_step.values()),
        "duplicate_validation_steps": len(raw_val) - len({row["step"] for row in raw_val}),
        "rollback_observed_from_step": abandoned_high,
        "resume_checkpoint_next_step": replay_from,
        "replayed_steps_to_abandoned_high_watermark": replay_steps,
        "replayed_tokens": replay_steps * TOKENS_PER_STEP,
        "estimated_replay_compute_seconds_at_168455_tps": replay_steps * TOKENS_PER_STEP / 168_455.0,
        "wall_gap_seconds_at_rollback": gap_s,
        "overlap_loss_abs_delta_mean": statistics.mean(row["loss_abs_delta"] for row in overlap),
        "overlap_loss_abs_delta_max": max(row["loss_abs_delta"] for row in overlap),
        "overlap_grad_relative_delta_mean": statistics.mean(row["grad_norm_relative_delta"] for row in overlap),
        "overlap_grad_relative_delta_max": max(row["grad_norm_relative_delta"] for row in overlap),
        "overlap_rows": overlap,
    }


def system_analysis(train: list[dict[str, Any]], raw_train: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint_interval = 26_158
    checkpoint_log_steps = {
        int(math.ceil((checkpoint_interval * index) / 100.0) * 100)
        for index in range(1, 11)
    }
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_rows: list[dict[str, Any]] = []
    for row in train:
        step = int(row["step"])
        if step <= 5:
            category = "compile_startup"
        elif step in checkpoint_log_steps:
            category = "checkpoint_interval"
        elif step % 5000 == 0:
            category = "full_telemetry"
        elif step % 2000 == 100:
            category = "post_validation"
        else:
            category = "regular"
        categories[category].append(row)
        event_rows.append({"step": step, "category": category, "tokens_per_sec": row["tokens_per_sec"]})
    summary: dict[str, Any] = {}
    baseline = statistics.median(row["tokens_per_sec"] for row in categories["regular"])
    for category, rows in categories.items():
        values = [row["tokens_per_sec"] for row in rows]
        summary[category] = {
            "n": len(values),
            "mean_tps": statistics.mean(values),
            "median_tps": statistics.median(values),
            "median_slowdown_vs_regular": 1.0 - statistics.median(values) / baseline,
        }
    times = [datetime.fromisoformat(row["time_generated"]) for row in raw_train]
    wall_s = (max(times) - min(times)).total_seconds()
    gap_s = max((b - a).total_seconds() for a, b in zip(times, times[1:]))
    cumulative_tokens = 29_990_912_000
    summary["wall"] = {
        "first_to_last_log_hours": wall_s / 3600,
        "largest_inter_log_gap_hours": gap_s / 3600,
        "effective_tps_including_gap": cumulative_tokens / wall_s,
        "effective_tps_excluding_largest_gap": cumulative_tokens / (wall_s - gap_s),
        "regular_median_tps": baseline,
        "ideal_hours_at_regular_median": cumulative_tokens / baseline / 3600,
        "observed_hours_excluding_largest_gap": (wall_s - gap_s) / 3600,
        "non_gap_overhead_vs_regular_median_hours": (wall_s - gap_s - cumulative_tokens / baseline) / 3600,
    }
    interval_tokens = 100 * TOKENS_PER_STEP
    for category in ("checkpoint_interval", "post_validation", "full_telemetry"):
        if category in summary:
            median = summary[category]["median_tps"]
            count = summary[category]["n"]
            summary[category]["estimated_overhead_seconds"] = count * interval_tokens * (1.0 / median - 1.0 / baseline)
    memory = [row["memory"] for row in train]
    summary["memory"] = {
        "peak_allocated_gb": max(row["max_allocated_gb"] for row in memory),
        "reserved_first_gb": memory[0]["reserved_gb"],
        "reserved_final_gb": memory[-1]["reserved_gb"],
        "reserved_growth_gb": memory[-1]["reserved_gb"] - memory[0]["reserved_gb"],
        "minimum_free_gb": min(row["free_gb"] for row in memory),
        "logged_allocated_final_gb": memory[-1]["allocated_gb"],
    }
    return summary, event_rows


def final_quantization(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    q_layers = metrics["quantization_error"]["layers"]
    health = {row["name"]: row for row in metrics["module_health"]["layers"]}
    out = []
    for role in sorted({row["role"] for row in q_layers}):
        qr = [row for row in q_layers if row["role"] == role]
        hr = [health[row["name"]] for row in qr]
        total = sum(row["total"] for row in qr)
        out.append({
            "role": role,
            "modules": len(qr),
            "parameters": total,
            "relative_mse": sum(row["relative_mse"] * row["total"] for row in qr) / total,
            "mean_abs_error": sum(row["mean_abs_error"] * row["total"] for row in qr) / total,
            "alpha": sum(row["alpha"] * row["total"] for row in hr) / total,
            "p_zero": sum(row["p_zero"] * row["total"] for row in hr) / total,
            "tail_fraction": sum(row["saturated_tail_fraction_1p5"] * row["total"] for row in hr) / total,
            "tail_error_fraction": sum(row["saturated_tail_error_fraction_of_total"] * row["total"] for row in hr) / total,
            "kurtosis": sum(row["normalized_abs_kurtosis"] * row["total"] for row in hr) / total,
        })
    return out


def benchmark_table() -> list[dict[str, Any]]:
    tasks = ["arc_easy", "hellaswag", "lambada", "sciq", "winogrande"]
    rows = []
    for task in tasks:
        before = read_json(ROOT / "evaluations" / "ckpt_step_209264.pt" / task / "result.json")
        after = read_json(ROOT / "evaluations" / "ckpt_final.pt" / task / "result.json")
        if task == "hellaswag":
            key, nkey = "acc_token_norm", "examples"
            a, b = before["metrics"][key], after["metrics"][key]
            n = int(after["metrics"][nkey])
        elif task == "lambada":
            key = "exact_accuracy"
            a, b, n = before[key], after[key], int(after["examples"])
        elif task == "arc_easy":
            key = "raw_accuracy"
            a, b, n = before[key], after[key], int(after["examples"])
        elif task == "sciq":
            key = "raw_accuracy"
            a, b, n = before[key], after[key], int(after["examples"])
        else:
            key = "raw_accuracy"
            a, b, n = before[key], after[key], int(after["examples"])
        unpaired_se = math.sqrt(a * (1 - a) / n + b * (1 - b) / n)
        rows.append({
            "task": task,
            "metric": key,
            "step_209264": a,
            "final": b,
            "delta_percentage_points": 100 * (b - a),
            "n": n,
            "unpaired_z_for_delta": (b - a) / unpaired_se if unpaired_se else math.nan,
            "paired_significance_available": False,
        })
    return rows


def make_figures(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    quant_roles: list[dict[str, Any]],
    system_events: list[dict[str, Any]],
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    phase_boundaries = (5.0, 20.0)

    # Validation slices expose the late regression's localization.
    x = np.asarray([row["step"] * TOKENS_PER_STEP / 1e9 for row in val])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    axes[0].plot(x, [row["val_loss"] for row in val], color="#991b1b", lw=1.5, label="32-batch validation")
    axes[0].scatter(x, [row["val_loss"] for row in val], color="#991b1b", s=9)
    for idx in range(4):
        axes[1].plot(x, [row["fixed_val_slices"][idx]["loss"] for row in val], lw=1.0, label=f"fixed slice {idx}")
    for ax in axes:
        for boundary in phase_boundaries:
            ax.axvline(boundary, color="black", alpha=0.35, ls="--", lw=0.8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("validation CE (nats)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("fixed-slice CE")
    axes[1].set_xlabel("cumulative tokens (billions)")
    axes[1].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "forensic_validation_slices.pdf", bbox_inches="tight")
    plt.close(fig)

    sx = np.asarray([row["tokens_b"] for row in snapshots])
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 11), sharex=True)
    axes[0].plot(sx, [row["val_loss_interp"] for row in snapshots], label="validation (interpolated)", color="#991b1b")
    axes[0].plot(sx, [row["train_running_loss"] for row in snapshots], label="running train loss", color="#2563eb")
    axes[1].plot(sx, [row["quant_rel_mse"] for row in snapshots], label="relative ternary MSE")
    axes[1].plot(sx, [row["flip_rate"] for row in snapshots], label="5k-step code flip rate")
    axes[2].plot(sx, [row["alpha_mean"] for row in snapshots], label="mean grouped alpha")
    axes[2].plot(sx, [row["latent_weight_rms"] for row in snapshots], label="latent RMS")
    axes[2].plot(sx, [row["effective_weight_rms"] for row in snapshots], label="effective RMS")
    axes[3].plot(sx, [row["attention_entropy"] for row in snapshots], label="attention entropy")
    axes[3].plot(sx, [row["attention_global_distant_mass"] for row in snapshots], label="global distant mass")
    for ax in axes:
        for boundary in phase_boundaries:
            ax.axvline(boundary, color="black", alpha=0.3, ls="--", lw=0.8)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("cumulative tokens (billions)")
    fig.tight_layout()
    fig.savefig(FIG / "forensic_coupled_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.8), sharex=True)
    axes[0].plot(sx, [row["latent_update_ratio"] for row in snapshots], label="latent update / latent weight")
    axes[0].plot(sx, [row["effective_update_ratio"] for row in snapshots], label="effective update / effective weight")
    axes[1].plot(sx, [row["zero_crossing_rate"] for row in snapshots], label="one-step zero-boundary crossing")
    axes[1].plot(sx, [row["boundary_band_fraction"] for row in snapshots], label="within 0.05 alpha of boundary")
    axes[1].plot(sx, [row["effective_value_change_rate"] for row in snapshots], label="effective value changed")
    axes[2].plot(sx, [row["embedding_weight_rms"] for row in snapshots], label="embedding RMS")
    axes[2].plot(sx, [row["norm_weight_rms"] for row in snapshots], label="learned norm RMS")
    for ax in axes:
        for boundary in phase_boundaries:
            ax.axvline(boundary, color="black", alpha=0.3, ls="--", lw=0.8)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("cumulative tokens (billions)")
    fig.tight_layout()
    fig.savefig(FIG / "forensic_optimizer_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)

    categories = sorted({row["category"] for row in system_events})
    colors = dict(zip(categories, plt.cm.tab10(np.linspace(0, 1, len(categories)))))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for category in categories:
        rows = [row for row in system_events if row["category"] == category]
        ax.scatter([row["step"] * TOKENS_PER_STEP / 1e9 for row in rows], [row["tokens_per_sec"] / 1000 for row in rows],
                   s=10, alpha=0.65, label=category, color=colors[category])
    ax.set_xlabel("cumulative tokens (billions)")
    ax.set_ylabel("logged throughput (K tok/s)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "forensic_system_throughput.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    roles = [row["role"] for row in quant_roles]
    axes[0].bar(roles, [row["relative_mse"] for row in quant_roles], color="#c2410c")
    axes[0].set_ylabel("final relative MSE")
    axes[1].bar(roles, [row["tail_error_fraction"] for row in quant_roles], color="#7e22ce")
    axes[1].set_ylabel("fraction of error from |w| > 1.5 alpha")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "forensic_quantization_by_role.pdf", bbox_inches="tight")
    plt.close(fig)

    top = correlations[:14]
    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    labels = [row["metric"] for row in top][::-1]
    vals = [row["pearson_first_difference_vs_val_difference"] for row in top][::-1]
    ax.barh(labels, vals, color=["#b91c1c" if value > 0 else "#1d4ed8" for value in vals])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r: first difference vs validation first difference")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "forensic_differenced_correlations.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_train = read_jsonl(ROOT / "logs" / "train.jsonl")
    raw_val = read_jsonl(ROOT / "logs" / "validation.jsonl")
    attention = read_jsonl(ROOT / "logs" / "attention.jsonl")
    dynamics = read_jsonl(ROOT / "logs" / "dynamics.jsonl")
    train = canonicalize(raw_train)
    val = canonicalize(raw_val)
    metrics = read_json(ROOT / "milestones" / "final" / "metrics.json")

    snapshots = snapshot_features(train, val, attention, dynamics)
    correlations = analyze_correlations(snapshots)
    phases = phase_statistics(val, train)
    resume = resume_analysis(raw_train, raw_val)
    systems, system_events = system_analysis(train, raw_train)
    quant_roles = final_quantization(metrics)
    benchmarks = benchmark_table()

    # Boundary-local evidence (three scheduled validations on either side).
    boundary_effects = []
    for tokens in (5e9, 20e9):
        before = [row for row in val if row["step"] * TOKENS_PER_STEP < tokens][-3:]
        after = [row for row in val if row["step"] * TOKENS_PER_STEP >= tokens][:3]
        boundary_effects.append({
            "boundary_tokens_b": tokens / 1e9,
            "before_mean": statistics.mean(row["val_loss"] for row in before),
            "after_mean": statistics.mean(row["val_loss"] for row in after),
            "raw_after_minus_before": statistics.mean(row["val_loss"] for row in after) - statistics.mean(row["val_loss"] for row in before),
            "before_steps": [row["step"] for row in before],
            "after_steps": [row["step"] for row in after],
        })

    q_rows = [row for row in train if row.get("quantization_mean_relative_mse") is not None]
    flip_rows = [row for row in train if row.get("weight_flip_rate") is not None]
    alpha_min_floor = next(row for row in train if row["alpha_stats"]["min"] <= 1.1e-6)
    alpha_mean_peak = max(train, key=lambda row: row["alpha_stats"]["mean_abs"])
    alpha_max_peak = max(train, key=lambda row: row["alpha_stats"]["max"])
    final_health = metrics["module_health"]

    summary = {
        "run_id": RUN_ID,
        "evidence_counts": {
            "train_raw": len(raw_train), "train_canonical": len(train),
            "validation_raw": len(raw_val), "validation_canonical": len(val),
            "attention": len(attention), "dynamics": len(dynamics),
            "checkpoint_tensors_present": False,
            "activation_log_present": (ROOT / "logs" / "activation.jsonl").exists(),
            "system_timeseries_present": (ROOT / "logs" / "system_timeseries.jsonl").exists(),
            "layerwise_gradient_log_present": (ROOT / "logs" / "gradients.jsonl").exists(),
        },
        "resume": {key: value for key, value in resume.items() if key != "overlap_rows"},
        "systems": systems,
        "phases": phases,
        "boundary_effects": boundary_effects,
        "quantization_temporal": {
            "first_relative_mse": q_rows[0]["quantization_mean_relative_mse"],
            "last_periodic_relative_mse": q_rows[-1]["quantization_mean_relative_mse"],
            "final_exhaustive_relative_mse": metrics["quantization_error"]["mean_relative_mse"],
            "first_flip_rate": flip_rows[0]["weight_flip_rate"],
            "peak_5k_flip_rate": max(row["weight_flip_rate"] for row in flip_rows),
            "final_5k_flip_rate": flip_rows[-1]["weight_flip_rate"],
            "final_flip_ema": train[-1]["weight_flip_rate_ema"],
            "alpha_floor_first_step": alpha_min_floor["step"],
            "alpha_mean_peak_step": alpha_mean_peak["step"],
            "alpha_mean_peak": alpha_mean_peak["alpha_stats"]["mean_abs"],
            "alpha_max_peak_step": alpha_max_peak["step"],
            "alpha_max_peak": alpha_max_peak["alpha_stats"]["max"],
            "alpha_final": metrics["alpha_stats"],
            "final_ternary": metrics["ternary"],
            "final_decomposition": metrics["quantization_decomposition"],
            "final_health_summary": final_health["summary"],
            "floor_modules": [row["name"] for row in final_health["layers"] if row["alpha_min"] <= 1.1e-6],
        },
        "attention": {
            "max_sampled_logit": max(row["summary"]["max_abs_logit"] for row in attention),
            "max_sampled_logit_step": max(attention, key=lambda row: row["summary"]["max_abs_logit"])["step"],
            "final_summary": attention[-1]["summary"],
            "all_nonfinite_logit_count": sum(item["nonfinite_logit_count"] for row in attention for item in row["layers"]),
            "post_qknorm_rms_max_deviation": max(
                abs(item[key] - 1.0)
                for row in attention for item in row["layers"]
                for key in ("post_qknorm_q_vector_rms_mean", "post_qknorm_k_vector_rms_mean")
            ),
        },
        "validation": {
            "first": val[0]["val_loss"],
            "best": min(row["val_loss"] for row in val),
            "best_step": min(val, key=lambda row: row["val_loss"])["step"],
            "last_scheduled": val[-1]["val_loss"],
            "final_milestone": metrics["val_loss"],
            "last_scheduled_minus_best": val[-1]["val_loss"] - min(row["val_loss"] for row in val),
            "final_milestone_minus_best": metrics["val_loss"] - min(row["val_loss"] for row in val),
            "late_slice_changes_252k_to_260k": [
                val[-1]["fixed_val_slices"][idx]["loss"]
                - next(row for row in val if row["step"] == 252000)["fixed_val_slices"][idx]["loss"]
                for idx in range(4)
            ],
        },
        "warnings": Counter(warning for row in train for warning in (row.get("warnings") or [])),
        "nonfinite_gradient_logs": sum(not row["grads_finite"] for row in train),
        "skipped_optimizer_steps": sum(not row["optimizer_step_applied"] for row in train),
        "optimization": {
            "gradient_norm_first": train[0]["grad_norm"],
            "gradient_norm_max": max(row["grad_norm"] for row in train),
            "gradient_norm_max_step": max(train, key=lambda row: row["grad_norm"])["step"],
            "gradient_norm_final_logged": train[-1]["grad_norm"],
            "gradient_clipping_configured": False,
            "final_realized_lr": {
                "q": train[-1]["lr"]["muon_transformer_l0_q_proj"],
                "ffn": train[-1]["lr"]["muon_transformer_l0_gate_proj"],
                "embedding": train[-1]["lr"]["adamw_decay"],
                "norm": train[-1]["lr"]["adamw_norm"],
            },
            "cumulative_lr_sum": {
                "q": train[-1]["optimizer_exposure"]["muon_transformer_l0_q_proj"]["lr_sum"],
                "ffn": train[-1]["optimizer_exposure"]["muon_transformer_l0_gate_proj"]["lr_sum"],
                "embedding": train[-1]["optimizer_exposure"]["adamw_decay"]["lr_sum"],
                "norm": train[-1]["optimizer_exposure"]["adamw_norm"]["lr_sum"],
            },
            "cumulative_lr_weighted_decay_sum": {
                "q": train[-1]["optimizer_exposure"]["muon_transformer_l0_q_proj"]["lr_weighted_decay_sum"],
                "ffn": train[-1]["optimizer_exposure"]["muon_transformer_l0_gate_proj"]["lr_weighted_decay_sum"],
                "embedding": train[-1]["optimizer_exposure"]["adamw_decay"]["lr_weighted_decay_sum"],
                "norm": train[-1]["optimizer_exposure"]["adamw_norm"]["lr_weighted_decay_sum"],
            },
            "snapshot_start": {key: snapshots[0][key] for key in (
                "latent_weight_rms", "effective_weight_rms", "latent_update_ratio",
                "effective_update_ratio", "embedding_weight_rms", "norm_weight_rms")},
            "snapshot_final": {key: snapshots[-1][key] for key in (
                "latent_weight_rms", "effective_weight_rms", "latent_update_ratio",
                "effective_update_ratio", "embedding_weight_rms", "norm_weight_rms")},
            "norm_weight_rms_min": min(row["norm_weight_rms"] for row in snapshots),
            "norm_weight_rms_min_step": min(snapshots, key=lambda row: row["norm_weight_rms"])["step"],
        },
    }
    # Counter is JSON-compatible but make the intent explicit.
    summary["warnings"] = dict(summary["warnings"])

    write_csv(OUT / "canonical_train.csv", [{
        "step": row["step"], "time_generated": row["time_generated"], "loss": row["loss"],
        "running_loss": row["running_loss"], "grad_norm": row["grad_norm"],
        "alpha_mean": row["alpha_stats"]["mean_abs"], "alpha_min": row["alpha_stats"]["min"],
        "alpha_max": row["alpha_stats"]["max"], "tokens_per_sec": row["tokens_per_sec"],
        "logged_curriculum_stage_unreliable": row["curriculum"]["stage"],
        "logged_curriculum_source_unreliable": row["curriculum"]["source"],
    } for row in train])
    write_csv(OUT / "snapshot_features_5k.csv", snapshots)
    write_csv(OUT / "correlations.csv", correlations)
    write_csv(OUT / "phase_statistics.csv", phases)
    write_csv(OUT / "resume_overlap.csv", resume["overlap_rows"])
    write_csv(OUT / "system_events.csv", system_events)
    write_csv(OUT / "final_quantization_by_role.csv", quant_roles)
    write_csv(OUT / "benchmark_comparison.csv", benchmarks)
    with (OUT / "analysis_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    make_figures(train, val, snapshots, correlations, quant_roles, system_events)
    print(json.dumps({
        "summary": str(OUT / "analysis_summary.json"),
        "canonical_train_rows": len(train),
        "snapshot_rows": len(snapshots),
        "correlation_rows": len(correlations),
    }, indent=2))


if __name__ == "__main__":
    main()
