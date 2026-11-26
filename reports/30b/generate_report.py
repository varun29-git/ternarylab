#!/usr/bin/env python3
"""Generate a data-backed LaTeX report for the completed 30B-token run.

All values are read from the immutable committed provenance bundle. The script
deliberately fails if a required artifact is missing instead of substituting a
nearby run, a configuration default, or a derived benchmark table.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


RUN_ID = "fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "provenance" / "30b-run"
FIG = HERE / "figures"
TABLE = HERE / "tables"
EVALUATION_TASKS = ("hellaswag", "lambada", "arc_easy", "winogrande", "sciq")


def require_complete_evidence() -> None:
    run_id_path = ROOT / "RUN_ID.json"
    if not run_id_path.is_file():
        raise FileNotFoundError(f"required source artifact is absent: {run_id_path}")
    recorded_run_id = json.loads(run_id_path.read_text())["run_id"]
    if recorded_run_id != RUN_ID:
        raise RuntimeError(f"run identity mismatch: expected {RUN_ID}, got {recorded_run_id}")
    required = [
        ROOT / "configs" / "model_config.json",
        ROOT / "configs" / "train_config.json",
        ROOT / "configs" / "data_manifest.json",
        ROOT / "configs" / "resolved_budget.json",
        ROOT / "configs" / "environment.json",
        ROOT / "milestones" / "final" / "metrics.json",
        ROOT / "logs" / "validation.jsonl",
        ROOT / "logs" / "attention.jsonl",
        ROOT / "logs" / "dynamics.jsonl",
    ]
    if not (ROOT / "logs" / "train.jsonl").exists():
        required.append(ROOT / "logs" / "train.jsonl.gz")
    for checkpoint in ("ckpt_step_209264.pt", "ckpt_final.pt"):
        required.extend(
            ROOT / "evaluations" / checkpoint / task / "result.json"
            for task in EVALUATION_TASKS
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required source artifacts are absent; refusing partial report regeneration:\n"
            + "\n".join(str(path) for path in missing)
        )


def load_json(rel: str) -> dict[str, Any]:
    with (ROOT / rel).open() as handle:
        return json.load(handle)


def load_jsonl(rel: str) -> list[dict[str, Any]]:
    path = ROOT / rel
    compressed = path.with_suffix(path.suffix + ".gz")
    if not path.exists() and not compressed.exists():
        raise FileNotFoundError(f"required source artifact is absent: {path}")
    opener = gzip.open if not path.exists() else Path.open
    selected = compressed if opener is gzip.open else path
    with opener(selected, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def weighted_mean(items: Iterable[tuple[float, float]]) -> float:
    values = list(items)
    denom = sum(w for _, w in values)
    return sum(v * w for v, w in values) / denom if denom else math.nan


def layer_from_group(name: str) -> int | None:
    match = re.search(r"_l(\d+)_", name)
    return int(match.group(1)) if match else None


def kind(layer: int) -> str:
    return "global" if layer in {4, 9, 14, 19} else "local"


def b(value: float) -> float:
    return value / 1e9


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fig_save(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()


def make_figures(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    attention: list[dict[str, Any]],
    dynamics: list[dict[str, Any]],
    tokens_per_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    steps = np.array([row["step"] for row in train])
    tokens_b = steps * tokens_per_step / 1e9
    raw_loss = np.array([row["loss"] for row in train])
    running_loss = np.array([row["running_loss"] for row in train])
    grad = np.array([row["grad_norm"] for row in train])
    q_mse = np.array([row.get("quantization_mean_relative_mse", np.nan) for row in train])
    q_mae = np.array([row.get("quantization_mean_abs_error", np.nan) for row in train])
    q_rmae = np.array([row.get("quantization_mean_relative_mae", np.nan) for row in train])
    flips = np.array([row["weight_flip_rate_ema"] for row in train])
    q_lr = np.array([row["lr"].get("muon_transformer_l0_q_proj", np.nan) for row in train])
    ffn_lr = np.array([row["lr"].get("muon_transformer_l0_gate_proj", np.nan) for row in train])
    adamw_lr = np.array([row["lr"].get("adamw_decay", np.nan) for row in train])
    val_steps = np.array([row["step"] for row in val])
    val_tokens_b = val_steps * tokens_per_step / 1e9
    val_loss = np.array([row["val_loss"] for row in val])

    # Global dynamics: raw loss is deliberately shown faintly because batches
    # are drawn from a changing curriculum; running loss is the logged EMA.
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 10.2), sharex=True)
    axes[0].plot(tokens_b, raw_loss, color="#9ab9d6", alpha=0.35, lw=0.55, label="batch loss")
    axes[0].plot(tokens_b, running_loss, color="#0b4f8a", lw=1.25, label="logged running loss")
    axes[0].plot(val_tokens_b, val_loss, color="#bf3f2f", marker="o", ms=2.5, lw=1.1, label="held-out validation")
    axes[0].set_ylabel("cross-entropy (nats)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(tokens_b, grad, color="#5a3c99", lw=1.0)
    axes[1].set_ylabel("global gradient norm")
    axes[1].grid(alpha=0.2)
    axes[2].plot(tokens_b, q_lr, label="Muon Q/K/O effective LR", color="#2c7fb8")
    axes[2].plot(tokens_b, ffn_lr, label="Muon FFN effective LR", color="#41ab5d")
    axes[2].plot(tokens_b, adamw_lr, label="AdamW embedding LR", color="#dd8d2a")
    axes[2].set_xlabel("cumulative tokens (billions)")
    axes[2].set_ylabel("realized group LR")
    axes[2].legend(fontsize=8, ncol=3)
    axes[2].grid(alpha=0.2)
    for ax in axes:
        for boundary in (5, 20):
            ax.axvline(boundary, color="black", alpha=0.25, lw=0.8, ls="--")
    fig_save("global_training_dynamics.pdf")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True)
    axes[0].plot(tokens_b, q_mse, color="#8c2d04", lw=1.3, label="relative MSE")
    axes[0].plot(tokens_b, q_rmae, color="#cc4c02", lw=1.0, label="relative MAE")
    axes[0].set_ylabel("relative reconstruction error")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(tokens_b, q_mae, color="#e6550d", label="absolute MAE")
    axes[1].plot(tokens_b, flips, color="#3182bd", label="ternary flip-rate EMA")
    axes[1].set_xlabel("cumulative tokens (billions)")
    axes[1].set_ylabel("value")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    for ax in axes:
        for boundary in (5, 20):
            ax.axvline(boundary, color="black", alpha=0.25, lw=0.8, ls="--")
    fig_save("quantization_dynamics.pdf")

    attention_steps = np.array([row["step"] for row in attention])
    attention_tokens = attention_steps * tokens_per_step / 1e9
    max_logit = np.array([row["summary"]["max_abs_logit"] for row in attention])
    mean_entropy = np.array([row["summary"]["mean_normalized_entropy"] for row in attention])
    min_entropy = np.array([row["summary"]["min_normalized_entropy"] for row in attention])
    local_logit = np.array([row["summary"]["by_attention_kind"]["local"]["max_abs_logit"] for row in attention])
    global_logit = np.array([row["summary"]["by_attention_kind"]["global"]["max_abs_logit"] for row in attention])
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True)
    axes[0].plot(attention_tokens, max_logit, color="#7f0000", label="all-layer max")
    axes[0].plot(attention_tokens, local_logit, color="#cb181d", ls="--", label="local max")
    axes[0].plot(attention_tokens, global_logit, color="#2171b5", ls="--", label="global max")
    axes[0].set_ylabel("sampled max |logit|")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(alpha=0.2)
    axes[1].plot(attention_tokens, mean_entropy, color="#238b45", label="mean normalized entropy")
    axes[1].plot(attention_tokens, min_entropy, color="#006d2c", ls="--", label="minimum layer entropy")
    axes[1].set_xlabel("cumulative tokens (billions)")
    axes[1].set_ylabel("normalized attention entropy")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    for ax in axes:
        for boundary in (5, 20):
            ax.axvline(boundary, color="black", alpha=0.25, lw=0.8, ls="--")
    fig_save("attention_global_dynamics.pdf")

    # Full per-layer telemetry, not the summary's top-five view.
    def matrix(metric: str) -> np.ndarray:
        out = np.full((20, len(attention)), np.nan)
        for col, record in enumerate(attention):
            for item in record["layers"]:
                out[int(item["layer"]), col] = item[metric]
        return out

    logit_mat = matrix("max_abs_logit")
    entropy_mat = matrix("normalized_entropy")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True)
    for ax, values, label, cmap in (
        (axes[0], logit_mat, "sampled max |attention logit|", "magma"),
        (axes[1], entropy_mat, "normalized attention entropy", "viridis"),
    ):
        image = ax.imshow(values, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap,
                          extent=[attention_tokens[0], attention_tokens[-1], -0.5, 19.5])
        ax.set_ylabel("layer index")
        fig.colorbar(image, ax=ax, label=label)
        for layer in (4, 9, 14, 19):
            ax.axhline(layer, color="white", lw=0.55, alpha=0.65)
    axes[1].set_xlabel("cumulative tokens (billions)")
    fig_save("attention_layerwise_heatmaps.pdf")

    final_attention = attention[-1]["layers"]
    final_attention.sort(key=lambda item: item["layer"])
    layer_ids = np.array([item["layer"] for item in final_attention])
    final_rows: list[dict[str, Any]] = []
    for item in final_attention:
        final_rows.append({
            "layer": item["layer"], "attention_kind": item["attention_kind"],
            "max_abs_logit": item["max_abs_logit"], "p999_logit": item["p999_logit"],
            "normalized_entropy": item["normalized_entropy"],
            "mean_max_attention_probability": item["mean_max_attention_probability"],
            "near_one_hot_query_fraction": item["near_one_hot_query_fraction"],
            "low_entropy_query_fraction": item["low_normalized_entropy_query_fraction"],
            "post_rope_q_rms": item["post_rope_q_vector_rms_mean"],
            "post_rope_k_rms": item["post_rope_k_vector_rms_mean"],
            "distant_mass": item["distant_mass"],
        })
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True)
    axes[0, 0].bar(layer_ids, [x["max_abs_logit"] for x in final_attention], color="#9e0142")
    axes[0, 0].set_ylabel("max |logit|")
    axes[0, 1].bar(layer_ids, [x["normalized_entropy"] for x in final_attention], color="#3288bd")
    axes[0, 1].set_ylabel("normalized entropy")
    axes[1, 0].bar(layer_ids, [x["mean_max_attention_probability"] for x in final_attention], color="#66c2a5", label="mean max probability")
    axes[1, 0].plot(layer_ids, [x["near_one_hot_query_fraction"] for x in final_attention], color="#d53e4f", marker="o", ms=3, label="near-one-hot fraction")
    axes[1, 0].set_ylabel("probability / fraction")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot(layer_ids, [x["post_rope_q_vector_rms_mean"] for x in final_attention], marker="o", ms=3, label="Q RMS")
    axes[1, 1].plot(layer_ids, [x["post_rope_k_vector_rms_mean"] for x in final_attention], marker="s", ms=3, label="K RMS")
    axes[1, 1].set_ylabel("post-RoPE vector RMS")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        for layer in (4, 9, 14, 19):
            ax.axvline(layer, color="black", alpha=0.2, lw=0.7, ls="--")
    for ax in axes[1, :]:
        ax.set_xlabel("layer index")
    fig_save("final_layerwise_attention.pdf")

    # Aggregate all BitLinear matrices in a layer using their logged parameter
    # counts. This is an explicit weighted mean, not a cherry-picked matrix.
    dynamic_rows: list[dict[str, Any]] = []
    for record in dynamics:
        per_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for name, metric in record["groups"].items():
            layer = layer_from_group(name)
            if layer is not None:
                per_layer[layer].append(metric)
        for layer, group in per_layer.items():
            dynamic_rows.append({
                "step": record["step"], "tokens_b": b(record["step"] * tokens_per_step), "layer": layer,
                "weight_rms": weighted_mean((x["weight_rms_before"], x["total_params"]) for x in group),
                "update_rms": weighted_mean((x["update_rms"], x["total_params"]) for x in group),
                "update_to_weight_rms": weighted_mean((x["update_to_weight_rms"], x["total_params"]) for x in group),
                "sign_change_rate": weighted_mean((x["sample_sign_change_rate"], x["total_params"]) for x in group),
            })
    dsteps = sorted({row["step"] for row in dynamic_rows})
    def dyn_matrix(metric: str) -> np.ndarray:
        out = np.full((20, len(dsteps)), np.nan)
        for row in dynamic_rows:
            out[row["layer"], dsteps.index(row["step"])] = row[metric]
        return out
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.2), sharex=True)
    for ax, metric, title, cmap in (
        (axes[0], "weight_rms", "latent weight RMS", "cividis"),
        (axes[1], "update_to_weight_rms", "update RMS / weight RMS", "magma"),
        (axes[2], "sign_change_rate", "sampled sign-change rate", "plasma"),
    ):
        image = ax.imshow(dyn_matrix(metric), aspect="auto", origin="lower", interpolation="nearest", cmap=cmap,
                          extent=[b(dsteps[0] * tokens_per_step), b(dsteps[-1] * tokens_per_step), -0.5, 19.5])
        ax.set_ylabel("layer index")
        fig.colorbar(image, ax=ax, label=title)
    axes[-1].set_xlabel("cumulative tokens (billions)")
    fig_save("layerwise_update_dynamics.pdf")
    return final_rows, dynamic_rows, [
        {"step": row["step"], "tokens_b": b(row["step"] * tokens_per_step), "train_loss": row["loss"],
         "running_loss": row["running_loss"], "grad_norm": row["grad_norm"],
         "quant_relative_mse": row.get("quantization_mean_relative_mse"),
         "quant_abs_mae": row.get("quantization_mean_abs_error"),
         "flip_rate_ema": row["weight_flip_rate_ema"],
         "q_proj_effective_lr": row["lr"].get("muon_transformer_l0_q_proj"),
         "ffn_effective_lr": row["lr"].get("muon_transformer_l0_gate_proj"),
         "adamw_embedding_lr": row["lr"].get("adamw_decay")}
        for row in train
    ]


def final_quantization_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    error = metrics["quantization_error"]["layers"]
    health = {item["name"]: item for item in metrics["module_health"]["layers"]}
    grouped: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in error:
        grouped[int(item["layer_index"])].append((item, health[item["name"]]))
    rows: list[dict[str, Any]] = []
    for layer, values in sorted(grouped.items()):
        rows.append({
            "layer": layer, "attention_kind": kind(layer),
            "relative_mse": weighted_mean((e["relative_mse"], e["total"]) for e, _ in values),
            "absolute_mae": weighted_mean((e["mean_abs_error"], e["total"]) for e, _ in values),
            "relative_mae": weighted_mean((e["relative_mae"], e["total"]) for e, _ in values),
            "zero_fraction": weighted_mean((h["zero_cell_fraction"], h["total"]) for _, h in values),
            "tail_error_fraction": weighted_mean((h["saturated_tail_error_fraction_of_total"], h["total"]) for _, h in values),
            "latent_rms": weighted_mean((h["latent_rms"], h["total"]) for _, h in values),
        })
    return rows


def quantization_figure(rows: list[dict[str, Any]]) -> None:
    layers = [row["layer"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True)
    specs = [
        ("relative_mse", "relative MSE", "#b2182b"),
        ("absolute_mae", "absolute MAE", "#ef8a62"),
        ("zero_fraction", "ternary zero fraction", "#2166ac"),
        ("tail_error_fraction", "error in |w| > 1.5 alpha", "#762a83"),
    ]
    for ax, (key, label, color) in zip(axes.flat, specs):
        ax.bar(layers, [row[key] for row in rows], color=color)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        for layer in (4, 9, 14, 19):
            ax.axvline(layer, color="black", alpha=0.2, lw=0.7, ls="--")
    for ax in axes[1, :]:
        ax.set_xlabel("layer index")
    fig_save("final_layerwise_quantization.pdf")


def eval_metric(task: str, payload: dict[str, Any]) -> tuple[str, float]:
    if task == "hellaswag":
        return "token-normalized accuracy", payload["metrics"]["acc_token_norm"] * 100
    if task == "lambada":
        return "exact accuracy", payload["exact_accuracy"] * 100
    if task == "arc_easy":
        return "raw accuracy", payload["raw_accuracy"] * 100
    if task == "winogrande":
        return "raw accuracy", payload["raw_accuracy"] * 100
    if task == "sciq":
        return "raw accuracy", payload["raw_accuracy"] * 100
    raise KeyError(task)


def tex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def write_tex(
    train: list[dict[str, Any]], val: list[dict[str, Any]], attention: list[dict[str, Any]],
    dynamics: list[dict[str, Any]], metrics: dict[str, Any], model: dict[str, Any],
    config: dict[str, Any], manifest: dict[str, Any], budget: dict[str, Any], env: dict[str, Any],
    final_attention: list[dict[str, Any]], final_quant: list[dict[str, Any]], dynamic_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    final_val = metrics["val_loss"]
    best_val = min(val, key=lambda x: x["val_loss"])
    last_val = val[-1]
    final_att = attention[-1]
    final_dyn_step = dynamics[-1]["step"]
    final_q = metrics["quantization_error"]
    qd = metrics["quantization_decomposition"]
    ternary = metrics["ternary"]
    health = metrics["module_health"]["summary"]
    max_logit_layer = max(final_attention, key=lambda x: x["max_abs_logit"])
    min_entropy_layer = min(final_attention, key=lambda x: x["normalized_entropy"])
    max_onehot_layer = max(final_attention, key=lambda x: x["near_one_hot_query_fraction"])
    highest_q = max(final_quant, key=lambda x: x["relative_mse"])
    highest_zero = max(final_quant, key=lambda x: x["zero_fraction"])
    last_train = train[-1]
    max_grad = max(train, key=lambda x: x["grad_norm"])
    max_loss = max(train, key=lambda x: x["loss"])
    min_loss = min(train, key=lambda x: x["loss"])
    phase = manifest["config"]["phase_source_mixes"]
    phase_text = "\\\\ ".join(
        f"{tex_escape(p['name'])}: {p['tokens']/1e9:.0f}B tokens; " + ", ".join(
            f"{tex_escape(k)} {100*v:.1f}\\%" for k, v in p["source_mix"].items()
        ) for p in phase
    )
    attention_outlier_rows = [
        ("highest sampled logit", max_logit_layer),
        ("lowest entropy", min_entropy_layer),
        ("highest near-one-hot fraction", max_onehot_layer),
    ]
    update_layer = max((row for row in dynamic_rows if row["step"] == final_dyn_step), key=lambda x: x["update_to_weight_rms"])

    table_eval = "\n".join(
        f"{tex_escape(r['task'])} & {r['metric']} & {r['prior']:.2f} & {r['final']:.2f} & {r['delta']:+.2f} \\\\" for r in eval_rows
    )
    table_attention = "\n".join(
        f"{label} & {item['layer']} & {item['attention_kind']} & {item['max_abs_logit']:.3f} & {item['normalized_entropy']:.3f} & {100*item['near_one_hot_query_fraction']:.2f}\\% \\\\" for label, item in attention_outlier_rows
    )
    table_quant = "\n".join(
        f"{row['layer']} & {row['attention_kind']} & {row['relative_mse']:.3f} & {row['absolute_mae']:.4f} & {100*row['zero_fraction']:.2f}\\% & {100*row['tail_error_fraction']:.2f}\\% \\\\" for row in final_quant
    )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{graphicx,booktabs,amsmath,amssymb,hyperref,caption,subcaption,float,xcolor,longtable}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}}
\title{{Technical Report: 30B-Token FinalArchitecture Transformer++ Training Run}}
\author{{Artifact-derived analysis of the canonical run}}
\date{{Generated from the completed run artifacts}}
\begin{{document}}
\maketitle

\begin{{abstract}}
This report analyzes the completed 30B-token training run identified in the artifact inventory. The run trained a 150.44M-parameter ternary Transformer++ for {budget['resolved_cumulative_tokens'] / 1e9:.3f}B cumulative tokens and exited with code 0. The final held-out loss was {final_val:.4f} nats (perplexity {metrics['val_ppl']:.3f}); the best scheduled validation loss was {best_val['val_loss']:.4f} at step {best_val['step']}. Attention telemetry remained finite: its largest sampled absolute logit was {max(row['summary']['max_abs_logit'] for row in attention):.3f}, while the final sampled maximum was {final_att['summary']['max_abs_logit']:.3f}. The run was numerically stable under the logged observables, but its final relative ternary reconstruction MSE was {final_q['mean_relative_mse']:.3f}. The configuration contains no numerical pass/fail threshold for this quantity, so the artifacts establish the measurement but do not establish a formal MSE gate pass.
\end{{abstract}}

\section{{Run identity and evidence boundary}}
This paper uses the immutable committed artifacts under \texttt{{provenance/30b-run/}}, named by the run identifier recorded in \texttt{{RUN\_ID.json}}. Identity is supported by the resolved cumulative token count ({budget['resolved_cumulative_tokens']:,}), the final checkpoint at step {metrics['step']:,}, the final milestone record, and \texttt{{exit\_code.txt}}. The run resumed from step {budget['start_step']:,}; the reporting series therefore includes the full cumulative trajectory through the final step rather than treating the resumed segment as an independent experiment. All figures were generated by \texttt{{generate\_report.py}} from the raw JSON/JSONL files. Machine-readable derived tables are emitted in \texttt{{tables/}}.

\section{{Setup discovered from run artifacts}}
The model configuration reports {model['n_layers']} Transformer layers, hidden width {model['dim']}, {model['n_heads']} query heads, {model['n_kv_heads']} KV heads, and FFN width {model['ffn_dim']}. A parameter-count audit of this exact configuration gives 150,436,648 trainable parameters: 125,829,120 BitLinear matrix parameters and 24,607,528 embedding/norm parameters. The attention policy is sixteen local layers and four global layers: global layers $\{{4,9,14,19\}}$ and a local window of {model['local_attention_window']} tokens. The sequence length is {model['max_seq_len']}; the global-to-local layer ratio is 1:4.

\begin{{table}}[H]\centering
\caption{{Discovered training configuration.}}\label{{tab:setup}}
\begin{{tabular}}{{ll}}\toprule
Item & Recorded value \\\midrule
Tokenizer & \texttt{{{tex_escape(config['tokenizer_name'])}}}, revision \texttt{{{config['tokenizer_revision'][:12]}...{config['tokenizer_revision'][-8:]}}} \\
Precision / device & {config['dtype'].upper()} on {tex_escape(env['cuda_device']['name'])} ({env['cuda_device']['total_memory_gb']:.1f} GB) \\
Batch / sequence / tokens per step & {config['batch_size_override']} / {model['max_seq_len']} / {budget['tokens_per_step']:,} \\
Total budget / final step & {budget['resolved_cumulative_tokens'] / 1e9:.3f}B tokens / {metrics['step']:,} \\
Attention & FA3; local window {model['local_attention_window']}; globals 4, 9, 14, 19 \\
Normalization & unit-RMS Q/K before RoPE; SubLN enabled; residual init scale {model['residual_init_scale']:.1f} \\
Weight quantization & deterministic grouped absmean; group size {model['alpha_group_size']}; activation quantization disabled \\
Optimizers & Muon for BitLinear matrices; AdamW for embeddings and RMSNorm weights \\
Base learning rates & Muon {config['muon_lr']:.4f}; AdamW {config['adamw_lr']:.4f}; norm {config['norm_lr']:.4f} \\
Schedule / decay & cosine; warmup {100*config['warmup_fraction']:.1f}\%; terminal LR fraction {config['final_lr_fraction']:.2f} \\
Weight decay & Muon {config['muon_weight_decay']:.2f}; AdamW {config['adamw_weight_decay']:.2f} \\
\bottomrule\end{{tabular}}
\end{{table}}

The data manifest defines the following phase-wise mixture: {phase_text}. Phase boundaries at 5B and 20B cumulative tokens are marked in Figs.~\ref{{fig:global}}--\ref{{fig:quant}}.

\section{{Global training dynamics}}
Figure~\ref{{fig:global}} separates raw batch loss from the logged running loss and held-out validation loss. The first logged batch loss was {train[0]['loss']:.4f}; the final logged batch loss was {last_train['loss']:.4f}; the final running loss was {last_train['running_loss']:.4f}. Held-out validation fell from {val[0]['val_loss']:.4f} at step {val[0]['step']:,} to {last_val['val_loss']:.4f} at the final scheduled validation step {last_val['step']:,}. The best scheduled value was {best_val['val_loss']:.4f} at step {best_val['step']:,}; the final milestone value was {final_val:.4f}, {100*(final_val / best_val['val_loss'] - 1):.2f}\% above that best point. This late increase is measured, but the logs do not contain a causal annotation that permits attribution to a specific data component or schedule event.

The global gradient norm declined from {train[0]['grad_norm']:.3f} at the first logged step to {last_train['grad_norm']:.3f} at the final logged step. Its maximum, {max_grad['grad_norm']:.3f}, occurred at step {max_grad['step']:,}. The highest raw batch loss, {max_loss['loss']:.3f}, occurred at step {max_loss['step']:,}; the minimum, {min_loss['loss']:.3f}, occurred at step {min_loss['step']:,}. There is no logged NaN/Inf event in the analyzed series. The realized matrix learning rates in Fig.~\ref{{fig:global}} are larger than the base Muon LR because the recorded RMS-matching multiplier is applied per matrix.

\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/global_training_dynamics.pdf}}
\caption{{Loss, gradient norm, and realized optimizer-group learning rates across the full cumulative trajectory. Dashed vertical lines mark the 5B- and 20B-token curriculum boundaries.}}\label{{fig:global}}
\end{{figure}}

\section{{Quantization and numerical stability}}
The model uses ternary weight quantization and no activation quantization. Figure~\ref{{fig:quant}} reports the logged reconstruction series. Relative MSE increased from {next(x['quantization_mean_relative_mse'] for x in train if x.get('quantization_mean_relative_mse') is not None):.3f} at the first quantified point to {final_q['mean_relative_mse']:.3f} at the final exhaustive milestone measurement; final absolute MAE was {final_q['mean_abs_error']:.4f}. The final ternary distribution was $p(-1)={ternary['p_neg']:.3f}$, $p(0)={ternary['p_zero']:.3f}$, and $p(+1)={ternary['p_pos']:.3f}$, with normalized ternary entropy {ternary['normalized_entropy']:.5f}. The final flip-rate EMA was {last_train['weight_flip_rate_ema']:.3f}.

The configuration sets \texttt{{max\_quantization\_relative\_mse=null}}, \texttt{{min\_ternary\_zero\_fraction=null}}, and \texttt{{max\_ternary\_zero\_fraction=null}}. Consequently, the artifact record contains no numerical quantization gate. The measured final relative MSE of {final_q['mean_relative_mse']:.3f} is therefore \emph{{not formally classifiable as pass or fail}} from this run alone.

The decomposition records an oracle relative MSE of {qd['oracle_relative_mse']:.3f}, scale-gap contribution {qd['scale_gap_relative_mse']:.3f}, support-gap contribution {qd['support_gap_relative_mse']:.3f}, and identifies \texttt{{{qd['dominant_component']}}} as the largest component. The module-health scan attributes {100*health['mean_saturated_tail_error_fraction_of_total']:.1f}\% of mean reconstruction error to weights with magnitude above $1.5\alpha$, while zero-cell error contributes {100*health['mean_zero_cell_error_fraction_of_total']:.1f}\%. These are reconstruction diagnostics, not evidence of non-finite numerical behavior.

\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/quantization_dynamics.pdf}}
\caption{{Logged quantization error and ternary flip dynamics. No MSE or occupancy threshold was configured, so no line is drawn as a pass/fail gate.}}\label{{fig:quant}}
\end{{figure}}

\section{{Layerwise analysis}}
All 20 layers are represented in the telemetry, not only the top-layer summary. Figure~\ref{{fig:attheat}} shows the complete 52-snapshot layer-by-time logit and entropy matrices. The largest sampled absolute logit across the run was {max(row['summary']['max_abs_logit'] for row in attention):.3f}; it occurred before the final snapshot. The final snapshot at step {final_att['step']:,} has maximum sampled absolute logit {final_att['summary']['max_abs_logit']:.3f}, mean normalized entropy {final_att['summary']['mean_normalized_entropy']:.3f}, and minimum layer entropy {final_att['summary']['min_normalized_entropy']:.3f}. Every logged layer reports finite-logit fraction 1.0 and zero non-finite logits.

At the final snapshot, layer {max_logit_layer['layer']} ({max_logit_layer['attention_kind']}) has the largest sampled absolute logit, {max_logit_layer['max_abs_logit']:.3f}. Layer {min_entropy_layer['layer']} ({min_entropy_layer['attention_kind']}) has the lowest normalized entropy, {min_entropy_layer['normalized_entropy']:.3f}. Layer {max_onehot_layer['layer']} has the largest near-one-hot query fraction, {100*max_onehot_layer['near_one_hot_query_fraction']:.2f}\%. These are not a sink diagnosis: the telemetry contains no explicit sink-token metric, and the sampled low-entropy fraction stays below 1\% in every final layer. Post-RoPE Q and K vector RMS values remain close to 1.0 by construction and are displayed in Fig.~\ref{{fig:attfinal}}.

\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/attention_global_dynamics.pdf}}
\caption{{Aggregate attention telemetry over time.}}\label{{fig:attglobal}}
\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/attention_layerwise_heatmaps.pdf}}
\caption{{Full layer-by-time attention telemetry. White horizontal guides identify global layers.}}\label{{fig:attheat}}
\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/final_layerwise_attention.pdf}}
\caption{{Final sampled layerwise attention statistics at step {final_att['step']:,}.}}\label{{fig:attfinal}}
\end{{figure}}

Figure~\ref{{fig:updates}} aggregates every BitLinear matrix within each layer with parameter-count weighting. At the last update-stat snapshot (step {final_dyn_step:,}), layer {update_layer['layer']} has the largest update-to-weight RMS ratio ({update_layer['update_to_weight_rms']:.6f}). This is an update-scale outlier, not a divergence signal: the full layerwise maximum remains below $10^{{-3}}$ and all logged updates are finite. Weight RMS, update ratios, and sampled sign-change rates are shown for all layers in Fig.~\ref{{fig:updates}}.

\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/layerwise_update_dynamics.pdf}}
\caption{{Parameter-count-weighted BitLinear update diagnostics by layer.}}\label{{fig:updates}}
\end{{figure}}

The final quantization scan covers {final_q['weights_scanned']:,} weights. Layer {highest_q['layer']} ({highest_q['attention_kind']}) has the largest aggregate relative MSE ({highest_q['relative_mse']:.3f}); layer {highest_zero['layer']} has the largest aggregate zero fraction ({100*highest_zero['zero_fraction']:.2f}\%). These are reconstruction outliers. They do not coincide with the largest attention logit, and the artifact does not log a causal linkage between reconstruction error and attention behavior.

\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/final_layerwise_quantization.pdf}}
\caption{{Final full-scan, parameter-count-weighted quantization diagnostics by layer.}}\label{{fig:qfinal}}
\end{{figure}}

\begin{{table}}[H]\centering
\caption{{Final attention outliers from the full layerwise snapshot.}}\label{{tab:attout}}
\begin{{tabular}}{{lrrrrr}}\toprule
Criterion & Layer & Kind & max $|z|$ & Entropy & near-one-hot \\\midrule
{table_attention}
\bottomrule\end{{tabular}}
\end{{table}}

\section{{Evaluation results}}
Table~\ref{{tab:eval}} compares the final checkpoint with the preceding evaluated checkpoint at step 209,264 using the surviving derived evaluation tables. The original analysis recorded the same checked-in model source hash and SDPA reference backend on an NVIDIA L40S for both checkpoints. HellaSwag token-normalized accuracy, LAMBADA exact accuracy, ARC-Easy raw accuracy, and WinoGrande raw accuracy increased. SciQ raw accuracy decreased by 0.70 percentage points. The evidence supports mixed late-stage downstream movement, not a claim that every benchmark improved. The underlying raw result JSON files are absent from this checkout, so these values are not independently raw-reverified here.

\begin{{table}}[H]\centering
\caption{{Final checkpoint versus step-209,264 evaluation. Values are percentage points.}}\label{{tab:eval}}
\begin{{tabular}}{{llrrr}}\toprule
Task & Metric & Step 209,264 & Final & $\Delta$ \\\midrule
{table_eval}
\bottomrule\end{{tabular}}
\end{{table}}

\section{{Discussion}}
The logged numerical trajectory is stable in the limited, observable sense: the run completed with exit code 0; all sampled attention logits were finite; the maximum sampled absolute logit stayed in the interval [{min(row['summary']['max_abs_logit'] for row in attention):.3f}, {max(row['summary']['max_abs_logit'] for row in attention):.3f}]; post-RoPE Q/K RMS stayed approximately one; and update-to-weight RMS remained below $10^{{-3}}$ at the final update snapshot. The data do not support calling the final phase strictly improving on held-out loss, because the best scheduled validation was {best_val['val_loss']:.4f} and the final milestone was {final_val:.4f}.

Quantization reconstruction error remains the clearest unresolved scale risk: final relative MSE was {final_q['mean_relative_mse']:.3f}, with {100*health['mean_saturated_tail_error_fraction_of_total']:.1f}\% of mean error assigned to the saturated tail. However, this run configured no numerical quantization gate, so a formal quality verdict cannot be inferred from MSE alone. Before a larger scale-up, the next run should preserve the current full layerwise attention telemetry and add an explicit, pre-registered quantization criterion with an allowed MSE/MAE range and a downstream-evaluation decision rule.

\section{{Limitations}}
The attention statistics are sampled over one batch and 256 evenly spaced sequence positions, not an exhaustive $4096\times4096$ matrix. There is no explicit sink-token counter, no per-layer activation norm for the FFN/residual stream, and no configured numerical MSE, zero-fraction, or attention-logit pass/fail threshold. Therefore this report can establish observed finiteness and layerwise sampled behavior, but cannot certify an unlogged sink absence, full-sequence attention distribution, or formal gate clearance. Model checkpoints, token memmaps, and raw downstream result JSON files are external to this checkout. The downstream table is preserved derived evidence and cannot be independently regenerated from the committed raw artifacts.

\appendix
\section{{Full final layerwise quantization table}}
\begin{{longtable}}{{rrrrrr}}
\caption{{Final full-scan, parameter-count-weighted layer metrics. The complete time-series CSV supplements are in \texttt{{tables/}}.}}\\
\toprule Layer & Kind & rel. MSE & abs. MAE & zero frac. & tail-error frac. \\\midrule
\endfirsthead
\toprule Layer & Kind & rel. MSE & abs. MAE & zero frac. & tail-error frac. \\\midrule
\endhead
{table_quant}
\bottomrule
\end{{longtable}}

\section{{Artifact inventory}}
The report was originally generated from \texttt{{configs/model\_config.json}}, \texttt{{configs/train\_config.json}}, \texttt{{configs/data\_manifest.json}}, \texttt{{configs/resolved\_budget.json}}, \texttt{{configs/environment.json}}, \texttt{{logs/train.jsonl}}, \texttt{{logs/validation.jsonl}}, \texttt{{logs/attention.jsonl}}, \texttt{{logs/dynamics.jsonl}}, \texttt{{milestones/final/metrics.json}}, and the final and step-209,264 evaluation result files, all rooted at \texttt{{provenance/30b-run/}}. The committed training log is losslessly compressed as \texttt{{train.jsonl.gz}}. The evaluation result files are absent from this checkout. The generated files \texttt{{tables/training\_timeseries.csv}}, \texttt{{tables/final\_attention\_layers.csv}}, \texttt{{tables/final\_quantization\_layers.csv}}, and \texttt{{tables/layerwise\_update\_timeseries.csv}} are machine-readable derived tables underlying the figures, not raw evidence.
\end{{document}}
"""
    (HERE / "30B_TRAINING_RUN_TECHNICAL_REPORT.tex").write_text(tex)


def main() -> None:
    require_complete_evidence()
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    model = load_json("configs/model_config.json")
    config = load_json("configs/train_config.json")
    manifest = load_json("configs/data_manifest.json")
    budget = load_json("configs/resolved_budget.json")
    env = load_json("configs/environment.json")
    metrics = load_json("milestones/final/metrics.json")
    train = load_jsonl("logs/train.jsonl")
    val = load_jsonl("logs/validation.jsonl")
    attention = load_jsonl("logs/attention.jsonl")
    dynamics = load_jsonl("logs/dynamics.jsonl")
    final_attention, dynamic_rows, train_rows = make_figures(
        train, val, attention, dynamics, int(budget["tokens_per_step"])
    )
    final_quant = final_quantization_rows(metrics)
    quantization_figure(final_quant)
    csv_write(TABLE / "training_timeseries.csv", train_rows)
    csv_write(TABLE / "validation_timeseries.csv", [
        {"step": r["step"], "tokens_b": b(r["step"] * budget["tokens_per_step"]), "val_loss": r["val_loss"],
         "best_val_loss": r["best_val_loss"], "best_val_step": r["best_val_step"]} for r in val
    ])
    csv_write(TABLE / "final_attention_layers.csv", final_attention)
    csv_write(TABLE / "final_quantization_layers.csv", final_quant)
    csv_write(TABLE / "layerwise_update_timeseries.csv", dynamic_rows)
    tasks = ["hellaswag", "lambada", "arc_easy", "winogrande", "sciq"]
    eval_rows = []
    for task in tasks:
        prior = load_json(f"evaluations/ckpt_step_209264.pt/{task}/result.json")
        final = load_json(f"evaluations/ckpt_final.pt/{task}/result.json")
        label, prior_value = eval_metric(task, prior)
        _, final_value = eval_metric(task, final)
        eval_rows.append({"task": task, "metric": label, "prior": prior_value, "final": final_value, "delta": final_value - prior_value})
    csv_write(TABLE / "evaluation_comparison.csv", eval_rows)
    write_tex(train, val, attention, dynamics, metrics, model, config, manifest, budget, env,
              final_attention, final_quant, dynamic_rows, eval_rows)
    print(HERE / "30B_TRAINING_RUN_TECHNICAL_REPORT.tex")


if __name__ == "__main__":
    main()
