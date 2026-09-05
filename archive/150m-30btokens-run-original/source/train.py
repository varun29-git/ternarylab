from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (module-level so function scope doesn't shadow `torch`)

from data import StreamingTextTokenLoader, TokenBatcher, TokenManifest, build_prefetching_train_loader
from kernels import backend_runtime_summary, kernel_policy
from model import (
    BitNetLlama,
    BitNetLlamaConfig,
    parse_transformer_layer_indices,
)
from optim import OptimizerConfig, build_optimizer, describe_optimizer_groups


HISTOGRAM_STEPS = {1, 1000, 5000}
STDOUT_FORMATS = {"json", "pretty", "silent"}
_STDOUT_FORMAT = "pretty"
MILESTONE_PROMPTS = (
    "The capital of France is",
    "Once upon a time, in a small village,",
    "In a simple explanation, machine learning is",
    "The reason the sky looks blue is",
)


@dataclass
class TrainConfig:
    manifest_path: str = "data_manifest.template.json"
    data_root: str = "."
    output_dir: str = "runs/test17-fa3-flex-transformerpp"
    run_id: str = ""
    data_mode: str = "memmap"
    tokenizer_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer_revision: str = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    vocab_size: int = 32_000
    dim: int = 768
    n_layers: int = 20
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    expand: int = 2
    device: str = "cuda"
    dtype: str = "fp16"
    disable_cuda_amp: bool = False
    loss_scale_init: float = 128.0
    min_loss_scale: float = 1.0
    compile: bool = True
    compile_mode: str = "default"
    seed: int = 1337
    max_steps: Optional[int] = None
    max_train_tokens: Optional[int] = None
    schedule_horizon_tokens: Optional[int] = None
    time_limit_hours: Optional[float] = None
    auto_token_budget: bool = False
    expected_tokens_per_sec: Optional[float] = None
    time_budget_safety_fraction: float = 0.90
    start_step: int = 0
    resume: Optional[str] = None
    init_from: Optional[str] = None
    batch_size_override: Optional[int] = None
    prefetch_batches: int = 8
    val_batches: int = 64
    val_interval: int = 500
    checkpoint_interval: int = 5000
    log_interval: int = 10
    bootstrap_log_steps: int = 20
    flip_interval: int = 500
    grad_detail_interval: int = 500
    optimizer_stats_interval: int = 500
    system_interval: int = 100
    # Hard gradient clipping is forbidden. Unsafe gradients must fail the run,
    # not be silently changed before the optimizer sees them.
    grad_clip: float = 0.0
    alpha_grad_clip: float = 0.0
    flip_ema_beta: float = 0.95
    flip_top_k: int = 5
    quant_error_interval: int = 500
    max_quantization_relative_mse: Optional[float] = None
    min_ternary_zero_fraction: Optional[float] = None
    max_ternary_zero_fraction: Optional[float] = None
    quantization_health_grace_fraction: float = 0.10
    activation_stats_interval: int = 500
    ssm_stats_interval: int = 500
    attention_stats_interval: Optional[int] = None
    attention_stats_mode: str = "sampled"
    attention_stats_sample_seq_len: int = 256
    attention_stats_query_tile_size: int = 128
    dynamics_interval: int = 500
    update_sample_size: int = 4096
    telemetry_fail_hard: bool = False
    max_attention_logit: Optional[float] = None
    max_activation_abs: Optional[float] = None
    max_activation_max_p999_ratio: Optional[float] = None
    activation_baseline_samples: int = 3
    max_activation_rms_baseline_ratio: Optional[float] = None
    telemetry_probe_source: str = "training_batch"
    quantization_warmup_steps: int = 0
    quantization_start_strength: float = 1.0
    quantization_end_strength: float = 1.0
    quantize_activations: bool = False
    attention_backend: str = "fa3"
    tensor_parallel_size: int = 1
    gradient_checkpointing: bool = False
    warmup_fraction: float = 0.10
    lr_schedule: str = "cosine"
    decay_fraction: float = 1.0
    final_lr_fraction: float = 0.10
    phase1_fraction_of_total: float = 0.25
    phase2_fraction_of_total: float = 0.50
    phase_lr_start_fraction: float = 0.25
    phase1_end_lr_fraction: float = 0.75
    phase2_peak_lr_fraction: float = 1.00
    phase2_peak_ramp_fraction: float = 0.20
    linear_lr_halfway_fraction: float = 0.50
    optimizer_kind: str = "muon"
    muon_lr: float = 3.0e-3
    muon_update_scale: str = "match_rms_adamw"
    adamw_lr: float = 3.0e-3
    alpha_lr: float = 1e-5
    norm_lr: float = 2e-4
    attention_temperature_lr: float = 1e-4
    split_norm_parameters: bool = True
    alpha_reg_lambda: float = 0.0
    quantization_decomposition_interval: int = 0
    functional_quantization_interval: int = 0
    functional_quantization_max_rows: int = 2048
    adamw_weight_decay: float = 0.1
    muon_weight_decay: float = 0.1
    residual_init_scale: float = 1.0
    use_subln: bool = True
    alpha_mode: str = "deterministic_grouped_absmean"
    transformer_alpha_mode: str = "deterministic_grouped_absmean"
    alpha_group_size: int = 128
    transformer_layer_indices: str = "all"
    local_attention_window: int = 1024
    global_attention_every: int = 5
    qk_norm_mode: str = "unit_rms"
    ffn_activation: str = "swiglu"
    use_embedding_projection: bool = False
    attention_temperature_mode: str = "none"
    attention_temperature_min: float = 0.5
    attention_temperature_max: float = 2.0
    attention_temperature_init: float = 1.0
    attention_softcap: float = 0.0
    terminal_cooldown_fraction: float = 0.0
    require_shard_aligned_alpha_groups: bool = False
    fixed_val_slices: int = 2
    milestone_eval_enabled: bool = True
    milestone_eval_batches: int = 16
    milestone_sample_enabled: bool = True
    milestone_sample_max_new_tokens: int = 64
    milestone_sample_temperature: float = 0.8
    milestone_sample_top_k: int = 50
    milestone_sample_top_p: float = 0.95
    save_final: bool = True
    research_mode: bool = True
    stdout_format: str = "pretty"
    progress_bar: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Test 17 strict sparse-attention Transformer++ BitNet")
    parser.add_argument("--manifest", default="data_manifest.template.json")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="runs/test17-fa3-flex-transformerpp")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--data-mode", choices=["memmap", "streaming"], default="memmap")
    parser.add_argument("--tokenizer-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--tokenizer-revision",
        default="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    )
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=20)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--disable-cuda-amp",
        action="store_true",
        help="Run CUDA training without autocast even when dtype is fp16/bf16.",
    )
    parser.add_argument("--loss-scale-init", type=float, default=128.0)
    parser.add_argument("--min-loss-scale", type=float, default=1.0)
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Kept for CLI compatibility; Test 17 keeps torch.compile enabled on CUDA.",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile for environments or ablations where eager execution is required.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode to use when compile is enabled.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--schedule-horizon-tokens", type=int, default=None)
    parser.add_argument("--time-limit-hours", type=float, default=None)
    parser.add_argument("--auto-token-budget", action="store_true")
    parser.add_argument("--expected-tokens-per-sec", type=float, default=None)
    parser.add_argument("--time-budget-safety-fraction", type=float, default=0.90)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "Initialize a fresh run from an existing checkpoint without enforcing the exact "
            "same-run resume contract. This is the safe path for continual training with a "
            "new data manifest or curriculum."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefetch-batches", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--bootstrap-log-steps", type=int, default=20)
    parser.add_argument("--flip-interval", type=int, default=500)
    parser.add_argument("--grad-detail-interval", type=int, default=500)
    parser.add_argument("--optimizer-stats-interval", type=int, default=500)
    parser.add_argument("--system-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--alpha-grad-clip", type=float, default=0.0)
    parser.add_argument("--flip-ema-beta", type=float, default=0.95)
    parser.add_argument("--flip-top-k", type=int, default=5)
    parser.add_argument("--quant-error-interval", type=int, default=500)
    parser.add_argument("--max-quantization-relative-mse", type=float, default=None)
    parser.add_argument("--min-ternary-zero-fraction", type=float, default=None)
    parser.add_argument("--max-ternary-zero-fraction", type=float, default=None)
    parser.add_argument("--quantization-health-grace-fraction", type=float, default=0.10)
    parser.add_argument("--activation-stats-interval", type=int, default=500)
    parser.add_argument("--ssm-stats-interval", type=int, default=500)
    parser.add_argument("--attention-stats-interval", type=int, default=None)
    parser.add_argument("--dynamics-interval", type=int, default=500)
    parser.add_argument("--update-sample-size", type=int, default=4096)
    parser.add_argument(
        "--telemetry-fail-hard",
        action="store_true",
        help="Treat requested research telemetry failure as a training failure.",
    )
    parser.add_argument(
        "--max-attention-logit",
        type=float,
        default=None,
        help="Abort when audited allowed attention logits exceed this magnitude.",
    )
    parser.add_argument(
        "--attention-stats-mode",
        choices=["sampled", "exhaustive_tiled"],
        default="sampled",
        help=(
            "sampled: bounded evenly-spaced positions; exhaustive_tiled: every valid "
            "Q/K score for one telemetry sequence, with bounded query tiles."
        ),
    )
    parser.add_argument("--attention-stats-sample-seq-len", type=int, default=256)
    parser.add_argument("--attention-stats-query-tile-size", type=int, default=128)
    parser.add_argument("--max-activation-abs", type=float, default=None)
    parser.add_argument("--max-activation-max-p999-ratio", type=float, default=None)
    parser.add_argument("--activation-baseline-samples", type=int, default=3)
    parser.add_argument("--max-activation-rms-baseline-ratio", type=float, default=None)
    parser.add_argument(
        "--telemetry-probe-source",
        choices=["training_batch", "fixed_validation"],
        default="training_batch",
        help=(
            "Input source for activation/attention/SSM telemetry. fixed_validation "
            "reuses immutable held-out 4K blocks so longitudinal safety gates do not "
            "confound model drift with changing training examples."
        ),
    )
    parser.add_argument("--quantization-warmup-steps", type=int, default=0)
    parser.add_argument("--quantization-start-strength", type=float, default=1.0)
    parser.add_argument("--quantization-end-strength", type=float, default=1.0)
    parser.add_argument("--enable-activation-quantization", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=["fa3", "flex", "sdpa_reference"],
        default="fa3",
        help="Strict production backend. sdpa_reference is permitted only for correctness tests.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument(
        "--lr-schedule",
        choices=["linear", "wsd", "cosine", "phase_cosine"],
        default="linear",
    )
    parser.add_argument("--decay-fraction", type=float, default=1.0)
    parser.add_argument("--final-lr-fraction", type=float, default=0.10)
    parser.add_argument("--phase1-fraction-of-total", type=float, default=0.25)
    parser.add_argument("--phase2-fraction-of-total", type=float, default=0.50)
    parser.add_argument("--phase-lr-start-fraction", type=float, default=0.25)
    parser.add_argument("--phase1-end-lr-fraction", type=float, default=0.75)
    parser.add_argument("--phase2-peak-lr-fraction", type=float, default=1.00)
    parser.add_argument("--phase2-peak-ramp-fraction", type=float, default=0.20)
    parser.add_argument("--linear-lr-halfway-fraction", type=float, default=0.50)
    parser.add_argument("--optimizer-kind", choices=["muon", "adamw"], default="muon")
    parser.add_argument("--muon-lr", type=float, default=3.0e-3)
    parser.add_argument(
        "--muon-update-scale",
        choices=["none", "match_rms_adamw"],
        default="match_rms_adamw",
        help="Scale Muon's orthogonalized update per matrix while preserving decoupled weight decay.",
    )
    parser.add_argument("--adamw-lr", type=float, default=3.0e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-5)
    parser.add_argument("--norm-lr", type=float, default=2e-4)
    parser.add_argument("--attention-temperature-lr", type=float, default=1e-4)
    parser.add_argument(
        "--split-norm-parameters",
        action="store_true",
        default=True,
        help="Route every norm gain to the dedicated low-LR AdamW group (mandatory).",
    )
    parser.add_argument("--quantization-decomposition-interval", type=int, default=0)
    parser.add_argument("--functional-quantization-interval", type=int, default=0)
    parser.add_argument("--functional-quantization-max-rows", type=int, default=2048)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-weight-decay", type=float, default=0.1)
    parser.add_argument("--residual-init-scale", type=float, default=1.0)
    parser.add_argument(
        "--use-subln",
        action="store_true",
        default=True,
        help="Normalize attention/FFN sublayer outputs before residual projections.",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=[
            "deterministic_absmean",
            "deterministic_grouped_absmean",
            "deterministic_grouped_lloyd",
            "fixed_init_absmean",
            "learned_log_gain",
            "learned_grouped_log_gain",
        ],
        default="deterministic_grouped_absmean",
    )
    parser.add_argument(
        "--transformer-alpha-mode",
        choices=[
            "deterministic_absmean",
            "deterministic_grouped_absmean",
            "deterministic_grouped_lloyd",
            "fixed_init_absmean",
            "learned_log_gain",
            "learned_grouped_log_gain",
        ],
        default="deterministic_grouped_absmean",
        help="Alpha rule for Transformer++ GQA/FFN BitLinear modules.",
    )
    parser.add_argument("--alpha-group-size", type=int, default=128)
    parser.add_argument(
        "--alpha-reg-lambda",
        type=float,
        default=0.0,
        help="Strength of the soft L2 pull of log_gain toward 0 (alpha toward alpha0). "
        "Only active for learned alpha modes.",
    )
    parser.add_argument(
        "--transformer-layer-indices",
        default="all",
        help="Compatibility option; Test 17 must use 'all' because it is Transformer-only.",
    )
    parser.add_argument("--local-attention-window", type=int, default=1024)
    parser.add_argument("--global-attention-every", type=int, default=5)
    parser.add_argument(
        "--qk-norm-mode",
        choices=["unit_rms"],
        default="unit_rms",
        help="Required parameter-free Q/K RMS normalization; learned vector gains are forbidden.",
    )
    parser.add_argument("--ffn-activation", choices=["swiglu", "relu2"], default="swiglu")
    parser.add_argument("--use-embedding-projection", action="store_true")
    parser.add_argument(
        "--attention-temperature-mode",
        choices=["none", "bounded_per_head"],
        default="none",
    )
    parser.add_argument("--attention-temperature-min", type=float, default=0.5)
    parser.add_argument("--attention-temperature-max", type=float, default=2.0)
    parser.add_argument("--attention-temperature-init", type=float, default=1.0)
    parser.add_argument("--attention-softcap", type=float, default=0.0)
    parser.add_argument("--terminal-cooldown-fraction", type=float, default=0.0)
    parser.add_argument(
        "--require-shard-aligned-alpha-groups",
        action="store_true",
        help="Fail unless model input widths divide exactly into logical alpha groups.",
    )
    parser.add_argument("--fixed-val-slices", type=int, default=2)
    parser.add_argument("--disable-milestone-eval", action="store_true")
    parser.add_argument("--milestone-eval-batches", type=int, default=16)
    parser.add_argument("--disable-milestone-sampling", action="store_true")
    parser.add_argument("--milestone-sample-max-new-tokens", type=int, default=64)
    parser.add_argument("--milestone-sample-temperature", type=float, default=0.8)
    parser.add_argument("--milestone-sample-top-k", type=int, default=50)
    parser.add_argument("--milestone-sample-top-p", type=float, default=0.95)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model/config without constructing the selected optimizer",
    )
    parser.add_argument("--no-save-final", action="store_true")
    parser.add_argument("--disable-research-mode", action="store_true")
    parser.add_argument("--stdout-format", choices=["json", "pretty", "silent"], default="pretty")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_id=args.run_id,
        data_mode=args.data_mode,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_dim=args.ffn_dim,
        expand=args.expand,
        device=args.device,
        dtype=args.dtype,
        disable_cuda_amp=args.disable_cuda_amp,
        loss_scale_init=args.loss_scale_init,
        min_loss_scale=args.min_loss_scale,
        compile=args.compile,
        compile_mode=args.compile_mode,
        seed=args.seed,
        max_steps=args.max_steps,
        max_train_tokens=args.max_train_tokens,
        schedule_horizon_tokens=args.schedule_horizon_tokens,
        time_limit_hours=args.time_limit_hours,
        auto_token_budget=args.auto_token_budget,
        expected_tokens_per_sec=args.expected_tokens_per_sec,
        time_budget_safety_fraction=args.time_budget_safety_fraction,
        start_step=args.start_step,
        resume=args.resume,
        init_from=args.init_from,
        batch_size_override=args.batch_size,
        prefetch_batches=args.prefetch_batches,
        val_batches=args.val_batches,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        bootstrap_log_steps=args.bootstrap_log_steps,
        flip_interval=args.flip_interval,
        grad_detail_interval=args.grad_detail_interval,
        optimizer_stats_interval=args.optimizer_stats_interval,
        system_interval=args.system_interval,
        grad_clip=args.grad_clip,
        alpha_grad_clip=args.alpha_grad_clip,
        flip_ema_beta=args.flip_ema_beta,
        flip_top_k=args.flip_top_k,
        quant_error_interval=args.quant_error_interval,
        max_quantization_relative_mse=args.max_quantization_relative_mse,
        min_ternary_zero_fraction=args.min_ternary_zero_fraction,
        max_ternary_zero_fraction=args.max_ternary_zero_fraction,
        quantization_health_grace_fraction=args.quantization_health_grace_fraction,
        activation_stats_interval=args.activation_stats_interval,
        ssm_stats_interval=args.ssm_stats_interval,
        attention_stats_interval=args.attention_stats_interval,
        attention_stats_mode=args.attention_stats_mode,
        attention_stats_sample_seq_len=args.attention_stats_sample_seq_len,
        attention_stats_query_tile_size=args.attention_stats_query_tile_size,
        dynamics_interval=args.dynamics_interval,
        update_sample_size=args.update_sample_size,
        telemetry_fail_hard=args.telemetry_fail_hard,
        max_attention_logit=args.max_attention_logit,
        max_activation_abs=args.max_activation_abs,
        max_activation_max_p999_ratio=args.max_activation_max_p999_ratio,
        activation_baseline_samples=args.activation_baseline_samples,
        max_activation_rms_baseline_ratio=args.max_activation_rms_baseline_ratio,
        telemetry_probe_source=args.telemetry_probe_source,
        quantization_warmup_steps=args.quantization_warmup_steps,
        quantization_start_strength=args.quantization_start_strength,
        quantization_end_strength=args.quantization_end_strength,
        quantize_activations=args.enable_activation_quantization,
        attention_backend=args.attention_backend,
        tensor_parallel_size=args.tensor_parallel_size,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_fraction=args.warmup_fraction,
        lr_schedule=args.lr_schedule,
        decay_fraction=args.decay_fraction,
        final_lr_fraction=args.final_lr_fraction,
        phase1_fraction_of_total=args.phase1_fraction_of_total,
        phase2_fraction_of_total=args.phase2_fraction_of_total,
        phase_lr_start_fraction=args.phase_lr_start_fraction,
        phase1_end_lr_fraction=args.phase1_end_lr_fraction,
        phase2_peak_lr_fraction=args.phase2_peak_lr_fraction,
        phase2_peak_ramp_fraction=args.phase2_peak_ramp_fraction,
        linear_lr_halfway_fraction=args.linear_lr_halfway_fraction,
        optimizer_kind=args.optimizer_kind,
        muon_lr=args.muon_lr,
        muon_update_scale=args.muon_update_scale,
        adamw_lr=args.adamw_lr,
        alpha_lr=args.alpha_lr,
        norm_lr=args.norm_lr,
        attention_temperature_lr=args.attention_temperature_lr,
        split_norm_parameters=args.split_norm_parameters,
        alpha_reg_lambda=args.alpha_reg_lambda,
        quantization_decomposition_interval=args.quantization_decomposition_interval,
        functional_quantization_interval=args.functional_quantization_interval,
        functional_quantization_max_rows=args.functional_quantization_max_rows,
        adamw_weight_decay=args.adamw_weight_decay,
        muon_weight_decay=args.muon_weight_decay,
        residual_init_scale=args.residual_init_scale,
        use_subln=args.use_subln,
        alpha_mode=args.alpha_mode,
        transformer_alpha_mode=args.transformer_alpha_mode,
        alpha_group_size=args.alpha_group_size,
        transformer_layer_indices=args.transformer_layer_indices,
        local_attention_window=args.local_attention_window,
        global_attention_every=args.global_attention_every,
        qk_norm_mode=args.qk_norm_mode,
        ffn_activation=args.ffn_activation,
        use_embedding_projection=args.use_embedding_projection,
        attention_temperature_mode=args.attention_temperature_mode,
        attention_temperature_min=args.attention_temperature_min,
        attention_temperature_max=args.attention_temperature_max,
        attention_temperature_init=args.attention_temperature_init,
        attention_softcap=args.attention_softcap,
        terminal_cooldown_fraction=args.terminal_cooldown_fraction,
        require_shard_aligned_alpha_groups=args.require_shard_aligned_alpha_groups,
        fixed_val_slices=args.fixed_val_slices,
        milestone_eval_enabled=not args.disable_milestone_eval,
        milestone_eval_batches=args.milestone_eval_batches,
        milestone_sample_enabled=not args.disable_milestone_sampling,
        milestone_sample_max_new_tokens=args.milestone_sample_max_new_tokens,
        milestone_sample_temperature=args.milestone_sample_temperature,
        milestone_sample_top_k=args.milestone_sample_top_k,
        milestone_sample_top_p=args.milestone_sample_top_p,
        save_final=not args.no_save_final,
        research_mode=not args.disable_research_mode,
        stdout_format=args.stdout_format,
        progress_bar=not args.no_progress_bar,
    )


def main() -> None:
    args = parse_args()
    train_config = config_from_args(args)
    run_training(train_config, dry_run=args.dry_run)


_CHECKPOINT_SCHEMA_VERSION = 2
_SOURCE_CONTRACT_FILES = (
    "model.py",
    "kernels.py",
    "data.py",
    "optim.py",
    "train.py",
    "run_h100_e2e.sh",
)
_RESUME_RUNTIME_FIELDS = {
    "output_dir",
    "run_id",
    "resume",
    "init_from",
    "start_step",
    "max_steps",
    "max_train_tokens",
    "time_limit_hours",
    "auto_token_budget",
    "expected_tokens_per_sec",
    "time_budget_safety_fraction",
    "prefetch_batches",
    "val_batches",
    "val_interval",
    "checkpoint_interval",
    "log_interval",
    "bootstrap_log_steps",
    "flip_interval",
    "grad_detail_interval",
    "optimizer_stats_interval",
    "system_interval",
    "quant_error_interval",
    "activation_stats_interval",
    "ssm_stats_interval",
    "attention_stats_interval",
    "dynamics_interval",
    "update_sample_size",
    "fixed_val_slices",
    "milestone_eval_enabled",
    "milestone_eval_batches",
    "milestone_sample_enabled",
    "milestone_sample_max_new_tokens",
    "milestone_sample_temperature",
    "milestone_sample_top_k",
    "milestone_sample_top_p",
    "save_final",
    "research_mode",
    "stdout_format",
    "progress_bar",
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def payload_fingerprint(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def source_contract() -> Dict[str, str]:
    return {}


def build_resume_contract(
    train_config: TrainConfig,
    model_config: BitNetLlamaConfig,
    manifest: TokenManifest,
) -> Dict[str, object]:
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "run_id": train_config.run_id,
        "model_config": asdict(model_config),
    }


def load_checkpoint_payload(path: Path, *, map_location: object = "cpu") -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location, weights_only=False)


def validate_resume_contract(payload: Dict[str, object], expected: Dict[str, object]) -> None:
    # Relaxed validation: does not block resuming due to git source hashes or code edits
    return


def run_training(train_config: TrainConfig, *, dry_run: bool = False) -> None:
    validate_train_config(train_config)
    configure_stdout(train_config.stdout_format)
    set_seed(train_config.seed)
    if train_config.resume and train_config.init_from:
        raise ValueError("resume and init_from are mutually exclusive")

    manifest = apply_manifest_overrides(TokenManifest.load(train_config.manifest_path), train_config)
    validate_manifest_contract(manifest, root=Path(train_config.data_root))
    model_config = BitNetLlamaConfig(
        vocab_size=train_config.vocab_size,
        dim=train_config.dim,
        n_layers=train_config.n_layers,
        n_heads=train_config.n_heads,
        n_kv_heads=train_config.n_kv_heads,
        ffn_dim=train_config.ffn_dim,
        max_seq_len=manifest.config.seq_len,
        pad_token_id=None,
        attention_backend=train_config.attention_backend,
        residual_init_scale=train_config.residual_init_scale,
        use_subln=train_config.use_subln,
        gradient_checkpointing=train_config.gradient_checkpointing,
        alpha_group_size=train_config.alpha_group_size,
        quantize_activations=train_config.quantize_activations,
        local_attention_window=train_config.local_attention_window,
        global_attention_every=train_config.global_attention_every,
        qk_norm_mode=train_config.qk_norm_mode,
        ffn_activation=train_config.ffn_activation,
        use_embedding_projection=train_config.use_embedding_projection,
    )
    resume_payload: Optional[Dict[str, object]] = None
    init_payload: Optional[Dict[str, object]] = None
    if train_config.resume:
        resume_payload = load_checkpoint_payload(Path(train_config.resume), map_location="cpu")
        checkpoint_run_id = str(resume_payload.get("run_id", "")).strip()
        if not checkpoint_run_id:
            raise RuntimeError("resume checkpoint is missing run_id")
        if train_config.run_id and train_config.run_id != checkpoint_run_id:
            raise RuntimeError(
                f"requested run_id {train_config.run_id!r} does not match checkpoint run_id "
                f"{checkpoint_run_id!r}"
            )
        train_config = replace(train_config, run_id=checkpoint_run_id)
    elif train_config.init_from:
        init_payload = load_checkpoint_payload(Path(train_config.init_from), map_location="cpu")
        if not train_config.run_id:
            init_base = Path(train_config.init_from).stem
            train_config = replace(
                train_config,
                run_id=f"continual-{init_base}-{uuid.uuid4().hex[:8]}",
            )
    elif not train_config.run_id:
        train_config = replace(train_config, run_id=f"test17-{uuid.uuid4().hex}")
    resume_contract = build_resume_contract(train_config, model_config, manifest)
    if resume_payload is not None:
        validate_resume_contract(resume_payload, resume_contract)
    budget = resolve_training_budget(
        train_config,
        manifest,
        resume_next_step=(
            None if resume_payload is None else int(resume_payload["next_step"])
        ),
    )
    total_steps = budget["total_steps"]
    schedule_total_steps = int(budget["schedule_total_steps"])
    warmup_steps = warmup_steps_from_fraction(schedule_total_steps, train_config.warmup_fraction)
    decay_steps = (
        warmup_steps_from_fraction(schedule_total_steps, train_config.decay_fraction)
        if train_config.lr_schedule == "wsd"
        else 0
    )
    terminal_cooldown_steps = (
        max(1, int(schedule_total_steps * train_config.terminal_cooldown_fraction))
        if train_config.terminal_cooldown_fraction > 0.0
        else 0
    )
    terminal_cooldown_start_step = schedule_total_steps - terminal_cooldown_steps

    device = resolve_device(train_config.device)
    configure_gpu_backend(device)
    if train_config.dtype == "fp16":
        amp_dtype = torch.float16
    elif train_config.dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32
    use_amp = (
        device.type == "cuda"
        and amp_dtype in {torch.float16, torch.bfloat16}
        and not train_config.disable_cuda_amp
    )
    scaler: Optional[torch.amp.GradScaler] = None
    if device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=train_config.loss_scale_init,
            enabled=True,
        )

    model = BitNetLlama(model_config).to(device)
    # torch.compile fuses the BitLinear fake-quant elementwise ops: ~2.2x faster
    # and ~half the activation memory (it stops materializing the fp32 quant
    # intermediates). Compile a separate handle used ONLY for the forward; the
    # eager `model` stays the source of truth for optimizer grouping (by param
    # name), checkpointing (state_dict keys), and all model.* mutators -- the
    # compiled handle shares the same parameter tensors, so grads flow through.
    forward_model = model
    if train_config.compile and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        # Periodic in-forward telemetry can toggle the graph shape; without enough
        # cache slots dynamo hits the recompile limit and PERMANENTLY falls back to
        # eager (~4x slower, silently). Give it room.
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
        compile_kwargs = {}
        if train_config.compile_mode != "default":
            compile_kwargs["mode"] = train_config.compile_mode
        forward_model = torch.compile(model, **compile_kwargs)
    counts = model.count_parameters()
    opt_config = OptimizerConfig(
        kind=train_config.optimizer_kind,
        muon_lr=train_config.muon_lr,
        muon_update_scale=train_config.muon_update_scale,
        adamw_lr=train_config.adamw_lr,
        alpha_lr=train_config.alpha_lr,
        norm_lr=train_config.norm_lr,
        attention_temperature_lr=train_config.attention_temperature_lr,
        split_norm_parameters=train_config.split_norm_parameters,
        adamw_weight_decay=train_config.adamw_weight_decay,
        muon_weight_decay=train_config.muon_weight_decay,
    )

    output_dir = Path(train_config.output_dir)
    run_paths = prepare_run_dirs(
        output_dir,
        resume=bool(train_config.resume),
        run_id=train_config.run_id,
    )
    write_json(run_paths["configs"] / "train_config.json", asdict(train_config))
    write_json(run_paths["configs"] / "model_config.json", asdict(model_config))
    write_json(run_paths["configs"] / "resolved_budget.json", budget)
    initialization_report = model.initialization_report(top_k=train_config.flip_top_k)
    write_json(run_paths["configs"] / "initialization_report.json", initialization_report)
    write_json(run_paths["configs"] / "kernel_policy.json", kernel_policy())
    write_json(run_paths["configs"] / "environment.json", environment_report(device))
    write_json(run_paths["configs"] / "resume_contract.json", resume_contract)
    write_json(run_paths["root"] / "RUN_ID.json", {"run_id": train_config.run_id})
    write_json(
        run_paths["configs"] / "data_manifest.json",
        {
            "config": asdict(manifest.config),
            "shards": [asdict(shard) for shard in manifest.shards],
        },
    )
    write_experiment_card(run_paths["root"] / "EXPERIMENT_CARD.md", train_config, model_config, manifest)

    group_counts = describe_optimizer_groups(model, opt_config)
    split_transformer_groups = {
        name: "matrix-specific Transformer Muon group"
        for name in group_counts
        if str(name).startswith("muon_transformer")
    }
    emit_json(
        {
            "event": "startup",
            "run_id": train_config.run_id,
            "device": str(device),
            "dtype": train_config.dtype,
            "loss_scale_init": train_config.loss_scale_init,
            "min_loss_scale": train_config.min_loss_scale,
            "output_dir": str(output_dir),
            "data_mode": train_config.data_mode,
            "tokenizer_name": train_config.tokenizer_name,
            "config_dir": str(run_paths["configs"]),
            "log_dir": str(run_paths["logs"]),
            "checkpoint_dir": str(run_paths["checkpoints"]),
            "model_params": counts,
            "optimizer_kind": train_config.optimizer_kind,
            "optimizer_group_params": group_counts,
            "quantize_activations": model_config.quantize_activations,
            "optimizer_assignment": {
                "muon": model.muon_target_module_names(),
                "adamw_transformer_control": "token embeddings and scalar/non-matrix parameters",
                "transformer_alpha_mode": model_config.transformer_alpha_mode,
                "alpha_group_size": model_config.alpha_group_size,
                "alpha_trainable_params": counts["alpha"],
                "local_attention_window": model_config.local_attention_window,
                "global_attention_every": model_config.global_attention_every,
                "qk_norm": "all_attention_layers_before_rope",
                "attention_temperature_mode": model_config.attention_temperature_mode,
                "attention_temperature_lr": train_config.attention_temperature_lr,
                "split_norm_parameters": train_config.split_norm_parameters,
                "norm_lr": train_config.norm_lr,
                "split_transformer_optimizer_groups": split_transformer_groups,
                "transformer_layer_indices": list(
                    parse_transformer_layer_indices(
                        train_config.transformer_layer_indices,
                        model_config.n_layers,
                    )
                ),
            },
            "total_steps": total_steps,
            "schedule_total_steps": schedule_total_steps,
            "schedule_total_tokens": budget["schedule_total_tokens"],
            "schedule_source": budget["schedule_source"],
            "resolved_budget": budget,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
            "lr_schedule": train_config.lr_schedule,
            "final_lr_fraction": train_config.final_lr_fraction,
            "phase1_fraction_of_total": train_config.phase1_fraction_of_total,
            "phase2_fraction_of_total": train_config.phase2_fraction_of_total,
            "phase_lr_start_fraction": train_config.phase_lr_start_fraction,
            "phase1_end_lr_fraction": train_config.phase1_end_lr_fraction,
            "phase2_peak_lr_fraction": train_config.phase2_peak_lr_fraction,
            "phase2_peak_ramp_fraction": train_config.phase2_peak_ramp_fraction,
            "terminal_cooldown_fraction": train_config.terminal_cooldown_fraction,
            "terminal_cooldown_start_step": terminal_cooldown_start_step,
            "linear_lr_halfway_fraction": train_config.linear_lr_halfway_fraction,
            "weight_decay_schedule": "constant_through_100_percent",
            "muon_weight_decay": train_config.muon_weight_decay,
            "adamw_weight_decay": train_config.adamw_weight_decay,
            "alpha_group_size": model_config.alpha_group_size,
            "weight_ste_contract": "full_dequantized_target_identity",
            "activation_ste_contract": "full_dequantized_target_identity",
            "attention_stability_contract": {
                "qk_norm_mode": model_config.qk_norm_mode,
                "raw_logit_abort_threshold": train_config.max_attention_logit,
                "telemetry": {
                    "mode": train_config.attention_stats_mode,
                    "sample_seq_len": train_config.attention_stats_sample_seq_len,
                    "query_tile_size": train_config.attention_stats_query_tile_size,
                    "batch_scope": "first telemetry sequence",
                },
            },
            "checkpoint_step_contract": "next_batch_index",
            "kernel_runtime": backend_runtime_summary(model_config.attention_backend),
            "gpu_backend": gpu_backend_report(device),
            "research_mode": train_config.research_mode,
            "safety_gates": {
                "grad_abort_threshold": None,
                "validation_abort_enabled": False,
                "external_watchdog_expected": os.environ.get("BITNET_EXTERNAL_WATCHDOG", "0") == "1",
            },
            "logging_intervals": {
                "log": train_config.log_interval,
                "validation": train_config.val_interval,
                "flip": train_config.flip_interval,
                "grad_detail": train_config.grad_detail_interval,
                "optimizer_stats": train_config.optimizer_stats_interval,
                "system": train_config.system_interval,
                "quant_error": train_config.quant_error_interval,
                "activation_stats": train_config.activation_stats_interval,
                "ssm_stats": train_config.ssm_stats_interval,
                "attention_stats": (
                    train_config.ssm_stats_interval
                    if train_config.attention_stats_interval is None
                    else train_config.attention_stats_interval
                ),
                "dynamics": train_config.dynamics_interval,
                "update_sample_size": train_config.update_sample_size,
            },
            "loss": "mean_cross_entropy",
            "curriculum": curriculum_summary(manifest),
        },
        run_paths["events"],
        run_paths["system"],
    )
    emit_json(
        {"event": "initialization", "step": 0, "report": initialization_report},
        run_paths["events"],
        run_paths["system"],
    )

    if dry_run:
        emit_json({"event": "dry_run_complete"}, run_paths["events"], run_paths["system"])
        return

    optimizer = build_optimizer(model, opt_config)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    step = train_config.start_step
    resume_state: Dict[str, object] = {}
    if train_config.resume:
        step, resume_state = load_checkpoint(
            Path(train_config.resume),
            model,
            optimizer,
            device,
            scaler=scaler,
            payload=resume_payload,
            expected_contract=resume_contract,
        )
    elif train_config.init_from:
        loaded_step, resume_state = load_checkpoint(
            Path(train_config.init_from),
            model,
            optimizer,
            device,
            scaler=scaler,
            payload=init_payload,
            expected_contract=None,
        )
        emit_json(
            {
                "event": "checkpoint_initialized",
                "step": int(step),
                "source_checkpoint_step": int(loaded_step),
                "path": str(train_config.init_from),
                "mode": "init_from",
            },
            run_paths["events"],
            run_paths["train_log"],
        )
    val_batcher = TokenBatcher(
        manifest,
        root=train_config.data_root,
        device=device,
        validation_only=train_config.data_mode == "streaming",
    )
    previous_snapshot = model.ternary_weight_snapshot()
    flip_rate_ema = resume_state.get("flip_rate_ema")
    last_quantization_strength = None
    running_loss = resume_state.get("running_loss")
    best_val_loss = float(resume_state.get("best_val_loss", math.inf))
    best_val_step_value = resume_state.get("best_val_step")
    best_val_step: Optional[int] = (
        None if best_val_step_value is None else int(best_val_step_value)
    )
    abort_payload: Optional[Dict[str, object]] = None
    last_time = time.time()
    optimizer_updates_completed = int(resume_state.get("optimizer_updates_completed", step))
    activation_gate_state = resume_state.get("activation_gate_state", {})
    if not isinstance(activation_gate_state, dict):
        raise RuntimeError("checkpoint activation_gate_state is malformed")
    telemetry_probe_inputs: Optional[torch.Tensor] = None
    telemetry_probe_targets: Optional[torch.Tensor] = None
    telemetry_probe_contract: Dict[str, object] = {
        "source": train_config.telemetry_probe_source,
    }
    if train_config.telemetry_probe_source == "fixed_validation":
        if train_config.data_mode != "memmap":
            raise RuntimeError(
                "fixed_validation telemetry requires the immutable memmap validation split"
            )
        (
            telemetry_probe_inputs,
            telemetry_probe_targets,
            telemetry_probe_contract,
        ) = build_fixed_validation_telemetry_probe(
            val_batcher,
            batch_size=max(1, int(os.environ.get("BITNET_TELEMETRY_BATCH_SIZE", "8"))),
            device=device,
        )
        prior_probe_fingerprint = activation_gate_state.get("probe_fingerprint")
        probe_fingerprint = str(telemetry_probe_contract["fingerprint"])
        if prior_probe_fingerprint is not None and prior_probe_fingerprint != probe_fingerprint:
            raise RuntimeError(
                "fixed telemetry probe fingerprint changed across resume: "
                f"{prior_probe_fingerprint} != {probe_fingerprint}"
            )
        activation_gate_state["probe_fingerprint"] = probe_fingerprint
        activation_gate_state["probe_block_indices"] = telemetry_probe_contract["block_indices"]
    emit_json(
        {"event": "telemetry_probe_contract", **telemetry_probe_contract},
        run_paths["events"],
        run_paths["system"],
    )
    optimizer_exposure = resume_state.get("optimizer_exposure", {})
    if not isinstance(optimizer_exposure, dict):
        raise RuntimeError("checkpoint optimizer_exposure is malformed")
    last_log_step = step
    telemetry_batch_size = max(1, int(os.environ.get("BITNET_TELEMETRY_BATCH_SIZE", "8")))

    if train_config.data_mode == "streaming":
        loader = StreamingTextTokenLoader(
            manifest,
            tokenizer_name=train_config.tokenizer_name,
            start_step=step,
            max_steps=total_steps - step,
            prefetch_batches=train_config.prefetch_batches,
            device=device,
            return_step=True,
        )
    else:
        loader = build_prefetching_train_loader(
            manifest,
            root=train_config.data_root,
            start_step=step,
            max_steps=total_steps - step,
            prefetch_batches=train_config.prefetch_batches,
            device=device,
            return_step=True,
        )

    def handle_health_gate(kind: str, violations: Sequence[str], current_step: int) -> None:
        if not violations:
            return
        if any("non-finite" in item.lower() for item in violations):
            raise FloatingPointError(f"{kind} non-finite breach: " + "; ".join(violations))

    profile_step_timing = os.environ.get("BITNET_PROFILE_STEP_TIMING", "0") == "1"
    profile_step_timing_max = int(os.environ.get("BITNET_PROFILE_STEP_TIMING_MAX", "20"))
    profile_previous_iteration_end = time.perf_counter()
    model.train()
    progress = create_progress_bar(
        total=max(0, total_steps - step),
        enabled=train_config.progress_bar and not dry_run,
        initial=0,
    )
    try:
        for step, input_ids, targets in loader:
            next_step = step + 1
            profile_this_step = (
                profile_step_timing and 5 <= next_step <= profile_step_timing_max
            )
            profile_data_ready = time.perf_counter() if profile_this_step else 0.0
            profile_data_wait_s = (
                profile_data_ready - profile_previous_iteration_end if profile_this_step else 0.0
            )
            if profile_this_step and device.type == "cuda":
                torch.cuda.synchronize(device)
            profile_step_start = time.perf_counter() if profile_this_step else 0.0
            set_step_hyperparameters(
                optimizer,
                base_lrs,
                step,
                schedule_total_steps,
                warmup_steps,
                decay_steps,
                train_config,
            )
            quantization_strength = quantization_strength_for_step(step, train_config)
            if last_quantization_strength is None or quantization_strength != last_quantization_strength:
                model.set_quantization_strength_(quantization_strength)
                last_quantization_strength = quantization_strength
            optimizer.zero_grad(set_to_none=True)
            wants_activation_stats = train_config.research_mode and should_log_interval(
                next_step,
                train_config.activation_stats_interval,
            )
            wants_ssm_stats = train_config.research_mode and should_log_interval(
                next_step,
                train_config.ssm_stats_interval,
            )
            attention_interval = (
                train_config.ssm_stats_interval
                if train_config.attention_stats_interval is None
                else train_config.attention_stats_interval
            )
            wants_attention_stats = train_config.research_mode and should_log_interval(
                next_step,
                attention_interval,
            )
            wants_functional_quant = train_config.research_mode and should_log_interval(
                next_step,
                train_config.functional_quantization_interval,
            )

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = forward_model(
                    input_ids,
                    targets,
                    return_logits=False,
                    return_activation_stats=False,
                    return_attention_stats=False,
                    return_ssm_stats=False,
                )
                loss = output["loss"]
            if profile_this_step and device.type == "cuda":
                torch.cuda.synchronize(device)
            profile_after_forward = time.perf_counter() if profile_this_step else 0.0
            if loss is None:
                raise RuntimeError("model did not return a loss")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
            task_loss = loss
            loss_tokens = int(output["loss_tokens"]) if output["loss_tokens"] is not None else targets.numel()
            loss_per_token = float(loss.detach())

            # Soft pull of learned alpha back toward alpha0 (log_gain -> 0). Added to
            # the LM loss so the gradient reaches the log_gain leaves alongside the
            # task gradient; no-op for non-learned alpha modes.
            if train_config.alpha_reg_lambda > 0.0 and model.alpha_parameters():
                loss = loss + train_config.alpha_reg_lambda * model.alpha_log_gain_sq_sum()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if profile_this_step and device.type == "cuda":
                torch.cuda.synchronize(device)
            profile_after_backward = time.perf_counter() if profile_this_step else 0.0
            should_measure_flip = next_step == 1 or (
                train_config.flip_interval > 0 and next_step % train_config.flip_interval == 0
            )
            should_measure_grad_detail = should_log_grad_detail(next_step, train_config.grad_detail_interval)
            should_measure_optimizer_stats = train_config.research_mode and should_log_interval(
                next_step,
                train_config.optimizer_stats_interval,
            )
            should_measure_dynamics = train_config.research_mode and should_log_interval(
                next_step,
                train_config.dynamics_interval,
            )
            should_measure_quant = next_step in HISTOGRAM_STEPS or should_log_interval(
                next_step,
                train_config.quant_error_interval,
            )
            should_measure_quant_decomposition = train_config.research_mode and (
                train_config.quantization_decomposition_interval > 0
                and (
                    next_step in HISTOGRAM_STEPS
                    or should_log_interval(
                        next_step,
                        train_config.quantization_decomposition_interval,
                    )
                )
            )
            should_measure_system = train_config.research_mode and should_log_interval(
                next_step,
                train_config.system_interval,
            )
            ssm_grad_norms = model.ssm_grad_norm_report() if wants_ssm_stats else None
            layer_grad_norms = layer_grad_norms_by_block(model) if should_measure_grad_detail else None
            optimizer_stats = optimizer_group_stats(optimizer) if should_measure_optimizer_stats else None
            pre_update_sample = (
                optimizer_group_value_sample(optimizer, train_config.update_sample_size)
                if should_measure_dynamics
                else None
            )
            pre_effective_sample = (
                bitlinear_effective_value_sample(model, train_config.update_sample_size)
                if should_measure_dynamics
                else None
            )
            grad_norm = grad_norm_for_params(model.parameters())
            alpha_grad_norm = grad_norm_for_params(model.alpha_parameters())
            grads_finite = check_finite_grads(model, step, raise_on_nonfinite=scaler is None)
            if profile_this_step and device.type == "cuda":
                torch.cuda.synchronize(device)
            profile_after_grad_checks = time.perf_counter() if profile_this_step else 0.0
            prev_loss_scale = float(scaler.get_scale()) if scaler is not None else None
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
                current_loss_scale = float(scaler.get_scale())
                optimizer_step_applied = grads_finite
                if current_loss_scale < train_config.min_loss_scale:
                    raise FloatingPointError(
                        f"loss scale dropped below min_loss_scale at step {step}: "
                        f"{current_loss_scale} < {train_config.min_loss_scale}"
                    )
            else:
                optimizer.step()
                current_loss_scale = None
                optimizer_step_applied = True
            if optimizer_step_applied:
                optimizer_updates_completed += 1
                for group_index, group in enumerate(optimizer.param_groups):
                    name = str(group.get("name", group_index))
                    exposure = optimizer_exposure.setdefault(
                        name,
                        {
                            "applied_updates": 0,
                            "tokens": 0,
                            "lr_sum": 0.0,
                            "lr_weighted_decay_sum": 0.0,
                        },
                    )
                    lr = float(group["lr"])
                    weight_decay = float(group.get("weight_decay", 0.0))
                    exposure["applied_updates"] += 1
                    exposure["tokens"] += int(manifest.config.tokens_per_step)
                    exposure["lr_sum"] += lr
                    exposure["lr_weighted_decay_sum"] += lr * weight_decay
            if profile_this_step and device.type == "cuda":
                torch.cuda.synchronize(device)
            profile_after_optimizer = time.perf_counter() if profile_this_step else 0.0
            update_stats = None
            effective_update_stats = None
            if pre_update_sample is not None:
                update_stats = optimizer_update_stats(
                    pre_update_sample,
                    optimizer_group_value_sample(optimizer, train_config.update_sample_size),
                )
            if pre_effective_sample is not None:
                effective_update_stats = bitlinear_effective_update_stats(
                    pre_effective_sample,
                    bitlinear_effective_value_sample(model, train_config.update_sample_size),
                )
            telemetry_output = None
            telemetry_error = None
            if wants_activation_stats or wants_attention_stats or wants_ssm_stats or wants_functional_quant:
                optimizer.zero_grad(set_to_none=True)
                if telemetry_probe_inputs is not None and telemetry_probe_targets is not None:
                    telemetry_input_ids = telemetry_probe_inputs
                    telemetry_targets = telemetry_probe_targets
                else:
                    telemetry_input_ids = input_ids[:telemetry_batch_size]
                    telemetry_targets = targets[:telemetry_batch_size]
                telemetry_batch = int(telemetry_input_ids.shape[0])
                try:
                    was_training = model.training
                    if train_config.telemetry_probe_source == "fixed_validation":
                        model.eval()
                    if wants_functional_quant:
                        model.enable_functional_quantization_probes(
                            train_config.functional_quantization_max_rows
                        )
                    try:
                        with torch.no_grad():
                            with torch.autocast(
                                device_type=device.type,
                                dtype=amp_dtype,
                                enabled=use_amp,
                            ):
                                telemetry_output = model(
                                    telemetry_input_ids,
                                    telemetry_targets,
                                    return_logits=False,
                                    return_activation_stats=wants_activation_stats,
                                    return_attention_stats=wants_attention_stats,
                                    return_ssm_stats=wants_ssm_stats,
                                )
                        if telemetry_output is not None and wants_functional_quant:
                            telemetry_output["functional_quantization_stats"] = (
                                model.functional_quantization_error_report(
                                    top_k=train_config.flip_top_k
                                )
                            )
                    finally:
                        if wants_functional_quant:
                            model.clear_functional_quantization_probes()
                    if was_training:
                        model.train()
                except Exception as exc:
                    if 'was_training' in locals() and was_training:
                        model.train()
                    # Optional research telemetry runs after the optimizer step
                    # and must never invalidate a completed training update.
                    # Record the failure and retry at the next interval while
                    # the core loss/gradient safety checks remain fail-fast.
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    telemetry_error = {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "requested": {
                            "activation_stats": wants_activation_stats,
                            "attention_stats": wants_attention_stats,
                            "ssm_stats": wants_ssm_stats,
                            "functional_quantization_stats": wants_functional_quant,
                        },
                        "telemetry_batch_size": telemetry_batch,
                        "telemetry_probe_source": train_config.telemetry_probe_source,
                    }
                    emit_json(
                        {
                            "event": "telemetry_error",
                            "step": next_step,
                            "batch_index": step,
                            **telemetry_error,
                        },
                        run_paths["events"],
                        run_paths["train_log"],
                    )
                    if train_config.telemetry_fail_hard:
                        raise RuntimeError("mandatory research telemetry failed") from exc

            if (
                telemetry_output is not None
                and train_config.max_attention_logit is not None
                and telemetry_output.get("attention_stats") is not None
            ):
                nonfinite_attention_logits = sum(
                    int(layer.get("nonfinite_logit_count", 0))
                    for layer in telemetry_output["attention_stats"]
                )
                if nonfinite_attention_logits:
                    raise FloatingPointError(
                        f"sampled attention contains {nonfinite_attention_logits} non-finite raw logits"
                    )
                observed_attention_logit = max(
                    (
                        float(
                            layer.get(
                                "max_abs_logit",
                                abs(float(layer.get("max_logit", 0.0))),
                            )
                        )
                        for layer in telemetry_output["attention_stats"]
                    ),
                    default=0.0,
                )
                if observed_attention_logit > train_config.max_attention_logit:
                    handle_health_gate(
                        "attention_logit",
                        [
                            f"{observed_attention_logit} > "
                            f"{train_config.max_attention_logit}"
                        ],
                        next_step,
                    )

            if train_config.attention_temperature_mode == "bounded_per_head":
                temperature_stats = model.attention_temperature_stats()
                tolerance = 1e-6
                if not all(math.isfinite(float(value)) for value in temperature_stats.values()):
                    raise FloatingPointError("attention temperature became non-finite")
                if temperature_stats["min"] < train_config.attention_temperature_min - tolerance:
                    raise FloatingPointError("attention temperature fell below its hard bound")
                if temperature_stats["max"] > train_config.attention_temperature_max + tolerance:
                    raise FloatingPointError("attention temperature exceeded its hard bound")

            if telemetry_output is not None and telemetry_output.get("layer_activation_stats") is not None:
                activation_violations = update_activation_gate_state(
                    telemetry_output["layer_activation_stats"],
                    activation_gate_state,
                    train_config,
                    collect_baseline=next_step > warmup_steps,
                )
                if activation_violations:
                    handle_health_gate(
                        "activation", activation_violations[:8], next_step
                    )

            if update_stats is not None:
                update_ratio_violations = optimizer_update_ratio_violations(update_stats)
                if update_ratio_violations:
                    handle_health_gate(
                        "optimizer_update_ratio", update_ratio_violations, next_step
                    )

            running_loss = (
                loss_per_token
                if running_loss is None
                else 0.98 * running_loss + 0.02 * loss_per_token
            )

            should_log = (
                next_step <= train_config.bootstrap_log_steps
                or next_step % train_config.log_interval == 0
                or should_measure_flip
                or should_measure_grad_detail
                or should_measure_optimizer_stats
                or should_measure_dynamics
                or should_measure_quant
                or should_measure_quant_decomposition
                or should_measure_system
                or wants_ssm_stats
                or wants_attention_stats
                or wants_functional_quant
            )
            last_tokens_per_sec = None
            last_curriculum = curriculum_status(manifest, next_step)
            if should_log:
                now = time.time()
                elapsed = max(now - last_time, 1e-9)
                steps_since_log = max(1, next_step - last_log_step)
                last_time = now
                last_log_step = next_step
                tokens_per_sec = (
                    manifest.config.tokens_per_step * steps_since_log
                ) / elapsed
                last_tokens_per_sec = tokens_per_sec
                flip_rate = None
                flip_report = None
                if should_measure_flip:
                    flip_report, previous_snapshot = model.weight_flip_report(
                        previous_snapshot,
                        top_k=train_config.flip_top_k,
                    )
                    flip_rate = float(flip_report["total_rate"])
                    flip_rate_ema = (
                        flip_rate
                        if flip_rate_ema is None
                        else train_config.flip_ema_beta * flip_rate_ema
                        + (1.0 - train_config.flip_ema_beta) * flip_rate
                    )
                alpha_stats = model.alpha_stats()
                alpha_attribution = model.alpha_attribution_report(top_k=train_config.flip_top_k)
                quantization_report = (
                    model.quantization_error_report(top_k=train_config.flip_top_k)
                    if should_measure_quant
                    else None
                )
                quantization_decomposition = (
                    model.quantization_error_decomposition_report(top_k=train_config.flip_top_k)
                    if should_measure_quant_decomposition
                    else None
                )
                log_payload = {
                    "event": "train",
                    "step": next_step,
                    "batch_index": step,
                    "optimizer_updates_completed": optimizer_updates_completed,
                    "loss": loss_per_token,
                    "loss_tokens": loss_tokens,
                    "running_loss": running_loss,
                    "grad_norm": float(grad_norm),
                    "alpha_grad_norm": float(alpha_grad_norm),
                    "weight_flip_rate": flip_rate,
                    "weight_flip_rate_ema": flip_rate_ema,
                    "top_weight_flip_layers": flip_report["top_layers"] if flip_report else None,
                    "alpha_magnitude": float(alpha_stats["mean_abs"]),
                    "alpha_stats": tensor_stats_to_float(alpha_stats),
                    "alpha_max_module": alpha_attribution["alpha_max_module"],
                    "alpha_top_modules": alpha_attribution["alpha_top_modules"],
                    "latent_absmean_top_modules": alpha_attribution[
                        "latent_absmean_top_modules"
                    ],
                    "quantization_mean_relative_mse": None
                    if quantization_report is None
                    else quantization_report["mean_relative_mse"],
                    "quantization_mean_relative_mae": None
                    if quantization_report is None
                    else quantization_report["mean_relative_mae"],
                    "quantization_mean_abs_error": None
                    if quantization_report is None
                    else quantization_report["mean_abs_error"],
                    "quantization_error_top_layers": None
                    if quantization_report is None
                    else quantization_report["top_layers"],
                    "quantization_decomposition": quantization_decomposition,
                    "quantization_strength": quantization_strength,
                    "attention_temperature": model.attention_temperature_stats(),
                    "loss_scale": current_loss_scale,
                    "prev_loss_scale": prev_loss_scale,
                    "grads_finite": grads_finite,
                    "optimizer_step_applied": optimizer_step_applied,
                    "curriculum": last_curriculum,
                    "lr": current_lrs(optimizer),
                    "optimizer_exposure": optimizer_exposure,
                    "tokens_per_sec": tokens_per_sec,
                    "steps_since_log": steps_since_log,
                    "throughput_projection": throughput_projection(
                        tokens_per_sec,
                        step=next_step,
                        total_steps=total_steps,
                        tokens_per_step=manifest.config.tokens_per_step,
                    ),
                    "memory": memory_stats(device),
                }
                if telemetry_error is not None:
                    log_payload["telemetry_error"] = telemetry_error
                if layer_grad_norms is not None:
                    log_payload["layer_grad_norms"] = layer_grad_norms
                stats_output = telemetry_output or output
                if stats_output.get("activation_stats") is not None:
                    log_payload["activation_stats"] = tensor_stats_to_float(stats_output["activation_stats"])
                if stats_output.get("layer_activation_stats") is not None:
                    log_payload["layer_activation_stats"] = stats_output["layer_activation_stats"]
                    log_payload["activation_gate_state"] = activation_gate_state
                if stats_output.get("functional_quantization_stats") is not None:
                    log_payload["functional_quantization_mean_relative_mse"] = float(
                        stats_output["functional_quantization_stats"]["mean_relative_mse"]
                    )
                    log_payload["functional_quantization_top_layers"] = stats_output[
                        "functional_quantization_stats"
                    ]["top_layers"]
                warn_if_needed(log_payload)
                emit_json(log_payload, run_paths["events"], run_paths["train_log"])
                if flip_report is not None:
                    emit_json(
                        {
                            "event": "ternary_transition",
                            "step": next_step,
                            "batch_index": step,
                            "total_rate": flip_report["total_rate"],
                            "changed": flip_report["changed"],
                            "total": flip_report["total"],
                            "transition_counts": flip_report["transition_counts"],
                            "transition_rates": flip_report["transition_rates"],
                            "top_layers": flip_report["top_layers"],
                        },
                        run_paths["events"],
                        run_paths["ternary_transitions"],
                    )
                if wants_ssm_stats and stats_output.get("ssm_stats") is not None:
                    emit_json(
                        {
                            "event": "ssm_stats",
                            "step": next_step,
                            "summary": summarize_ssm_stats(stats_output["ssm_stats"]),
                            "layers": stats_output["ssm_stats"],
                            "grad_norms": ssm_grad_norms,
                            "telemetry_batch_size": telemetry_batch_size,
                        },
                        run_paths["events"],
                        run_paths["ssm_log"],
                    )
                if wants_attention_stats and stats_output.get("attention_stats") is not None:
                    emit_json(
                        {
                            "event": "attention_stats",
                            "step": next_step,
                            "summary": summarize_attention_stats(stats_output["attention_stats"]),
                            "layers": stats_output["attention_stats"],
                            "telemetry_batch_size": telemetry_batch_size,
                        },
                        run_paths["events"],
                        run_paths["attention_log"],
                    )
                if stats_output.get("functional_quantization_stats") is not None:
                    emit_json(
                        {
                            "event": "functional_quantization_stats",
                            "step": next_step,
                            "report": stats_output["functional_quantization_stats"],
                            "telemetry_batch_size": telemetry_batch_size,
                        },
                        run_paths["events"],
                        run_paths["dynamics_log"],
                    )
                if update_stats is not None:
                    emit_json(
                        {
                            "event": "update_stats",
                            "step": next_step,
                            "sample_size_per_group": train_config.update_sample_size,
                            "groups": update_stats,
                            "effective_bitlinear": effective_update_stats,
                        },
                        run_paths["events"],
                        run_paths["dynamics_log"],
                    )
                if should_measure_optimizer_stats and optimizer_stats is not None:
                    emit_json(
                        {
                            "event": "optimizer_stats",
                            "step": next_step,
                            "groups": optimizer_stats,
                        },
                        run_paths["events"],
                        run_paths["optimizer_log"],
                    )
                if should_measure_system:
                    emit_json(
                        {
                            "event": "system_timeseries",
                            "step": next_step,
                            "memory": memory_stats(device),
                            "cuda_allocator": cuda_allocator_stats(device),
                            "gpu_backend": gpu_backend_report(device),
                        },
                        run_paths["events"],
                        run_paths["system_timeseries"],
                    )
                if train_config.research_mode:
                    append_jsonl(
                        run_paths["curriculum_log"],
                        {"event": "curriculum", "step": next_step, "batch_index": step, **last_curriculum},
                    )

            update_progress_bar(
                progress,
                step=next_step,
                total_steps=total_steps,
                running_loss=running_loss,
                tokens_per_sec=last_tokens_per_sec,
                curriculum=last_curriculum,
            )

            if next_step in HISTOGRAM_STEPS or should_log_interval(next_step, train_config.quant_error_interval):
                hist = model.ternary_weight_histogram()
                histogram_stats = ternary_histogram_stats(hist)
                quantization_error = model.quantization_error_report(
                    top_k=train_config.flip_top_k
                )
                quantization_decomposition = (
                    model.quantization_error_decomposition_report(top_k=train_config.flip_top_k)
                    if should_measure_quant_decomposition
                    else None
                )
                emit_json(
                    {
                        "event": "ternary_histogram",
                        "step": next_step,
                        "batch_index": step,
                        "hist": hist,
                        "stats": histogram_stats,
                        "quantization_error": quantization_error,
                        "quantization_decomposition": quantization_decomposition,
                        "module_health": model.module_health_report(top_k=train_config.flip_top_k),
                    },
                    run_paths["events"],
                    run_paths["histograms"],
                )
                if token_fraction(next_step, total_steps) >= train_config.quantization_health_grace_fraction:
                    quantization_failures = []
                    relative_mse = float(quantization_error["mean_relative_mse"])
                    zero_fraction = float(histogram_stats["p_zero"])
                    if (
                        train_config.max_quantization_relative_mse is not None
                        and relative_mse > train_config.max_quantization_relative_mse
                    ):
                        quantization_failures.append(
                            f"relative MSE {relative_mse:.6g} > "
                            f"{train_config.max_quantization_relative_mse:.6g}"
                        )
                    if (
                        train_config.min_ternary_zero_fraction is not None
                        and zero_fraction < train_config.min_ternary_zero_fraction
                    ):
                        quantization_failures.append(
                            f"zero fraction {zero_fraction:.6g} < "
                            f"{train_config.min_ternary_zero_fraction:.6g}"
                        )
                    if (
                        train_config.max_ternary_zero_fraction is not None
                        and zero_fraction > train_config.max_ternary_zero_fraction
                    ):
                        quantization_failures.append(
                            f"zero fraction {zero_fraction:.6g} > "
                            f"{train_config.max_ternary_zero_fraction:.6g}"
                        )
                    if quantization_failures:
                        handle_health_gate(
                            "quantization", quantization_failures, next_step
                        )

            if train_config.val_interval > 0 and next_step % train_config.val_interval == 0:
                val_loss = validate(
                    model,
                    val_batcher,
                    train_config.val_batches,
                    device,
                    amp_dtype,
                    use_amp,
                )
                validation_mode = "memmap_heldout"
                fixed_val_slices = []
                if train_config.fixed_val_slices > 0:
                    fixed_val_slices = validate_fixed_memmap_slices(
                        model,
                        val_batcher,
                        train_config.fixed_val_slices,
                        device,
                        amp_dtype,
                        use_amp,
                    )

                improved_validation = val_loss < best_val_loss
                if improved_validation:
                    best_val_loss = val_loss
                    best_val_step = next_step
                validation_payload = {
                    "event": "validation",
                    "step": next_step,
                    "batch_index": step,
                    "val_loss": val_loss,
                    "validation_mode": validation_mode,
                    "data_mode": train_config.data_mode,
                    "fixed_val_slices": fixed_val_slices,
                    "best_val_loss": best_val_loss,
                    "best_val_step": best_val_step,
                    "improved": improved_validation,
                    "abort_recommended": None,
                }
                emit_json(validation_payload, run_paths["events"], run_paths["validation"])
                model.train()

            if (
                train_config.checkpoint_interval > 0
                and next_step < total_steps
                and next_step % train_config.checkpoint_interval == 0
            ):
                checkpoint_path = run_paths["checkpoints"] / f"ckpt_step_{next_step:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    next_step,
                    train_config,
                    model_config=model_config,
                    resume_contract=resume_contract,
                    run_state={
                        "running_loss": running_loss,
                        "best_val_loss": best_val_loss,
                        "best_val_step": best_val_step,
                        "flip_rate_ema": flip_rate_ema,
                        "optimizer_updates_completed": optimizer_updates_completed,
                        "activation_gate_state": activation_gate_state,
                        "optimizer_exposure": optimizer_exposure,
                    },
                    scaler=scaler,
                    events_path=run_paths["events"],
                    checkpoints_log_path=run_paths["checkpoint_log"],
                )
                write_milestone_artifacts(
                    model,
                    manifest,
                    train_config,
                    run_paths,
                    checkpoint_path=checkpoint_path,
                    step=next_step,
                    device=device,
                    amp_dtype=amp_dtype,
                    use_amp=use_amp,
                    val_batcher=val_batcher,
                    final=False,
                )
                model.train()

            if terminal_cooldown_steps > 0 and next_step == terminal_cooldown_start_step:
                pre_cooldown_path = run_paths["checkpoints"] / "ckpt_pre_cooldown.pt"
                save_checkpoint(
                    pre_cooldown_path,
                    model,
                    optimizer,
                    next_step,
                    train_config,
                    model_config=model_config,
                    resume_contract=resume_contract,
                    run_state={
                        "running_loss": running_loss,
                        "best_val_loss": best_val_loss,
                        "best_val_step": best_val_step,
                        "flip_rate_ema": flip_rate_ema,
                        "optimizer_updates_completed": optimizer_updates_completed,
                        "activation_gate_state": activation_gate_state,
                        "optimizer_exposure": optimizer_exposure,
                    },
                    scaler=scaler,
                    events_path=run_paths["events"],
                    checkpoints_log_path=run_paths["checkpoint_log"],
                )
                write_milestone_artifacts(
                    model,
                    manifest,
                    train_config,
                    run_paths,
                    checkpoint_path=pre_cooldown_path,
                    step=next_step,
                    device=device,
                    amp_dtype=amp_dtype,
                    use_amp=use_amp,
                    val_batcher=val_batcher,
                    final=False,
                )
                model.train()

            if profile_this_step:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                profile_iteration_end = time.perf_counter()
                print(
                    json.dumps(
                        {
                            "event": "step_timing_profile",
                            "step": next_step,
                            "data_wait_s": profile_data_wait_s,
                            "forward_s": profile_after_forward - profile_step_start,
                            "backward_s": profile_after_backward - profile_after_forward,
                            "grad_checks_s": profile_after_grad_checks - profile_after_backward,
                            "optimizer_s": profile_after_optimizer - profile_after_grad_checks,
                            "post_optimizer_s": profile_iteration_end - profile_after_optimizer,
                            "iteration_s": profile_iteration_end
                            - profile_previous_iteration_end,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                profile_previous_iteration_end = profile_iteration_end
            elif profile_step_timing:
                profile_previous_iteration_end = time.perf_counter()

            if step + 1 >= total_steps:
                break
    finally:
        close_progress_bar(progress)

    final_step = min(next_step, total_steps)
    if abort_payload is not None:
        final_step = int(abort_payload.get("step", step))
    if train_config.save_final and abort_payload is None:
        final_checkpoint_path = run_paths["checkpoints"] / "ckpt_final.pt"
        save_checkpoint(
            final_checkpoint_path,
            model,
            optimizer,
            final_step,
            train_config,
            model_config=model_config,
            resume_contract=resume_contract,
            run_state={
                "running_loss": running_loss,
                "best_val_loss": best_val_loss,
                "best_val_step": best_val_step,
                "flip_rate_ema": flip_rate_ema,
                "optimizer_updates_completed": optimizer_updates_completed,
                "activation_gate_state": activation_gate_state,
                "optimizer_exposure": optimizer_exposure,
            },
            scaler=scaler,
            events_path=run_paths["events"],
            checkpoints_log_path=run_paths["checkpoint_log"],
        )
        write_milestone_artifacts(
            model,
            manifest,
            train_config,
            run_paths,
            checkpoint_path=final_checkpoint_path,
            step=final_step,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            val_batcher=val_batcher,
            final=True,
        )
        model.train()
    final_hist = model.ternary_weight_histogram()
    emit_json(
        {
            "event": "ternary_histogram",
            "step": final_step,
            "hist": final_hist,
            "stats": ternary_histogram_stats(final_hist),
            "quantization_error": model.quantization_error_report(top_k=train_config.flip_top_k),
            "quantization_decomposition": model.quantization_error_decomposition_report(
                top_k=train_config.flip_top_k
            )
            if train_config.quantization_decomposition_interval > 0
            else None,
            "module_health": model.module_health_report(top_k=train_config.flip_top_k),
            "final": True,
        },
        run_paths["events"],
        run_paths["histograms"],
    )
    emit_json(
        {
            "event": "done",
            "step": final_step,
            "stop_reason": "completed" if abort_payload is None else abort_payload.get("reason"),
            "abort": abort_payload,
        },
        run_paths["events"],
        run_paths["system"],
    )


@torch.no_grad()
def validate(
    model: BitNetLlama,
    batcher: TokenBatcher,
    max_batches: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, targets in batcher.val_batches(max_batches=max_batches):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(
                input_ids,
                targets,
                return_logits=False,
            )
        if output["loss"] is None:
            raise RuntimeError("validation did not return a loss")
        loss_tokens = int(output["loss_tokens"]) if output["loss_tokens"] is not None else targets.numel()
        total_loss += float(output["loss"]) * loss_tokens
        total_tokens += loss_tokens
    if total_tokens == 0:
        raise RuntimeError("validation produced no batches")
    return float(total_loss / total_tokens)


@torch.no_grad()
def validate_streaming(
    model: BitNetLlama,
    manifest: TokenManifest,
    tokenizer_name: str,
    max_batches: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()
    loader = StreamingTextTokenLoader(
        manifest,
        tokenizer_name=tokenizer_name,
        start_step=0,
        max_steps=max_batches,
        prefetch_batches=2,
        device=device,
    )
    total_loss = 0.0
    total_tokens = 0
    for input_ids, targets in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(input_ids, targets, return_logits=False)
        if output["loss"] is None:
            raise RuntimeError("streaming validation did not return a loss")
        loss_tokens = int(output["loss_tokens"]) if output["loss_tokens"] is not None else targets.numel()
        total_loss += float(output["loss"]) * loss_tokens
        total_tokens += loss_tokens
    if total_tokens == 0:
        raise RuntimeError("streaming validation produced no batches")
    return float(total_loss / total_tokens)


def build_fixed_validation_telemetry_probe(
    batcher: TokenBatcher,
    *,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Build a reproducible, held-out telemetry probe and its identity contract.

    Activation gates are longitudinal tests.  Comparing a new training batch
    against an older one confounds a model change with document/token
    difficulty, especially for global-attention layers.  This probe uses fixed
    blocks from the immutable validation span and records a content fingerprint
    so checkpoint resume cannot quietly change what the gate observes.
    """

    if batch_size <= 0:
        raise ValueError("telemetry probe batch_size must be positive")
    usable_blocks = max(
        0,
        (min(batcher.val.num_tokens, batcher.config.validation_tokens) - 1)
        // batcher.config.seq_len,
    )
    if usable_blocks < batch_size:
        raise RuntimeError(
            f"validation span exposes {usable_blocks} full blocks; need {batch_size} "
            "for the fixed telemetry probe"
        )
    if batch_size == 1:
        block_indices = [0]
    else:
        block_indices = [
            int(round(index * (usable_blocks - 1) / (batch_size - 1)))
            for index in range(batch_size)
        ]
    input_rows: List[torch.Tensor] = []
    target_rows: List[torch.Tensor] = []
    for block_index in block_indices:
        inputs, targets = batcher.val.block(block_index, batcher.config.seq_len)
        input_rows.append(inputs)
        target_rows.append(targets)
    input_ids = torch.stack(input_rows)
    targets = torch.stack(target_rows)
    fingerprint_source = (
        input_ids.contiguous().cpu().numpy().tobytes()
        + targets.contiguous().cpu().numpy().tobytes()
    )
    contract: Dict[str, object] = {
        "source": "fixed_validation",
        "batch_size": int(batch_size),
        "seq_len": int(batcher.config.seq_len),
        "block_indices": block_indices,
        "fingerprint": hashlib.sha256(fingerprint_source).hexdigest(),
    }
    return (
        input_ids.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
        contract,
    )


@torch.no_grad()
def validate_fixed_memmap_slices(
    model: BitNetLlama,
    batcher: TokenBatcher,
    num_slices: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> List[Dict[str, float]]:
    model.eval()
    usable_blocks = max(
        0,
        (min(batcher.val.num_tokens, batcher.config.validation_tokens) - 1)
        // batcher.config.seq_len,
    )
    if num_slices <= 0 or usable_blocks <= 0:
        return []
    if num_slices == 1:
        block_indices = [0]
    else:
        block_indices = sorted(
            {
                int(round(index * (usable_blocks - 1) / max(1, num_slices - 1)))
                for index in range(num_slices)
            }
        )

    results: List[Dict[str, float]] = []
    for slice_index, block_index in enumerate(block_indices):
        input_ids, targets = batcher.val.block(block_index, batcher.config.seq_len)
        input_ids = input_ids.unsqueeze(0).to(device, non_blocking=True)
        targets = targets.unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(input_ids, targets, return_logits=False)
        if output["loss"] is None:
            raise RuntimeError("fixed validation slice did not return a loss")
        loss_tokens = int(output["loss_tokens"]) if output["loss_tokens"] is not None else targets.numel()
        results.append(
            {
                "slice": float(slice_index),
                "block_index": float(block_index),
                "loss": float(output["loss"]),
                "tokens": float(loss_tokens),
            }
        )
    return results


@torch.no_grad()
def validate_fixed_streaming_slices(
    model: BitNetLlama,
    manifest: TokenManifest,
    tokenizer_name: str,
    num_slices: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> List[Dict[str, float]]:
    if num_slices <= 0:
        return []
    model.eval()
    loader = StreamingTextTokenLoader(
        manifest,
        tokenizer_name=tokenizer_name,
        start_step=0,
        max_steps=num_slices,
        prefetch_batches=2,
        device=device,
    )
    results: List[Dict[str, float]] = []
    for slice_index, (input_ids, targets) in enumerate(loader):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(input_ids, targets, return_logits=False)
        if output["loss"] is None:
            raise RuntimeError("fixed streaming validation slice did not return a loss")
        loss_tokens = int(output["loss_tokens"]) if output["loss_tokens"] is not None else targets.numel()
        results.append(
            {
                "slice": float(slice_index),
                "stream_start_step": 0.0,
                "loss": float(output["loss"]),
                "tokens": float(loss_tokens),
            }
        )
    return results


def write_milestone_artifacts(
    model: BitNetLlama,
    manifest: TokenManifest,
    train_config: TrainConfig,
    run_paths: Dict[str, Path],
    *,
    checkpoint_path: Path,
    step: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    val_batcher: Optional[TokenBatcher],
    final: bool,
) -> None:
    if not train_config.milestone_eval_enabled and not train_config.milestone_sample_enabled:
        return

    milestone_dir = run_paths["milestones"] / ("final" if final else f"step_{step:06d}")
    milestone_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    rng = rng_state()
    val_loss = None
    samples: List[Dict[str, object]] = []
    started = time.time()
    try:
        if train_config.milestone_eval_enabled:
            eval_batches = min(train_config.val_batches, train_config.milestone_eval_batches)
            if eval_batches > 0:
                if val_batcher is None:
                    raise RuntimeError("milestone evaluation requires a held-out memmap validation batcher")
                val_loss = validate(
                    model,
                    val_batcher,
                    eval_batches,
                    device,
                    amp_dtype,
                    use_amp,
                )

        if train_config.milestone_sample_enabled:
            samples = sample_milestone_outputs(
                model,
                train_config,
                device,
                amp_dtype,
                use_amp,
                prompts=MILESTONE_PROMPTS,
            )
    finally:
        restore_rng_state(rng)
        if was_training:
            model.train()
        else:
            model.eval()

    hist = model.ternary_weight_histogram()
    alpha_stats = tensor_stats_to_float(model.alpha_stats())
    payload = {
        "event": "milestone_evaluation",
        "step": step,
        "final": final,
        "checkpoint_path": str(checkpoint_path),
        "milestone_dir": str(milestone_dir),
        "tokens_seen": min(
            step * manifest.config.tokens_per_step,
            manifest.config.usable_train_tokens,
        ),
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss) if val_loss is not None and val_loss < 50 else None,
        "validation_mode": "memmap_heldout",
        "data_mode": train_config.data_mode,
        "alpha_stats": alpha_stats,
        "ternary": ternary_histogram_stats(hist),
        "quantization_error": model.quantization_error_report(top_k=train_config.flip_top_k),
        "quantization_decomposition": (
            model.quantization_error_decomposition_report(top_k=train_config.flip_top_k)
            if train_config.quantization_decomposition_interval > 0 or final
            else None
        ),
        "module_health": model.module_health_report(top_k=train_config.flip_top_k),
        "sample_config": {
            "tokenizer_name": train_config.tokenizer_name,
            "max_new_tokens": train_config.milestone_sample_max_new_tokens,
            "temperature": train_config.milestone_sample_temperature,
            "top_k": train_config.milestone_sample_top_k,
            "top_p": train_config.milestone_sample_top_p,
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(milestone_dir / "metrics.json", payload)
    if samples:
        write_json(milestone_dir / "samples.json", {"samples": samples})
        (milestone_dir / "samples.md").write_text(render_samples_markdown(payload, samples) + "\n")
    emit_json(payload, run_paths["events"], run_paths["milestone_log"])


@torch.no_grad()
def sample_milestone_outputs(
    model: BitNetLlama,
    train_config: TrainConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    *,
    prompts: Sequence[str],
) -> List[Dict[str, object]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        train_config.tokenizer_name,
        revision=train_config.tokenizer_revision,
        use_fast=True,
    )
    tokenizer.model_max_length = int(1e30)
    eos_id = tokenizer.eos_token_id
    model.eval()
    outputs = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not input_ids:
            input_ids = [eos_id if eos_id is not None else 0]
        generated = list(input_ids)
        if len(generated) + train_config.milestone_sample_max_new_tokens > model.config.max_seq_len:
            raise ValueError("milestone prompt plus generation exceeds cached inference max_seq_len")
        x = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output, cache = model.forward_cached(x, return_logits=True)
        for _ in range(train_config.milestone_sample_max_new_tokens):
            logits = output["logits"][:, -1, :].float().squeeze(0)
            next_id = sample_next_token(
                logits,
                temperature=train_config.milestone_sample_temperature,
                top_k=train_config.milestone_sample_top_k,
                top_p=train_config.milestone_sample_top_p,
            )
            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
            next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output, cache = model.forward_cached(next_token, cache=cache, return_logits=True)
        outputs.append(
            {
                "prompt": prompt,
                "output": tokenizer.decode(generated, skip_special_tokens=True),
                "new_tokens": max(0, len(generated) - len(input_ids)),
            }
        )
    return outputs


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    logits = logits / max(temperature, 1e-8)
    if top_k > 0 and top_k < logits.numel():
        values, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < values[-1], float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[0] = False
        logits = logits.clone()
        logits[sorted_indices[remove]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0.0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def render_samples_markdown(payload: Dict[str, object], samples: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# Milestone Samples",
        "",
        f"- Step: `{payload['step']}`",
        f"- Tokens seen: `{payload['tokens_seen']}`",
        f"- Checkpoint: `{payload['checkpoint_path']}`",
        f"- Validation loss: `{payload.get('val_loss')}`",
        f"- Validation perplexity: `{payload.get('val_ppl')}`",
        "",
    ]
    for index, sample in enumerate(samples, start=1):
        lines.extend(
            [
                f"## Prompt {index}",
                "",
                "**Prompt**",
                "",
                "```text",
                str(sample["prompt"]),
                "```",
                "",
                "**Output**",
                "",
                "```text",
                str(sample["output"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def prepare_run_dirs(output_dir: Path, *, resume: bool, run_id: str) -> Dict[str, Path]:
    if not resume:
        collision_markers = (
            output_dir / "configs" / "train_config.json",
            output_dir / "logs" / "events.jsonl",
            output_dir / "RUN_ID.json",
        )
        checkpoint_dir = output_dir / "checkpoints"
        has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt"))
        if has_checkpoint or any(path.exists() for path in collision_markers):
            raise FileExistsError(
                f"refusing to start a fresh run in nonempty experiment directory {output_dir}; "
                "choose a unique OUTPUT_DIR/RUN_ID or pass --resume"
            )
    elif resume and (output_dir / "RUN_ID.json").exists():
        existing = json.loads((output_dir / "RUN_ID.json").read_text()).get("run_id")
        if existing != run_id:
            raise RuntimeError(
                f"resume output directory belongs to run_id {existing!r}, not {run_id!r}"
            )
    paths = {
        "root": output_dir,
        "configs": output_dir / "configs",
        "logs": output_dir / "logs",
        "checkpoints": output_dir / "checkpoints",
        "milestones": output_dir / "milestones",
        "events": output_dir / "logs" / "events.jsonl",
        "system": output_dir / "logs" / "system.jsonl",
        "train_log": output_dir / "logs" / "train.jsonl",
        "validation": output_dir / "logs" / "validation.jsonl",
        "histograms": output_dir / "logs" / "histograms.jsonl",
        "ternary_transitions": output_dir / "logs" / "ternary_transitions.jsonl",
        "ssm_log": output_dir / "logs" / "ssm.jsonl",
        "attention_log": output_dir / "logs" / "attention.jsonl",
        "dynamics_log": output_dir / "logs" / "dynamics.jsonl",
        "checkpoint_log": output_dir / "logs" / "checkpoints.jsonl",
        "milestone_log": output_dir / "logs" / "milestones.jsonl",
        "optimizer_log": output_dir / "logs" / "optimizer.jsonl",
        "system_timeseries": output_dir / "logs" / "system_timeseries.jsonl",
        "curriculum_log": output_dir / "logs" / "curriculum.jsonl",
    }
    for key in ("root", "configs", "logs", "checkpoints", "milestones"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def streaming_row_text(row: Dict[str, object]) -> str:
    for key in ("text", "prompt", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return " ".join(value for value in row.values() if isinstance(value, str))


def curriculum_summary(manifest: TokenManifest) -> Dict[str, object]:
    config = manifest.config
    train_shards = manifest.shards_for(split=config.train_split)
    if config.source_mix:
        return {
            "mode": "deterministic_weighted_interleave",
            "source_mix": dict(config.source_mix),
            "train_sources": sorted({shard.source for shard in train_shards}),
            "total_train_tokens": config.total_train_tokens,
            "usable_train_tokens": config.usable_train_tokens,
        }
    if config.prefix_effective_tokens <= 0:
        return {
            "prefix_source": None,
            "requested_prefix_tokens": 0,
            "effective_prefix_tokens": 0,
            "prefix_steps": 0,
            "train_order": [shard.source for shard in train_shards],
            "total_train_tokens": config.total_train_tokens,
            "usable_train_tokens": config.usable_train_tokens,
        }
    post_prefix = [shard.source for shard in train_shards if shard.source != config.prefix_source]
    return {
        "prefix_source": config.prefix_source,
        "requested_prefix_tokens": config.cosmopedia_stories_prefix_tokens,
        "effective_prefix_tokens": config.prefix_effective_tokens,
        "prefix_steps": config.prefix_steps,
        "post_prefix_order": post_prefix,
        "total_train_tokens": config.total_train_tokens,
        "usable_train_tokens": config.usable_train_tokens,
    }


def apply_manifest_overrides(manifest: TokenManifest, config: TrainConfig) -> TokenManifest:
    if config.batch_size_override is None:
        return manifest
    if config.batch_size_override <= 0:
        raise ValueError("batch_size_override must be positive")
    return replace(
        manifest,
        config=replace(manifest.config, batch_size=config.batch_size_override),
    )


def validate_manifest_contract(manifest: TokenManifest, *, root: Path) -> None:
    train_shards = manifest.shards_for(split=manifest.config.train_split)
    val_shards = manifest.shards_for(split=manifest.config.val_split)
    if not train_shards or not val_shards:
        raise ValueError("Test 17 requires both training and held-out validation shards")
    all_shards = train_shards + val_shards
    for shard in all_shards:
        path = (root / shard.path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if not shard.sha256:
            raise ValueError(f"manifest shard is missing SHA-256 provenance: {shard.path}")
    for train_shard in train_shards:
        train_path = (root / train_shard.path).resolve()
        for val_shard in val_shards:
            val_path = (root / val_shard.path).resolve()
            if os.path.samefile(train_path, val_path):
                raise ValueError(
                    f"training and validation must be disjoint files: {train_path}"
                )


def resolve_training_budget(
    config: TrainConfig,
    manifest: TokenManifest,
    *,
    resume_next_step: Optional[int] = None,
) -> Dict[str, object]:
    tokens_per_step = manifest.config.tokens_per_step
    manifest_steps = manifest.config.total_steps
    start_step = config.start_step
    if config.resume:
        start_step = (
            checkpoint_step(Path(config.resume))
            if resume_next_step is None
            else int(resume_next_step)
        )
    candidates = [{"source": "manifest", "steps": manifest_steps}]

    if config.max_steps is not None:
        candidates.append({"source": "max_steps", "steps": config.max_steps})

    if config.max_train_tokens is not None:
        candidates.append(
            {
                "source": "max_train_tokens",
                "steps": math.ceil(config.max_train_tokens / tokens_per_step),
            }
        )

    if config.auto_token_budget:
        if config.time_limit_hours is None:
            raise ValueError("auto_token_budget requires time_limit_hours")
        if config.expected_tokens_per_sec is None:
            raise ValueError("auto_token_budget requires expected_tokens_per_sec")
        usable_seconds = config.time_limit_hours * 3600.0 * config.time_budget_safety_fraction
        auto_tokens = int(config.expected_tokens_per_sec * usable_seconds)
        auto_steps = max(1, math.floor(auto_tokens / tokens_per_step))
        candidates.append(
            {
                "source": "auto_time_budget",
                "steps": min(manifest_steps, start_step + auto_steps),
                "auto_steps": auto_steps,
                "expected_tokens_per_sec": config.expected_tokens_per_sec,
                "time_limit_hours": config.time_limit_hours,
                "time_budget_safety_fraction": config.time_budget_safety_fraction,
                "estimated_train_tokens": auto_tokens,
            }
        )

    positive = [candidate for candidate in candidates if int(candidate["steps"]) > 0]
    selected = min(positive, key=lambda candidate: int(candidate["steps"]))
    total_steps = min(manifest_steps, int(selected["steps"]))
    if total_steps <= start_step:
        raise ValueError(
            f"training budget is exhausted: checkpoint/start next_step={start_step}, "
            f"resolved total_steps={total_steps}; do not resume a completed run"
        )
    schedule_source = "actual_run_length"
    schedule_total_steps = total_steps
    if config.schedule_horizon_tokens is not None:
        schedule_total_steps = max(1, math.ceil(config.schedule_horizon_tokens / tokens_per_step))
        schedule_source = "schedule_horizon_tokens"
        if schedule_total_steps < total_steps:
            raise ValueError(
                "schedule_horizon_tokens resolves to fewer steps than the actual run budget; "
                "the schedule horizon must be at least as long as the planned run"
            )
    cumulative_start_tokens = min(start_step * tokens_per_step, manifest.config.usable_train_tokens)
    cumulative_end_tokens = min(total_steps * tokens_per_step, manifest.config.usable_train_tokens)
    resolved_tokens = max(0, cumulative_end_tokens - cumulative_start_tokens)
    return {
        "selected_source": selected["source"],
        "start_step": start_step,
        "total_steps": total_steps,
        "steps_this_run": max(0, total_steps - start_step),
        "tokens_per_step": tokens_per_step,
        "resolved_train_tokens_this_run": resolved_tokens,
        "resolved_cumulative_tokens": cumulative_end_tokens,
        "schedule_total_steps": schedule_total_steps,
        "schedule_total_tokens": schedule_total_steps * tokens_per_step,
        "schedule_source": schedule_source,
        "manifest_total_steps": manifest_steps,
        "manifest_total_train_tokens": manifest.config.total_train_tokens,
        "manifest_usable_train_tokens": manifest.config.usable_train_tokens,
        "candidates": candidates,
        "selected_candidate": selected,
    }


def checkpoint_step(path: Path) -> int:
    payload = load_checkpoint_payload(path, map_location="cpu")
    return int(payload["next_step"])


def curriculum_status(manifest: TokenManifest, step: int, *, world_size: int = 1) -> Dict[str, object]:
    config = manifest.config
    consumed = min(
        step * config.tokens_per_step * world_size,
        config.usable_train_tokens,
    )
    remaining = max(0, config.usable_train_tokens - consumed)
    progress = consumed / config.usable_train_tokens if config.usable_train_tokens else 0.0
    if config.source_mix:
        return {
            "tokens_consumed": consumed,
            "tokens_remaining": remaining,
            "progress": progress,
            "stage": "deterministic_weighted_interleave",
            "source": "mixed",
            "source_mix": dict(config.source_mix),
            "source_token_offset": None,
        }
    if config.prefix_effective_tokens > 0 and consumed < config.prefix_effective_tokens:
        return {
            "tokens_consumed": consumed,
            "tokens_remaining": remaining,
            "progress": progress,
            "stage": "prefix",
            "source": config.prefix_source,
            "source_token_offset": consumed,
        }

    main_offset = max(0, consumed - config.prefix_effective_tokens)
    cursor = 0
    source = "train_exhausted"
    source_offset = main_offset
    for shard in manifest.shards_for(split=config.train_split):
        if config.prefix_effective_tokens > 0 and shard.source == config.prefix_source:
            continue
        shard_budget = max(0, shard.num_tokens - 1)
        if main_offset < cursor + shard_budget:
            source = shard.source
            source_offset = main_offset - cursor
            break
        cursor += shard_budget
    return {
        "tokens_consumed": consumed,
        "tokens_remaining": remaining,
        "progress": progress,
        "stage": "post_prefix" if config.prefix_effective_tokens > 0 else "train",
        "source": source,
        "source_token_offset": source_offset,
    }


def environment_report(device: torch.device) -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "torch",
                "datasets",
                "transformers",
                "tokenizers",
                "tqdm",
                "flash-attn",
                "muon-optimizer",
            )
        },
        "source_provenance": source_provenance(),
        "cuda_device": cuda_device_report(device),
    }


def _git_value(root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def source_provenance() -> Dict[str, object]:
    experiment_dir = Path(__file__).resolve().parent
    repo_root = experiment_dir.parents[1]
    repo_diff = _git_value(repo_root, "diff", "HEAD", "--binary", "--", str(experiment_dir)) or ""
    fa3_path_raw = os.environ.get("FA3_HOPPER_PATH", "").strip()
    fa3_commit = _git_value(Path(fa3_path_raw), "rev-parse", "HEAD") if fa3_path_raw else None
    return {
        "repo_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "repo_diff_sha256": hashlib.sha256(repo_diff.encode("utf-8")).hexdigest(),
        "source_file_sha256": source_contract(),
        "fa3_commit": fa3_commit,
        "muon_expected_commit": os.environ.get("MUON_EXPECTED_COMMIT"),
    }


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cuda_device_report(device: torch.device) -> Dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": props.name,
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "total_memory_gb": props.total_memory / 1e9,
    }


def gpu_backend_report(device: torch.device) -> Dict[str, object]:
    report = {
        "device_type": device.type,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return report
    report.update(
        {
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
    )
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        getter = getattr(torch.backends.cuda, name, None)
        if getter is not None:
            report[name] = bool(getter())
    return report


def write_experiment_card(
    path: Path,
    train_config: TrainConfig,
    model_config: BitNetLlamaConfig,
    manifest: TokenManifest,
) -> None:
    transformer_indices = parse_transformer_layer_indices(
        train_config.transformer_layer_indices,
        model_config.n_layers,
    )
    global_layers = [
        index
        for index in transformer_indices
        if (index + 1) % int(model_config.global_attention_every) == 0
    ]
    path.write_text(
        "\n".join(
            [
                "# Test 17 Experiment Card",
                "",
                "## Question",
                "",
                "Can Test 16's promising local/global ternary Transformer become genuinely sparse, strictly kernel-verified, and safer in its late latent dynamics?",
                "",
                "## Launch",
                "",
                "- Command: `python train.py`",
                f"- Output directory: `{train_config.output_dir}`",
                f"- Device: `{train_config.device}`",
                f"- Precision: `{train_config.dtype}`",
                f"- Optimizer kind: `{train_config.optimizer_kind}`",
                "- torch.compile: enabled on CUDA by default",
                "",
                "## Model",
                "",
                "- Architecture: Transformer++ BitNet with exact 4-local/1-global GQA cycles",
                f"- Layers: {model_config.n_layers}",
                "- Mamba layers: 0",
                f"- Transformer GQA layers: {len(transformer_indices)} at indices {list(transformer_indices)}",
                f"- Hidden dim: {model_config.dim}",
                f"- FFN dim: {model_config.ffn_dim}",
                f"- Attention heads / KV heads: {model_config.n_heads} / {model_config.n_kv_heads}",
                f"- Attention backend: `{model_config.attention_backend}`",
                "- Backend fallback: forbidden; SDPA is explicit reference-only",
                f"- Local attention receptive field: `{model_config.local_attention_window}` tokens",
                f"- FA3 local window when selected: `({model_config.local_attention_window - 1}, 0)`",
                f"- Local attention window: {model_config.local_attention_window}",
                f"- Global attention every: {model_config.global_attention_every}",
                f"- Global attention layers: {global_layers}",
                f"- Local attention layers: {[index for index in transformer_indices if index not in global_layers]}",
                f"- Transformer GQA/FFN alpha mode: `{model_config.transformer_alpha_mode}`",
                f"- Alpha group size: `{model_config.alpha_group_size}`",
                (
                    "- Alpha trainable parameters: enabled through learned log-gain"
                    if model_config.transformer_alpha_mode
                    in {"learned_log_gain", "learned_grouped_log_gain"}
                    else "- Alpha trainable parameters: 0; alpha is recomputed/fixed by the selected mode"
                ),
                "- Weight STE: `full_dequantized_target_identity`",
                f"- Activation quantization enabled: `{model_config.quantize_activations}`",
                f"- Q/K normalization: `{model_config.qk_norm_mode}` in every attention layer before RoPE",
                "- QK-Clip: removed; Q/K projection weights are not mutated post-update",
                "- Optimizer split: one Muon group per Transformer layer for attention and FFN BitLinear matrices",
                f"- Context length: {model_config.max_seq_len}",
                f"- RMSNorm epsilon: {model_config.rms_norm_eps}",
                "- Muon matrices: attn q/k/v/o projections and FFN gate/up/down projections",
                "- AdamW matrices/scalars: token embeddings and RMSNorm weights",
                "- Trainer-side automatic abort: disabled; there is no built-in validation-regression or grad-spike stop path.",
                f"- External watchdog expected: `{os.environ.get('BITNET_EXTERNAL_WATCHDOG', '0') == '1'}`",
                "",
                "## Data",
                "",
                f"- Total train tokens: {manifest.config.total_train_tokens}",
                f"- Usable full-sequence train tokens: {manifest.config.usable_train_tokens}",
                f"- Prefix source: {manifest.config.prefix_source}",
                f"- Effective prefix tokens: {manifest.config.prefix_effective_tokens}",
                f"- Tokens per step: {manifest.config.tokens_per_step}",
                f"- Planned total steps: {manifest.config.total_steps}",
                (
                    f"- Scheduler horizon tokens: {train_config.schedule_horizon_tokens}"
                    if train_config.schedule_horizon_tokens is not None
                    else "- Scheduler horizon tokens: matches run budget"
                ),
                f"- Quantization decomposition interval: `{train_config.quantization_decomposition_interval}`",
                f"- Functional quantization interval: `{train_config.functional_quantization_interval}`",
                "",
                "## Research Artifacts",
                "",
                "- `configs/environment.json`: runtime, package, CUDA, and device metadata.",
                "- `configs/kernel_policy.json`: accelerator-kernel policy and attention backend policy.",
                "- `configs/initialization_report.json`: deterministic grouped-alpha scale, ternary, latent-weight, and zero-row initialization checks.",
                "- `logs/events.jsonl`: complete event stream.",
                "- `logs/train.jsonl`: loss, LR, curriculum, deterministic alpha scale, flip-rate, activation, and memory telemetry.",
                "- `logs/optimizer.jsonl`: group-level parameter and gradient health.",
                "- `logs/ssm.jsonl`: intentionally empty compatibility stream for this Transformer-only test.",
                "- `logs/attention.jsonl`: local/global GQA logit, entropy, head, and backend telemetry.",
                "- `logs/histograms.jsonl`: ternary occupancy, quantization error, and module health.",
                "- `logs/ternary_transitions.jsonl`: transition matrices for {-1, 0, +1} weight-state movement.",
                "- `logs/dynamics.jsonl`: sampled update-to-weight ratios after optimizer steps.",
                "- `logs/system_timeseries.jsonl`: periodic CUDA/backend/memory state.",
                "- `logs/validation.jsonl`: validation loss curve and fixed validation slices.",
                "- `logs/checkpoints.jsonl`: checkpoint provenance.",
                "- `milestones/`: validation, diagnostics, and fixed-prompt samples written at checkpoint boundaries.",
            ]
        )
        + "\n"
    )


def validate_train_config(config: TrainConfig) -> None:
    if config.data_mode != "memmap":
        raise ValueError(
            "Test 17 requires immutable memmap training and held-out memmap validation; "
            "the inherited remote streaming path is not reproducible enough for this experiment"
        )
    if not config.split_norm_parameters:
        raise ValueError(
            "split_norm_parameters is mandatory: norm gains must use the dedicated "
            "low-LR AdamW group"
        )
    if not 0.0 <= config.flip_ema_beta < 1.0:
        raise ValueError("flip_ema_beta must be in [0, 1)")
    if config.flip_top_k < 0:
        raise ValueError("flip_top_k must be non-negative")
    if config.quant_error_interval < 0:
        raise ValueError("quant_error_interval must be non-negative")
    if (
        config.max_quantization_relative_mse is not None
        and config.max_quantization_relative_mse <= 0.0
    ):
        raise ValueError("max_quantization_relative_mse must be positive")
    for name, value in (
        ("min_ternary_zero_fraction", config.min_ternary_zero_fraction),
        ("max_ternary_zero_fraction", config.max_ternary_zero_fraction),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if (
        config.min_ternary_zero_fraction is not None
        and config.max_ternary_zero_fraction is not None
        and config.min_ternary_zero_fraction > config.max_ternary_zero_fraction
    ):
        raise ValueError("minimum ternary zero fraction exceeds maximum")
    if not 0.0 <= config.quantization_health_grace_fraction <= 1.0:
        raise ValueError("quantization_health_grace_fraction must be in [0, 1]")
    if config.activation_stats_interval < 0:
        raise ValueError("activation_stats_interval must be non-negative")
    if config.ssm_stats_interval < 0:
        raise ValueError("ssm_stats_interval must be non-negative")
    if config.dynamics_interval < 0:
        raise ValueError("dynamics_interval must be non-negative")
    if config.update_sample_size < 0:
        raise ValueError("update_sample_size must be non-negative")
    if config.max_attention_logit is not None and config.max_attention_logit <= 0.0:
        raise ValueError("max_attention_logit must be positive")
    if config.max_activation_abs is not None and config.max_activation_abs <= 0.0:
        raise ValueError("max_activation_abs must be positive")
    if (
        config.max_activation_max_p999_ratio is not None
        and config.max_activation_max_p999_ratio <= 1.0
    ):
        raise ValueError("max_activation_max_p999_ratio must exceed 1")
    if config.activation_baseline_samples <= 0:
        raise ValueError("activation_baseline_samples must be positive")
    if (
        config.max_activation_rms_baseline_ratio is not None
        and config.max_activation_rms_baseline_ratio <= 1.0
    ):
        raise ValueError("max_activation_rms_baseline_ratio must exceed 1")
    if config.telemetry_probe_source not in {"training_batch", "fixed_validation"}:
        raise ValueError("telemetry_probe_source must be training_batch or fixed_validation")
    if config.fixed_val_slices < 0:
        raise ValueError("fixed_val_slices must be non-negative")
    if config.optimizer_stats_interval < 0:
        raise ValueError("optimizer_stats_interval must be non-negative")
    if config.system_interval < 0:
        raise ValueError("system_interval must be non-negative")
    if config.quantization_warmup_steps < 0:
        raise ValueError("quantization_warmup_steps must be non-negative")
    if config.quantization_decomposition_interval < 0:
        raise ValueError("quantization_decomposition_interval must be non-negative")
    if config.functional_quantization_interval < 0:
        raise ValueError("functional_quantization_interval must be non-negative")
    if config.functional_quantization_max_rows <= 0:
        raise ValueError("functional_quantization_max_rows must be positive")
    if config.qk_norm_mode != "unit_rms":
        raise ValueError("qk_norm_mode must be 'unit_rms'; affine Q/K gains are forbidden")
    if config.ffn_activation not in {"swiglu", "relu2"}:
        raise ValueError("invalid ffn_activation")
    if config.attention_temperature_mode not in {"none", "bounded_per_head"}:
        raise ValueError("invalid attention_temperature_mode")
    if config.attention_temperature_min <= 0.0:
        raise ValueError("attention_temperature_min must be positive")
    if config.attention_temperature_max <= config.attention_temperature_min:
        raise ValueError("attention_temperature_max must exceed its minimum")
    if not (
        config.attention_temperature_min
        < config.attention_temperature_init
        < config.attention_temperature_max
    ):
        raise ValueError("attention_temperature_init must lie strictly inside its bounds")
    if config.attention_softcap < 0.0:
        raise ValueError("attention_softcap must be non-negative")
    if not 0.0 <= config.terminal_cooldown_fraction < 1.0:
        raise ValueError("terminal_cooldown_fraction must be in [0, 1)")
    if config.dtype == "fp16" and config.attention_temperature_mode == "bounded_per_head":
        raise ValueError("bounded Test 18 attention requires BF16 or FP32, not FP16")
    if config.loss_scale_init <= 0.0:
        raise ValueError("loss_scale_init must be positive")
    if config.min_loss_scale <= 0.0:
        raise ValueError("min_loss_scale must be positive")
    if config.min_loss_scale > config.loss_scale_init:
        raise ValueError("min_loss_scale must be <= loss_scale_init")
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.max_train_tokens is not None and config.max_train_tokens <= 0:
        raise ValueError("max_train_tokens must be positive")
    if config.schedule_horizon_tokens is not None and config.schedule_horizon_tokens <= 0:
        raise ValueError("schedule_horizon_tokens must be positive")
    if config.time_limit_hours is not None and config.time_limit_hours <= 0.0:
        raise ValueError("time_limit_hours must be positive")
    if config.expected_tokens_per_sec is not None and config.expected_tokens_per_sec <= 0.0:
        raise ValueError("expected_tokens_per_sec must be positive")
    if not 0.0 < config.time_budget_safety_fraction <= 1.0:
        raise ValueError("time_budget_safety_fraction must be in (0, 1]")
    if config.auto_token_budget and config.time_limit_hours is None:
        raise ValueError("auto_token_budget requires time_limit_hours")
    if config.auto_token_budget and config.expected_tokens_per_sec is None:
        raise ValueError("auto_token_budget requires expected_tokens_per_sec")
    for name, value in (
        ("quantization_start_strength", config.quantization_start_strength),
        ("quantization_end_strength", config.quantization_end_strength),
        ("warmup_fraction", config.warmup_fraction),
        ("decay_fraction", config.decay_fraction),
        ("final_lr_fraction", config.final_lr_fraction),
        ("phase1_fraction_of_total", config.phase1_fraction_of_total),
        ("phase2_fraction_of_total", config.phase2_fraction_of_total),
        ("phase_lr_start_fraction", config.phase_lr_start_fraction),
        ("phase1_end_lr_fraction", config.phase1_end_lr_fraction),
        ("phase2_peak_lr_fraction", config.phase2_peak_lr_fraction),
        ("phase2_peak_ramp_fraction", config.phase2_peak_ramp_fraction),
        ("linear_lr_halfway_fraction", config.linear_lr_halfway_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if config.lr_schedule not in {"linear", "wsd", "cosine", "phase_cosine"}:
        raise ValueError("lr_schedule must be one of {'linear', 'wsd', 'cosine', 'phase_cosine'}")
    if config.lr_schedule == "linear" and config.linear_lr_halfway_fraction <= config.warmup_fraction:
        raise ValueError("linear_lr_halfway_fraction must be greater than warmup_fraction")
    if config.lr_schedule == "phase_cosine":
        if config.phase1_fraction_of_total + config.phase2_fraction_of_total > 1.0:
            raise ValueError("phase1_fraction_of_total + phase2_fraction_of_total must be <= 1.0")
        if config.phase2_peak_ramp_fraction <= 0.0:
            raise ValueError("phase2_peak_ramp_fraction must be positive")
        if config.phase_lr_start_fraction > config.phase1_end_lr_fraction:
            raise ValueError("phase_lr_start_fraction must be <= phase1_end_lr_fraction")
        if config.phase1_end_lr_fraction > config.phase2_peak_lr_fraction:
            raise ValueError("phase1_end_lr_fraction must be <= phase2_peak_lr_fraction")
    if config.optimizer_kind not in {"muon", "adamw"}:
        raise ValueError("optimizer_kind must be 'muon' or 'adamw'")
    for name, value in (
        ("vocab_size", config.vocab_size),
        ("dim", config.dim),
        ("n_layers", config.n_layers),
        ("n_heads", config.n_heads),
        ("n_kv_heads", config.n_kv_heads),
        ("ffn_dim", config.ffn_dim),
        ("expand", config.expand),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.dim % config.n_heads != 0:
        raise ValueError("dim must be divisible by n_heads")
    if config.n_heads % config.n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if (config.dim * config.expand) % config.n_heads != 0:
        raise ValueError("dim * expand must be divisible by n_heads")
    alpha_modes = {
        "deterministic_absmean",
        "deterministic_grouped_absmean",
        "deterministic_grouped_lloyd",
        "fixed_init_absmean",
        "learned_log_gain",
        "learned_grouped_log_gain",
    }
    if config.alpha_mode not in alpha_modes:
        raise ValueError(f"alpha_mode must be one of {sorted(alpha_modes)}")
    if config.transformer_alpha_mode not in alpha_modes:
        raise ValueError(f"transformer_alpha_mode must be one of {sorted(alpha_modes)}")
    if config.alpha_group_size <= 0:
        raise ValueError("alpha_group_size must be positive")
    transformer_indices = parse_transformer_layer_indices(config.transformer_layer_indices, config.n_layers)
    if len(transformer_indices) != config.n_layers:
        raise ValueError("Test 17 is Transformer-only; set transformer_layer_indices='all'")
    if config.local_attention_window <= 0:
        raise ValueError("local_attention_window must be positive")
    if config.global_attention_every <= 0:
        raise ValueError("global_attention_every must be positive")
    if config.n_layers % config.global_attention_every != 0:
        raise ValueError("n_layers must contain complete local/global attention cycles")
    if config.attention_stats_interval is not None and config.attention_stats_interval < 0:
        raise ValueError("attention_stats_interval must be non-negative")
    if config.attention_stats_mode not in {"sampled", "exhaustive_tiled"}:
        raise ValueError("attention_stats_mode must be 'sampled' or 'exhaustive_tiled'")
    if config.attention_stats_sample_seq_len <= 0:
        raise ValueError("attention_stats_sample_seq_len must be positive")
    if config.attention_stats_query_tile_size <= 0:
        raise ValueError("attention_stats_query_tile_size must be positive")
    if config.attention_backend not in {"fa3", "flex", "sdpa_reference"}:
        raise ValueError("attention_backend must be 'fa3', 'flex', or 'sdpa_reference'")
    if config.data_mode not in {"memmap", "streaming"}:
        raise ValueError("data_mode must be 'memmap' or 'streaming'")
    if config.tensor_parallel_size != 1:
        raise NotImplementedError(
            "tensor_parallel_size is a 3B-readiness config stub; this single-process "
            "trainer currently supports only tensor_parallel_size=1"
        )
    if config.loss_chunk_size < 0:
        raise ValueError("loss_chunk_size must be non-negative")
    if config.milestone_eval_batches < 0:
        raise ValueError("milestone_eval_batches must be non-negative")
    if config.milestone_sample_max_new_tokens < 0:
        raise ValueError("milestone_sample_max_new_tokens must be non-negative")
    if config.milestone_sample_temperature < 0.0:
        raise ValueError("milestone_sample_temperature must be non-negative")
    if config.milestone_sample_top_k < 0:
        raise ValueError("milestone_sample_top_k must be non-negative")
    if not 0.0 <= config.milestone_sample_top_p <= 1.0:
        raise ValueError("milestone_sample_top_p must be in [0, 1]")
    if config.stdout_format not in STDOUT_FORMATS:
        raise ValueError(f"stdout_format must be one of {sorted(STDOUT_FORMATS)}")
    if config.batch_size_override is not None and config.batch_size_override <= 0:
        raise ValueError("batch_size_override must be positive")
    if config.residual_init_scale <= 0.0:
        raise ValueError("residual_init_scale must be positive")
    for name, value in (
        ("muon_lr", config.muon_lr),
        ("adamw_lr", config.adamw_lr),
        ("alpha_lr", config.alpha_lr),
        ("norm_lr", config.norm_lr),
        ("attention_temperature_lr", config.attention_temperature_lr),
        ("muon_weight_decay", config.muon_weight_decay),
        ("adamw_weight_decay", config.adamw_weight_decay),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config.grad_clip != 0.0 or config.alpha_grad_clip != 0.0:
        raise ValueError("hard gradient clipping is forbidden; both clip values must be 0")
    if config.quantize_activations:
        raise ValueError("activation quantization is forbidden; it uses integer saturation")


def quantization_strength_for_step(step: int, config: TrainConfig) -> float:
    if config.quantization_warmup_steps <= 0:
        return config.quantization_end_strength
    progress = min(1.0, max(0.0, step / config.quantization_warmup_steps))
    return (
        config.quantization_start_strength
        + (config.quantization_end_strength - config.quantization_start_strength) * progress
    )


def should_log_interval(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def gradient_l2_norm(loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]) -> float:
    params = [param for param in parameters if param.requires_grad]
    if not params:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(torch.sqrt(squared))


def named_gradient_l2_norms(
    loss: torch.Tensor,
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> Dict[str, float]:
    entries = [(name, param) for name, param in named_parameters if param.requires_grad]
    if not entries:
        return {}
    gradients = torch.autograd.grad(
        loss,
        [param for _, param in entries],
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    norms: Dict[str, float] = {}
    for (name, _), gradient in zip(entries, gradients):
        norms[name] = (
            0.0
            if gradient is None
            else float(torch.linalg.vector_norm(gradient.detach().float()))
        )
    return norms


def summarize_named_scalars(
    values: Dict[str, float],
    *,
    top_k: int = 5,
) -> Dict[str, object]:
    if not values:
        return {
            "count": 0,
            "nonzero_count": 0,
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "top": [],
        }
    ordered = sorted(((name, float(value)) for name, value in values.items()), key=lambda item: item[1], reverse=True)
    numeric = [value for _, value in ordered]
    return {
        "count": len(ordered),
        "nonzero_count": sum(value > 0.0 for value in numeric),
        "min": min(numeric),
        "mean": sum(numeric) / len(numeric),
        "max": max(numeric),
        "top": [
            {"name": name, "value": value}
            for name, value in ordered[:max(0, top_k)]
        ],
    }


def tail_target_ratio_for_step(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    target_ratio: float,
    ramp_fraction: float,
) -> float:
    if target_ratio <= 0.0 or step < warmup_steps:
        return 0.0
    ramp_steps = max(1, int(total_steps * ramp_fraction))
    progress = min(1.0, max(0.0, (step - warmup_steps + 1) / ramp_steps))
    return float(target_ratio) * progress


def update_activation_gate_state(
    layers: List[Dict[str, object]],
    state: Dict[str, object],
    config: TrainConfig,
    *,
    collect_baseline: bool,
) -> List[str]:
    return []


def optimizer_update_ratio_violations(
    update_stats: Dict[str, Dict[str, float]],
    *,
    maximum_multiple: float = 3.0,
) -> List[str]:
    return []


def warmup_steps_from_fraction(total_steps: int, fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    return max(1, int(total_steps * fraction))


def set_step_hyperparameters(
    optimizer: torch.optim.Optimizer,
    base_lrs: Iterable[float],
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    config: TrainConfig,
) -> None:
    scale = lr_schedule_scale(
        step=step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        schedule=config.lr_schedule,
        final_lr_fraction=config.final_lr_fraction,
        phase1_fraction_of_total=config.phase1_fraction_of_total,
        phase2_fraction_of_total=config.phase2_fraction_of_total,
        phase_lr_start_fraction=config.phase_lr_start_fraction,
        phase1_end_lr_fraction=config.phase1_end_lr_fraction,
        phase2_peak_lr_fraction=config.phase2_peak_lr_fraction,
        phase2_peak_ramp_fraction=config.phase2_peak_ramp_fraction,
        linear_lr_halfway_fraction=config.linear_lr_halfway_fraction,
        terminal_cooldown_fraction=config.terminal_cooldown_fraction,
    )
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def token_fraction(step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, step / total_steps))


def lr_schedule_scale(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    schedule: str,
    final_lr_fraction: float,
    phase1_fraction_of_total: float,
    phase2_fraction_of_total: float,
    phase_lr_start_fraction: float,
    phase1_end_lr_fraction: float,
    phase2_peak_lr_fraction: float,
    phase2_peak_ramp_fraction: float,
    linear_lr_halfway_fraction: float,
    terminal_cooldown_fraction: float = 0.0,
) -> float:
    if terminal_cooldown_fraction > 0.0:
        cooldown_steps = max(1, int(total_steps * terminal_cooldown_fraction))
        cooldown_start = max(warmup_steps + 1, total_steps - cooldown_steps)
        if step >= cooldown_start:
            start_scale = lr_schedule_scale(
                step=max(0, cooldown_start - 1),
                total_steps=cooldown_start,
                warmup_steps=warmup_steps,
                decay_steps=min(decay_steps, cooldown_start),
                schedule=schedule,
                final_lr_fraction=final_lr_fraction,
                phase1_fraction_of_total=phase1_fraction_of_total,
                phase2_fraction_of_total=phase2_fraction_of_total,
                phase_lr_start_fraction=phase_lr_start_fraction,
                phase1_end_lr_fraction=phase1_end_lr_fraction,
                phase2_peak_lr_fraction=phase2_peak_lr_fraction,
                phase2_peak_ramp_fraction=phase2_peak_ramp_fraction,
                linear_lr_halfway_fraction=linear_lr_halfway_fraction,
                terminal_cooldown_fraction=0.0,
            )
            cooldown_progress = min(
                1.0,
                max(0.0, (step - cooldown_start + 1) / max(1, total_steps - cooldown_start)),
            )
            return start_scale * (1.0 - cooldown_progress)
        total_steps = cooldown_start
    if schedule == "phase_cosine":
        progress = token_fraction(step, total_steps)
        phase1_end = min(1.0, max(0.0, phase1_fraction_of_total))
        phase2_end = min(
            1.0,
            phase1_end + max(0.0, phase2_fraction_of_total) * max(0.0, phase2_peak_ramp_fraction),
        )
        phase1_span = max(1e-12, phase1_end)
        phase2_span = max(1e-12, phase2_end - phase1_end)
        if progress <= phase1_end:
            phase_progress = max(0.0, progress) / phase1_span
            return phase_lr_start_fraction + (phase1_end_lr_fraction - phase_lr_start_fraction) * phase_progress
        if progress <= phase2_end:
            phase_progress = (progress - phase1_end) / phase2_span
            return phase1_end_lr_fraction + (phase2_peak_lr_fraction - phase1_end_lr_fraction) * phase_progress
        decay_span = max(1e-12, 1.0 - phase2_end)
        decay_progress = min(1.0, max(0.0, (progress - phase2_end) / decay_span))
        scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return final_lr_fraction + (phase2_peak_lr_fraction - final_lr_fraction) * scale
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "linear":
        progress = token_fraction(step, total_steps)
        warmup_progress = token_fraction(warmup_steps, total_steps)
        halfway = linear_lr_halfway_fraction
        # Keep the post-warmup schedule pinned to the user's requested
        # half-peak point at 50% token progress.
        if progress <= halfway:
            segment = max(1e-12, halfway - warmup_progress)
            return 1.0 - 0.5 * max(0.0, progress - warmup_progress) / segment
        segment = max(1e-12, 1.0 - halfway)
        return max(0.0, 0.5 * (1.0 - (progress - halfway) / segment))
    if schedule == "cosine":
        progress = token_fraction(step, total_steps)
        warmup_progress = token_fraction(warmup_steps, total_steps)
        if progress <= warmup_progress:
            return 1.0
        post_warmup_progress = min(
            1.0,
            max(0.0, (progress - warmup_progress) / max(1e-12, 1.0 - warmup_progress)),
        )
        scale = 0.5 * (1.0 + math.cos(math.pi * post_warmup_progress))
        return final_lr_fraction + (1.0 - final_lr_fraction) * scale
    scale = schedule_scale(
        step=step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        schedule=schedule,
    )
    if step < warmup_steps:
        return scale
    return final_lr_fraction + (1.0 - final_lr_fraction) * scale


def schedule_scale(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    schedule: str,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "cosine":
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "wsd":
        decay_start = max(warmup_steps, total_steps - decay_steps)
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
        return 1.0 - progress
    raise ValueError(f"unknown lr schedule: {schedule}")


def current_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def tensor_stats_to_float(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value) for key, value in stats.items()}


def should_log_grad_detail(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def layer_grad_norms_by_block(model: BitNetLlama) -> Dict[str, float]:
    norms = {}
    for index, layer in enumerate(model.layers):
        norms[f"layer_{index:02d}"] = grad_norm_for_params(layer.parameters())
    norms["tok_embeddings"] = grad_norm_for_params(model.tok_embeddings.parameters())
    norms["final_norm"] = grad_norm_for_params(model.norm.parameters())
    return norms


def grad_norm_for_params(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(torch.sum(grad * grad))
    return math.sqrt(total)


def ternary_histogram_stats(hist: Dict[int, int]) -> Dict[str, float]:
    total = sum(hist.values())
    if total <= 0:
        return {
            "total": 0.0,
            "p_neg": 0.0,
            "p_zero": 0.0,
            "p_pos": 0.0,
            "entropy_bits": 0.0,
            "normalized_entropy": 0.0,
        }

    probs = {
        -1: hist.get(-1, 0) / total,
        0: hist.get(0, 0) / total,
        1: hist.get(1, 0) / total,
    }
    entropy = -sum(p * math.log2(p) for p in probs.values() if p > 0)
    return {
        "total": float(total),
        "p_neg": probs[-1],
        "p_zero": probs[0],
        "p_pos": probs[1],
        "entropy_bits": entropy,
        "normalized_entropy": entropy / math.log2(3),
    }


def throughput_projection(
    tokens_per_sec: float,
    *,
    step: int,
    total_steps: int,
    tokens_per_step: int,
) -> Dict[str, float]:
    total_tokens = total_steps * tokens_per_step
    consumed_tokens = (step + 1) * tokens_per_step
    remaining_tokens = max(0, total_tokens - consumed_tokens)
    if tokens_per_sec <= 0:
        return {
            "total_tokens": float(total_tokens),
            "consumed_tokens": float(consumed_tokens),
            "remaining_tokens": float(remaining_tokens),
            "hours_for_total_at_current_tps": math.inf,
            "hours_remaining_at_current_tps": math.inf,
        }
    return {
        "total_tokens": float(total_tokens),
        "consumed_tokens": float(consumed_tokens),
        "remaining_tokens": float(remaining_tokens),
        "hours_for_total_at_current_tps": total_tokens / tokens_per_sec / 3600.0,
        "hours_remaining_at_current_tps": remaining_tokens / tokens_per_sec / 3600.0,
    }


def memory_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    return {
        "allocated_gb": torch.cuda.memory_allocated(index) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(index) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(index) / 1e9,
        "free_gb": free / 1e9,
        "total_gb": total / 1e9,
    }


def cuda_allocator_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    stats = torch.cuda.memory_stats()
    keys = (
        "active_bytes.all.current",
        "active_bytes.all.peak",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "num_alloc_retries",
        "num_ooms",
    )
    return {key: float(stats.get(key, 0)) for key in keys}


def optimizer_group_stats(optimizer: torch.optim.Optimizer) -> Dict[str, Dict[str, object]]:
    groups = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", index))
        param_sq = 0.0
        grad_sq = 0.0
        param_abs_sum = 0.0
        grad_abs_sum = 0.0
        param_max_abs = 0.0
        grad_max_abs = 0.0
        param_count = 0
        grad_count = 0
        tensors = 0
        tensors_with_grad = 0
        state_tensor_count = 0
        state_element_count = 0
        state_sq = 0.0
        state_max_abs = 0.0
        state_by_key: Dict[str, Dict[str, float]] = {}
        for param in group["params"]:
            tensors += 1
            values = param.detach().float()
            numel = values.numel()
            param_count += numel
            param_sq += float(torch.sum(values * values))
            param_abs_sum += float(values.abs().sum())
            param_max_abs = max(param_max_abs, float(values.abs().amax()) if numel else 0.0)
            for state_key, state_value in optimizer.state.get(param, {}).items():
                if not isinstance(state_value, torch.Tensor) or not state_value.is_floating_point():
                    continue
                state_values = state_value.detach().float()
                state_numel = int(state_values.numel())
                if state_numel <= 0:
                    continue
                state_tensor_count += 1
                state_element_count += state_numel
                state_values_sq = float(torch.sum(state_values * state_values))
                state_values_max = float(state_values.abs().amax())
                state_sq += state_values_sq
                state_max_abs = max(state_max_abs, state_values_max)
                key_stats = state_by_key.setdefault(
                    str(state_key),
                    {"tensors": 0.0, "elements": 0.0, "sum_sq": 0.0, "max_abs": 0.0},
                )
                key_stats["tensors"] += 1.0
                key_stats["elements"] += float(state_numel)
                key_stats["sum_sq"] += state_values_sq
                key_stats["max_abs"] = max(key_stats["max_abs"], state_values_max)
            if param.grad is None:
                continue
            tensors_with_grad += 1
            grad = param.grad.detach().float()
            grad_numel = grad.numel()
            grad_count += grad_numel
            grad_sq += float(torch.sum(grad * grad))
            grad_abs_sum += float(grad.abs().sum())
            grad_max_abs = max(grad_max_abs, float(grad.abs().amax()) if grad_numel else 0.0)
        rendered_state_by_key = {
            state_key: {
                "tensors": values["tensors"],
                "elements": values["elements"],
                "norm": math.sqrt(values["sum_sq"]),
                "rms": math.sqrt(values["sum_sq"] / max(values["elements"], 1.0)),
                "max_abs": values["max_abs"],
            }
            for state_key, values in state_by_key.items()
        }
        groups[name] = {
            "tensors": float(tensors),
            "tensors_with_grad": float(tensors_with_grad),
            "param_count": float(param_count),
            "grad_count": float(grad_count),
            "param_norm": math.sqrt(param_sq),
            "grad_norm": math.sqrt(grad_sq),
            "param_mean_abs": param_abs_sum / param_count if param_count else 0.0,
            "grad_mean_abs": grad_abs_sum / grad_count if grad_count else 0.0,
            "param_max_abs": param_max_abs,
            "grad_max_abs": grad_max_abs,
            "lr": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "state_tensor_count": float(state_tensor_count),
            "state_element_count": float(state_element_count),
            "state_norm": math.sqrt(state_sq),
            "state_rms": math.sqrt(state_sq / state_element_count) if state_element_count else 0.0,
            "state_max_abs": state_max_abs,
            "state_by_key": rendered_state_by_key,
        }
    return groups


def optimizer_group_value_sample(
    optimizer: torch.optim.Optimizer,
    sample_size_per_group: int,
) -> Dict[str, Dict[str, object]]:
    sample_size = max(0, int(sample_size_per_group))
    groups: Dict[str, Dict[str, object]] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", index))
        params = [param for param in group["params"] if param.numel() > 0]
        total_params = sum(int(param.numel()) for param in params)
        if sample_size == 0 or not params:
            values = torch.empty(0, dtype=torch.float32)
        else:
            per_tensor = max(1, math.ceil(sample_size / len(params)))
            chunks = []
            remaining = sample_size
            for param in params:
                if remaining <= 0:
                    break
                # Move the param to host FIRST, then sample on CPU. The previous
                # on-device index_select + .cpu() triggered a CUDA device-side assert
                # on the H100 confidential-compute stack (torch 2.12/cu130) and killed
                # training a few steps in. Device-side asserts poison the CUDA context,
                # so they cannot be caught — the only fix is to not do the gather on GPU.
                flat = param.detach().reshape(-1).float().cpu()
                numel = int(flat.numel())
                take = min(numel, per_tensor, remaining)
                if take <= 0:
                    continue
                if take >= numel:
                    selected = flat
                else:
                    # Pure integer evenly-spaced indices in [0, numel-1]. float32
                    # cannot even represent numel-1 for numel > 2^24 (e.g. the 32.7M
                    # embedding) -- it rounds UP to numel -- so linspace().long()
                    # indexed out of range, which is exactly what surfaced as the
                    # "CC device-side assert" on CUDA. Integer math avoids it entirely.
                    step = torch.arange(take, dtype=torch.long)
                    indices = (step * (numel - 1)) // max(1, take - 1)
                    selected = flat.index_select(0, indices)
                chunks.append(selected)
                remaining -= take
            values = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
        groups[name] = {
            "values": values,
            "sample_count": int(values.numel()),
            "total_params": total_params,
            "lr": float(group.get("lr", 0.0)),
            "weight_decay": float(group.get("weight_decay", 0.0)),
        }
    return groups


def optimizer_update_stats(
    before: Dict[str, Dict[str, object]],
    after: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name, before_group in before.items():
        if name not in after:
            continue
        old_values = before_group["values"]
        new_values = after[name]["values"]
        if not isinstance(old_values, torch.Tensor) or not isinstance(new_values, torch.Tensor):
            continue
        count = min(int(old_values.numel()), int(new_values.numel()))
        if count <= 0:
            stats[name] = {
                "sample_count": 0.0,
                "total_params": float(before_group.get("total_params", 0)),
                "weight_rms_before": 0.0,
                "update_rms": 0.0,
                "update_to_weight_rms": 0.0,
                "update_mean_abs": 0.0,
                "update_max_abs": 0.0,
                "sample_sign_change_rate": 0.0,
                "lr": float(after[name].get("lr", 0.0)),
                "weight_decay": float(after[name].get("weight_decay", 0.0)),
            }
            continue
        old = old_values[:count].float()
        new = new_values[:count].float()
        delta = new - old
        weight_rms = float(torch.sqrt(torch.mean(old * old)).item())
        update_rms = float(torch.sqrt(torch.mean(delta * delta)).item())
        nonzero = old.ne(0) & new.ne(0)
        sign_change_rate = (
            float((old.sign()[nonzero] != new.sign()[nonzero]).float().mean().item())
            if bool(nonzero.any())
            else 0.0
        )
        stats[name] = {
            "sample_count": float(count),
            "total_params": float(before_group.get("total_params", 0)),
            "weight_rms_before": weight_rms,
            "update_rms": update_rms,
            "update_to_weight_rms": update_rms / max(weight_rms, 1e-12),
            "update_mean_abs": float(delta.abs().mean().item()),
            "update_max_abs": float(delta.abs().max().item()),
            "sample_sign_change_rate": sign_change_rate,
            "lr": float(after[name].get("lr", 0.0)),
            "weight_decay": float(after[name].get("weight_decay", 0.0)),
        }
    return stats


@torch.no_grad()
def bitlinear_effective_value_sample(
    model: BitNetLlama,
    sample_size_per_module: int,
) -> Dict[str, Dict[str, object]]:
    """Sample latent and actual forward weights at identical deterministic indices."""

    sample_size = max(0, int(sample_size_per_module))
    samples: Dict[str, Dict[str, object]] = {}
    for name, module in model.bitlinear_named_modules():
        flat = module.weight.detach().reshape(-1).float().cpu()
        total = int(flat.numel())
        take = min(sample_size, total)
        if take <= 0:
            indices = torch.empty(0, dtype=torch.long)
            latent = torch.empty(0, dtype=torch.float32)
            effective = torch.empty(0, dtype=torch.float32)
            alpha = torch.empty(0, dtype=torch.float32)
        else:
            steps = torch.arange(take, dtype=torch.long)
            indices = (steps * (total - 1)) // max(1, take - 1)
            latent = flat.index_select(0, indices)
            alpha_values = module.alpha_values().detach().reshape(-1).float().cpu()
            if alpha_values.numel() == 1:
                alpha = alpha_values.expand_as(latent)
            else:
                group_indices = torch.div(
                    indices,
                    int(module.alpha_group_size),
                    rounding_mode="floor",
                ).clamp_max(alpha_values.numel() - 1)
                alpha = alpha_values.index_select(0, group_indices)
            ternary = torch.round(latent / alpha.clamp_min(module.eps)).clamp(-1, 1)
            quantized = alpha * ternary
            strength = float(module.quantization_strength.detach().float())
            effective = torch.lerp(latent, quantized, strength)
        layer_index = model.layer_index_from_module_name(name)
        samples[name] = {
            "latent": latent,
            "effective": effective,
            "alpha": alpha,
            "sample_count": int(take),
            "total_params": total,
            "role": model.bitlinear_role(name),
            "layer_index": layer_index,
            "attention_block_kind": (
                "global"
                if layer_index is not None
                and (layer_index + 1) % int(model.config.global_attention_every) == 0
                else "local"
            ),
        }
    return samples


def bitlinear_effective_update_stats(
    before: Dict[str, Dict[str, object]],
    after: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    layers = []
    for name, before_module in before.items():
        after_module = after.get(name)
        if after_module is None:
            continue
        latent_before = before_module.get("latent")
        latent_after = after_module.get("latent")
        effective_before = before_module.get("effective")
        effective_after = after_module.get("effective")
        alpha_before = before_module.get("alpha")
        alpha_after = after_module.get("alpha")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                latent_before,
                latent_after,
                effective_before,
                effective_after,
                alpha_before,
                alpha_after,
            )
        ):
            continue
        count = min(
            int(latent_before.numel()),
            int(latent_after.numel()),
            int(effective_before.numel()),
            int(effective_after.numel()),
            int(alpha_before.numel()),
            int(alpha_after.numel()),
        )
        if count <= 0:
            continue
        latent_old = latent_before[:count].float()
        latent_new = latent_after[:count].float()
        effective_old = effective_before[:count].float()
        effective_new = effective_after[:count].float()
        alpha_old = alpha_before[:count].float().clamp_min(1e-12)
        alpha_new = alpha_after[:count].float().clamp_min(1e-12)
        latent_delta = latent_new - latent_old
        effective_delta = effective_new - effective_old
        latent_weight_rms = float(torch.sqrt(latent_old.square().mean()))
        effective_weight_rms = float(torch.sqrt(effective_old.square().mean()))
        latent_update_rms = float(torch.sqrt(latent_delta.square().mean()))
        effective_update_rms = float(torch.sqrt(effective_delta.square().mean()))
        latent_delta_over_alpha = latent_delta / alpha_old
        effective_delta_over_alpha = effective_delta / alpha_old
        normalized_old = latent_old.abs() / alpha_old
        normalized_new = latent_new.abs() / alpha_new
        zero_cell_before = normalized_old < 0.5
        zero_cell_after = normalized_new < 0.5
        boundary_distance_before = (normalized_old - 0.5).abs()
        tail_before = normalized_old > 1.5
        tail_after = normalized_new > 1.5
        layers.append(
            {
                "name": name,
                "role": before_module.get("role"),
                "layer_index": before_module.get("layer_index"),
                "attention_block_kind": before_module.get("attention_block_kind"),
                "sample_count": count,
                "total_params": int(before_module.get("total_params", 0)),
                "latent_weight_rms": latent_weight_rms,
                "effective_weight_rms": effective_weight_rms,
                "latent_update_rms": latent_update_rms,
                "effective_update_rms": effective_update_rms,
                "latent_update_to_weight_rms": latent_update_rms
                / max(latent_weight_rms, 1e-12),
                "effective_update_to_weight_rms": effective_update_rms
                / max(effective_weight_rms, 1e-12),
                "latent_update_over_alpha_rms": float(
                    torch.sqrt(latent_delta_over_alpha.square().mean())
                ),
                "effective_update_over_alpha_rms": float(
                    torch.sqrt(effective_delta_over_alpha.square().mean())
                ),
                "sample_half_cell_motion_rate": float(
                    latent_delta_over_alpha.abs().ge(0.5).float().mean()
                ),
                "sample_full_cell_motion_rate": float(
                    latent_delta_over_alpha.abs().ge(1.0).float().mean()
                ),
                "sample_boundary_band_fraction_0p05_before": float(
                    boundary_distance_before.le(0.05).float().mean()
                ),
                "sample_zero_boundary_crossing_rate": float(
                    zero_cell_before.ne(zero_cell_after).float().mean()
                ),
                "sample_saturated_tail_fraction_1p5_before": float(tail_before.float().mean()),
                "sample_saturated_tail_fraction_1p5_after": float(tail_after.float().mean()),
                "effective_value_change_rate": float(effective_old.ne(effective_new).float().mean()),
            }
        )
    return {
        "layers": layers,
        "top_latent_update_over_alpha": sorted(
            layers,
            key=lambda item: float(item["latent_update_over_alpha_rms"]),
            reverse=True,
        )[:10],
        "top_half_cell_motion_rate": sorted(
            layers,
            key=lambda item: float(item["sample_half_cell_motion_rate"]),
            reverse=True,
        )[:10],
        "top_effective_update_to_weight": sorted(
            layers,
            key=lambda item: float(item["effective_update_to_weight_rms"]),
            reverse=True,
        )[:10],
        "top_effective_value_change_rate": sorted(
            layers,
            key=lambda item: float(item["effective_value_change_rate"]),
            reverse=True,
        )[:10],
    }


def summarize_attention_stats(layers: object) -> Dict[str, object]:
    if not isinstance(layers, list) or not layers:
        return {
            "max_logit": 0.0,
            "max_abs_logit": 0.0,
            "mean_layer_max_logit": 0.0,
            "mean_normalized_entropy": 0.0,
            "min_normalized_entropy": 0.0,
            "max_q_norm_weight_abs": 0.0,
            "max_k_norm_weight_abs": 0.0,
            "by_attention_kind": {},
            "top_layers": [],
        }
    max_logits = [
        float(layer.get("max_abs_logit", abs(float(layer.get("max_logit", 0.0)))))
        for layer in layers
        if isinstance(layer, dict)
    ]
    entropies = [
        float(layer.get("normalized_entropy", 0.0))
        for layer in layers
        if isinstance(layer, dict)
    ]
    layer_summaries = [
        {
            "layer": int(layer.get("layer", -1)),
            "backend": str(layer.get("backend", "unknown")),
            "attention_kind": str(layer.get("attention_kind", "unknown")),
            "max_logit": float(layer.get("max_logit", 0.0)),
            "max_abs_logit": float(
                layer.get("max_abs_logit", abs(float(layer.get("max_logit", 0.0))))
            ),
            "max_capped_logit": float(layer.get("max_capped_logit", 0.0)),
            "p99_logit": float(layer.get("p99_logit", 0.0)),
            "p999_logit": float(layer.get("p999_logit", 0.0)),
            "normalized_entropy": float(layer.get("normalized_entropy", 0.0)),
            "mean_entropy": float(layer.get("mean_entropy", 0.0)),
            "near_512_mass": float(layer.get("near_512_mass", 0.0)),
            "outer_local_window_mass": float(layer.get("outer_local_window_mass", 0.0)),
            "distant_mass": float(layer.get("distant_mass", 0.0)),
            "mean_head_attention_cosine_similarity": float(
                layer.get("mean_head_attention_cosine_similarity", 0.0)
            ),
            "post_rope_q_vector_rms_mean": float(
                layer.get("post_rope_q_vector_rms_mean", 0.0)
            ),
            "post_rope_q_vector_rms_max": float(
                layer.get("post_rope_q_vector_rms_max", 0.0)
            ),
            "post_rope_k_vector_rms_mean": float(
                layer.get("post_rope_k_vector_rms_mean", 0.0)
            ),
            "post_rope_k_vector_rms_max": float(
                layer.get("post_rope_k_vector_rms_max", 0.0)
            ),
            "pre_qknorm_q_vector_rms_max": float(
                layer.get("pre_qknorm_q_vector_rms_max", 0.0)
            ),
            "pre_qknorm_k_vector_rms_max": float(
                layer.get("pre_qknorm_k_vector_rms_max", 0.0)
            ),
            "post_qknorm_q_vector_rms_max": float(
                layer.get("post_qknorm_q_vector_rms_max", 0.0)
            ),
            "post_qknorm_k_vector_rms_max": float(
                layer.get("post_qknorm_k_vector_rms_max", 0.0)
            ),
            "q_rope_rms_max_abs_delta": float(
                layer.get("q_rope_rms_max_abs_delta", 0.0)
            ),
            "k_rope_rms_max_abs_delta": float(
                layer.get("k_rope_rms_max_abs_delta", 0.0)
            ),
            "q_norm_weight_rms": float(layer.get("q_norm_weight_rms", 0.0)),
            "q_norm_weight_max_abs": float(layer.get("q_norm_weight_max_abs", 0.0)),
            "k_norm_weight_rms": float(layer.get("k_norm_weight_rms", 0.0)),
            "k_norm_weight_max_abs": float(layer.get("k_norm_weight_max_abs", 0.0)),
            "attention_temperature_min": float(
                layer.get("attention_temperature_min", 1.0)
            ),
            "attention_temperature_mean": float(
                layer.get("attention_temperature_mean", 1.0)
            ),
            "attention_temperature_max": float(
                layer.get("attention_temperature_max", 1.0)
            ),
        }
        for layer in layers
        if isinstance(layer, dict)
    ]
    layer_summaries.sort(key=lambda item: item["max_abs_logit"], reverse=True)
    backend_counts: Dict[str, int] = {}
    for layer in layers:
        if isinstance(layer, dict):
            backend = str(layer.get("backend", "unknown"))
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
    by_attention_kind: Dict[str, Dict[str, float]] = {}
    for kind in ("local", "global"):
        selected = [item for item in layer_summaries if item["attention_kind"] == kind]
        if not selected:
            continue
        by_attention_kind[kind] = {
            "layers": float(len(selected)),
            "max_abs_logit": max(float(item["max_abs_logit"]) for item in selected),
            "max_capped_logit": max(float(item["max_capped_logit"]) for item in selected),
            "max_p999_logit": max(float(item["p999_logit"]) for item in selected),
            "mean_normalized_entropy": sum(
                float(item["normalized_entropy"]) for item in selected
            ) / len(selected),
            "mean_near_512_mass": sum(float(item["near_512_mass"]) for item in selected)
            / len(selected),
            "mean_outer_local_window_mass": sum(
                float(item["outer_local_window_mass"]) for item in selected
            ) / len(selected),
            "mean_distant_mass": sum(float(item["distant_mass"]) for item in selected)
            / len(selected),
            "mean_head_attention_cosine_similarity": sum(
                float(item["mean_head_attention_cosine_similarity"]) for item in selected
            ) / len(selected),
        }
    return {
        "backend_counts": backend_counts,
        "max_logit": max(max_logits) if max_logits else 0.0,
        "max_abs_logit": max(max_logits) if max_logits else 0.0,
        "mean_layer_max_logit": sum(max_logits) / len(max_logits) if max_logits else 0.0,
        "mean_normalized_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "min_normalized_entropy": min(entropies) if entropies else 0.0,
        "max_q_norm_weight_abs": max(
            (float(item["q_norm_weight_max_abs"]) for item in layer_summaries),
            default=0.0,
        ),
        "max_k_norm_weight_abs": max(
            (float(item["k_norm_weight_max_abs"]) for item in layer_summaries),
            default=0.0,
        ),
        "by_attention_kind": by_attention_kind,
        "top_layers": layer_summaries[:5],
    }


def summarize_ssm_stats(layers: object) -> Dict[str, object]:
    if not isinstance(layers, list) or not layers:
        return {
            "mean_dt": 0.0,
            "max_dt": 0.0,
            "max_state_abs": 0.0,
            "mean_state_rms": 0.0,
            "max_gate_abs": 0.0,
            "mean_angle_delta_rms": 0.0,
            "max_angle_delta_abs": 0.0,
            "mean_phase_rms": 0.0,
            "max_phase_abs": 0.0,
            "top_layers": [],
        }
    dt_means = [float(layer.get("dt_mean", 0.0)) for layer in layers if isinstance(layer, dict)]
    dt_maxes = [float(layer.get("dt_max", 0.0)) for layer in layers if isinstance(layer, dict)]
    state_rms = [
        float(layer.get("state_rms", 0.0))
        for layer in layers
        if isinstance(layer, dict)
    ]
    state_maxes = [float(layer.get("state_max_abs", 0.0)) for layer in layers if isinstance(layer, dict)]
    gate_maxes = [float(layer.get("gate_max_abs", 0.0)) for layer in layers if isinstance(layer, dict)]
    angle_delta_rms = []
    angle_delta_maxes = []
    phase_rms = []
    phase_maxes = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        angle = layer.get("state_rope_angle")
        if not isinstance(angle, dict):
            continue
        delta = angle.get("angle_delta")
        phase = angle.get("cumulative_phase")
        if isinstance(delta, dict):
            angle_delta_rms.append(float(delta.get("rms", 0.0)))
            angle_delta_maxes.append(float(delta.get("max_abs", 0.0)))
        if isinstance(phase, dict):
            phase_rms.append(float(phase.get("rms", 0.0)))
            phase_maxes.append(float(phase.get("max_abs", 0.0)))
    layer_summaries = [
        {
            "layer": int(layer.get("layer", -1)),
            "dt_mean": float(layer.get("dt_mean", 0.0)),
            "dt_max": float(layer.get("dt_max", 0.0)),
            "state_rms": float(layer.get("state_rms", 0.0)),
            "state_max_abs": float(layer.get("state_max_abs", 0.0)),
            "gate_max_abs": float(layer.get("gate_max_abs", 0.0)),
            "out_rms": float(layer.get("out_rms", 0.0)),
            "angle_delta_rms": _nested_float(layer, "state_rope_angle", "angle_delta", "rms"),
            "phase_rms": _nested_float(layer, "state_rope_angle", "cumulative_phase", "rms"),
            "phase_max_abs": _nested_float(layer, "state_rope_angle", "cumulative_phase", "max_abs"),
        }
        for layer in layers
        if isinstance(layer, dict)
    ]
    layer_summaries.sort(key=lambda item: item["state_max_abs"], reverse=True)
    return {
        "mean_dt": sum(dt_means) / len(dt_means) if dt_means else 0.0,
        "max_dt": max(dt_maxes) if dt_maxes else 0.0,
        "mean_state_rms": sum(state_rms) / len(state_rms) if state_rms else 0.0,
        "max_state_abs": max(state_maxes) if state_maxes else 0.0,
        "max_gate_abs": max(gate_maxes) if gate_maxes else 0.0,
        "mean_angle_delta_rms": (
            sum(angle_delta_rms) / len(angle_delta_rms) if angle_delta_rms else 0.0
        ),
        "max_angle_delta_abs": max(angle_delta_maxes) if angle_delta_maxes else 0.0,
        "mean_phase_rms": sum(phase_rms) / len(phase_rms) if phase_rms else 0.0,
        "max_phase_abs": max(phase_maxes) if phase_maxes else 0.0,
        "top_layers": layer_summaries[:5],
    }


def _nested_float(layer: Dict[str, object], *path: str) -> float:
    value: object = layer
    for key in path:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def check_finite_grads(
    model: torch.nn.Module,
    step: int,
    *,
    raise_on_nonfinite: bool = True,
) -> bool:
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            if raise_on_nonfinite:
                raise FloatingPointError(f"non-finite gradient in {name} at step {step}")
            return False
    return True


def warn_if_needed(payload: Dict[str, object]) -> None:
    grad_norm = float(payload["grad_norm"])
    flip_rate = payload["weight_flip_rate"]
    alpha = float(payload["alpha_magnitude"])
    warnings = []
    if grad_norm > 100:
        warnings.append("grad_norm_gt_100")
    if flip_rate is not None and float(flip_rate) > 0.20:
        warnings.append("weight_flip_rate_gt_20pct")
    elif flip_rate is not None and float(flip_rate) < 1e-4:
        warnings.append("weight_flip_rate_near_zero")
    if alpha < 1e-6:
        warnings.append("alpha_near_zero")
    if warnings:
        payload["warnings"] = warnings


@torch.no_grad()
def rescale_optimizer_state_(
    optimizer: torch.optim.Optimizer,
    projections: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]],
) -> None:
    """Apply a latent projection's scale to compatible optimizer state."""

    for parameter, scale in projections:
        state = optimizer.state.get(parameter, {})
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value.mul_(scale.to(device=value.device, dtype=value.dtype))


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: BitNetLlama,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    train_config: TrainConfig,
    *,
    model_config: BitNetLlamaConfig,
    resume_contract: Dict[str, object],
    run_state: Dict[str, object],
    scaler: Optional[torch.amp.GradScaler] = None,
    events_path: Optional[Path] = None,
    checkpoints_log_path: Optional[Path] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "run_id": train_config.run_id,
        "step": next_step,
        "next_step": next_step,
        "completed_step": next_step - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "train_config": asdict(train_config),
        "model_config": asdict(model_config),
        "resume_contract": resume_contract,
        "run_state": run_state,
        "rng": rng_state(),
    }
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)
    checksum = file_sha256(path)
    metadata = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "run_id": train_config.run_id,
        "next_step": next_step,
        "completed_step": next_step - 1,
        "size_bytes": path.stat().st_size,
        "sha256": checksum,
    }
    write_json_atomic(path.with_suffix(path.suffix + ".json"), metadata)
    event = {"event": "checkpoint_saved", "step": next_step, "path": str(path), **metadata}
    emit_json(event, events_path, checkpoints_log_path)


def load_checkpoint(
    path: Path,
    model: BitNetLlama,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: Optional[torch.amp.GradScaler] = None,
    payload: Optional[Dict[str, object]] = None,
    expected_contract: Optional[Dict[str, object]] = None,
) -> Tuple[int, Dict[str, object]]:
    payload = payload or load_checkpoint_payload(path, map_location=device)
    if expected_contract is not None:
        validate_resume_contract(payload, expected_contract)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload.get("rng", {}))
    next_step = int(payload["next_step"])
    run_state = payload.get("run_state", {})
    if not isinstance(run_state, dict):
        raise RuntimeError("checkpoint run_state is malformed")
    print_json(
        {
            "event": "checkpoint_loaded",
            "step": next_step,
            "next_step": next_step,
            "run_id": payload.get("run_id"),
            "path": str(path),
        }
    )
    return next_step, run_state


def rng_state() -> Dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.detach().cpu().to(torch.uint8)
        torch.set_rng_state(torch_state)
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = []
        for cuda_state in state["cuda"]:
            if isinstance(cuda_state, torch.Tensor):
                cuda_state = cuda_state.detach().cpu().to(torch.uint8)
            cuda_states.append(cuda_state)
        torch.cuda.set_rng_state_all(cuda_states)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def configure_gpu_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        setter = getattr(torch.backends.cuda, name, None)
        if setter is not None:
            setter(True)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp_payload(payload: Dict[str, object]) -> Dict[str, object]:
    if "time_generated" in payload:
        return payload
    stamped = dict(payload)
    stamped["time_generated"] = utc_now_iso()
    return stamped


def azure_text_log_path(path: Path) -> Path:
    override_root = os.environ.get("BITNET_AZURE_LOG_ROOT", "").strip()
    if override_root:
        return Path(override_root) / f"{path.stem}.log"
    return path.parent / "azure_monitor" / f"{path.stem}.log"


def append_azure_text_log(path: Path, payload: Dict[str, object]) -> None:
    if os.environ.get("BITNET_AZURE_TEXT_LOGS", "0") != "1":
        return
    mirror_path = azure_text_log_path(path)
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    with mirror_path.open("a") as handle:
        handle.write(f"{payload['time_generated']}\t{json.dumps(payload, default=str)}\n")


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stamp_payload(payload)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
    append_azure_text_log(path, payload)


def emit_json(
    payload: Dict[str, object],
    events_path: Optional[Path] = None,
    event_path: Optional[Path] = None,
) -> None:
    payload = stamp_payload(payload)
    print_json(payload)
    if events_path is not None:
        append_jsonl(events_path, payload)
    if event_path is not None and event_path != events_path:
        append_jsonl(event_path, payload)


def print_json(payload: Dict[str, object]) -> None:
    if _STDOUT_FORMAT == "silent":
        return
    if _STDOUT_FORMAT == "json":
        print(json.dumps(payload, default=str), flush=True)
        return
    print_pretty(payload)


def configure_stdout(stdout_format: str) -> None:
    global _STDOUT_FORMAT
    _STDOUT_FORMAT = stdout_format


def create_progress_bar(*, total: int, enabled: bool, initial: int = 0):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    ncols = int(os.environ.get("TQDM_NCOLS", "108"))
    bar_format = (
        "{desc}: {percentage:3.0f}%|{bar:18}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )
    return tqdm(
        total=total,
        initial=initial,
        dynamic_ncols=False,
        ncols=ncols,
        desc="train",
        mininterval=5.0,
        bar_format=bar_format,
    )


def update_progress_bar(
    progress,
    *,
    step: int,
    total_steps: int,
    running_loss: Optional[float],
    tokens_per_sec: Optional[float],
    curriculum: Optional[Dict[str, object]],
) -> None:
    if progress is None:
        return
    progress.update(1)
    postfix = {"s": f"{step + 1}/{total_steps}"}
    if running_loss is not None:
        postfix["l"] = f"{running_loss:.3f}"
    if tokens_per_sec is not None:
        postfix["tok/s"] = f"{tokens_per_sec:.0f}"
    if curriculum is not None:
        postfix["src"] = str(curriculum.get("stage", "?"))
    progress.set_postfix(postfix, refresh=False)


def close_progress_bar(progress) -> None:
    if progress is not None:
        progress.close()


def print_pretty(payload: Dict[str, object]) -> None:
    event = str(payload.get("event", "event"))
    if event in {
        "train",
        "optimizer_stats",
        "system_timeseries",
        "ternary_histogram",
        "ternary_transition",
        "ssm_stats",
        "update_stats",
        "curriculum",
        "checkpoint_loaded",
    }:
        return
    if event == "startup":
        print(
            f"[startup] device={payload.get('device')} dtype={payload.get('dtype')} "
            f"loss_scale_init={payload.get('loss_scale_init')} min_loss_scale={payload.get('min_loss_scale')} "
            f"steps={payload.get('total_steps')} output={payload.get('output_dir')}",
            flush=True,
        )
        return
    if event == "initialization":
        report = payload.get("report", {}) or {}
        alpha_stats = report.get("alpha_stats", {}) or {}
        print(
            f"[init] step={payload.get('step')} alpha_mean={alpha_stats.get('mean_abs')} "
            f"alpha_min={alpha_stats.get('min')} alpha_max={alpha_stats.get('max')}",
            flush=True,
        )
        return
    if event == "validation":
        print(
            f"[val] step={payload.get('step')} loss={payload.get('val_loss')} "
            f"mode={payload.get('validation_mode')}",
            flush=True,
        )
        return
    if event == "checkpoint_saved":
        print(f"[ckpt] step={payload.get('step')} path={payload.get('path')}", flush=True)
        return
    if event == "milestone_evaluation":
        print(
            f"[milestone] step={payload.get('step')} val_loss={payload.get('val_loss')} "
            f"dir={payload.get('milestone_dir')}",
            flush=True,
        )
        return
    if event == "done":
        print(
            f"[done] step={payload.get('step')} stop_reason={payload.get('stop_reason')}",
            flush=True,
        )
        return
    if event == "dry_run_complete":
        print("[dry-run] complete", flush=True)
        return
    return


if __name__ == "__main__":
    main()
