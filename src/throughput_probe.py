from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from model import BitNetLlama, BitNetLlamaConfig
from optim import OptimizerConfig, build_optimizer
from kernels import flash_attention3_available, flex_attention_available


MODE_CONFIGS: Dict[str, Dict[str, float | str]] = {"transformerpp": {}}


def comma_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def comma_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_azure_text_log(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"{payload['time_generated']}\t{json.dumps(payload, sort_keys=True)}\n")


def configure_runtime_caches() -> None:
    runtime_root = Path(
        os.environ.get(
            "BITNET_RUNTIME_CACHE_ROOT",
            Path(__file__).resolve().parent / ".runtime_cache",
        )
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = runtime_root / "tmp"
    triton_dir = runtime_root / "triton"
    inductor_dir = runtime_root / "torchinductor"
    for path in (tmp_dir, triton_dir, inductor_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))


def build_model_config(args: argparse.Namespace, mode: str) -> BitNetLlamaConfig:
    _ = MODE_CONFIGS[mode]
    return BitNetLlamaConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.seq_len,
        loss_chunk_size=args.loss_chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
        alpha_mode=args.alpha_mode,
        transformer_alpha_mode=args.transformer_alpha_mode,
        alpha_group_size=args.alpha_group_size,
        attention_backend=args.attention_backend,
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
    )


def prewarm_runtime_backends(cfg: BitNetLlamaConfig) -> None:
    # Resolve optional backend/module availability outside torch.compile so
    # Dynamo does not trace one-time filesystem and extension-loader work.
    if cfg.attention_backend == "fa3":
        flash_attention3_available()
    elif cfg.attention_backend == "flex":
        flex_attention_available()


def one_result(args: argparse.Namespace, *, mode: str, batch_size: int) -> Dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    cfg = build_model_config(args, mode)
    model = BitNetLlama(cfg).cuda().train()
    prewarm_runtime_backends(cfg)
    optimizer = (
        None
        if args.optimizer == "none"
        else build_optimizer(model, OptimizerConfig(kind=args.optimizer))
    )
    if args.compile:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
    forward_model = torch.compile(model, mode=args.compile_mode) if args.compile else model
    token_stream = torch.randint(
        1,
        cfg.vocab_size,
        (batch_size, args.seq_len + 1),
        device="cuda",
    )
    input_ids = token_stream[:, :-1]
    targets = token_stream[:, 1:]

    use_amp = not args.disable_cuda_amp

    def step() -> float:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = forward_model(input_ids, targets=targets, return_logits=False)
        loss = out["loss"]
        if not args.forward_only:
            loss.backward()
            if optimizer is not None:
                optimizer.step()
        return float(loss.detach().cpu())

    losses = []
    for _ in range(args.warmup_steps):
        losses.append(step())
    torch.cuda.synchronize()

    step_times = []
    for _ in range(args.steps):
        start = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)
    elapsed = sum(step_times)
    median_step_s = statistics.median(step_times)

    tokens = batch_size * args.seq_len * args.steps
    result: Dict[str, object] = {
        "event": "throughput_probe_result",
        "time_generated": utc_now_iso(),
        "mode": mode,
        "batch_size": batch_size,
        "seq_len": args.seq_len,
        "tokens_per_step": batch_size * args.seq_len,
        "steps": args.steps,
        "elapsed_s": elapsed,
        "step_s": elapsed / args.steps,
        "tokens_per_s": tokens / elapsed,
        "median_step_s": median_step_s,
        "median_tokens_per_s": (batch_size * args.seq_len) / median_step_s,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "loss": losses[-1],
        "compile": bool(args.compile),
        "compile_mode": args.compile_mode if args.compile else "eager",
        "gradient_checkpointing": bool(cfg.gradient_checkpointing),
        "optimizer": args.optimizer,
        "forward_only": bool(args.forward_only),
        "disable_cuda_amp": bool(args.disable_cuda_amp),
        "loss_chunk_size": cfg.loss_chunk_size,
        "alpha_mode": cfg.alpha_mode,
        "transformer_alpha_mode": cfg.transformer_alpha_mode,
        "alpha_group_size": cfg.alpha_group_size,
        "attention_backend": cfg.attention_backend,
        "qk_norm_mode": cfg.qk_norm_mode,
        "attention_temperature_mode": cfg.attention_temperature_mode,
        "attention_temperature_min": cfg.attention_temperature_min,
        "attention_temperature_max": cfg.attention_temperature_max,
        "attention_temperature_init": cfg.attention_temperature_init,
        "attention_softcap": cfg.attention_softcap,
        "ffn_activation": cfg.ffn_activation,
        "use_embedding_projection": cfg.use_embedding_projection,
        "transformer_layer_indices": cfg.transformer_layer_indices,
        "local_attention_window": cfg.local_attention_window,
        "global_attention_every": cfg.global_attention_every,
        "parameter_count": model.count_parameters(),
    }
    del forward_model, optimizer, model, input_ids, targets, token_stream
    torch.cuda.empty_cache()
    gc.collect()
    return result


def run(args: argparse.Namespace) -> None:
    configure_runtime_caches()
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for throughput_probe.py")
    modes = comma_strings(args.modes)
    bad_modes = [mode for mode in modes if mode not in MODE_CONFIGS]
    if bad_modes:
        raise SystemExit(f"unknown modes: {bad_modes}; choices={sorted(MODE_CONFIGS)}")

    results = []
    for mode in modes:
        for batch_size in comma_ints(args.batch_sizes):
            try:
                result = one_result(args, mode=mode, batch_size=batch_size)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                torch.cuda.empty_cache()
                gc.collect()
                result = {
                    "event": "throughput_probe_result",
                    "time_generated": utc_now_iso(),
                    "mode": mode,
                    "batch_size": batch_size,
                    "seq_len": args.seq_len,
                    "oom": "out of memory" in str(exc).lower(),
                    "error": str(exc).splitlines()[0],
                }
            if args.run_label:
                result["run_label"] = args.run_label
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.jsonl_output:
                append_jsonl(Path(args.jsonl_output), result)
            if args.azure_text_output:
                append_azure_text_log(Path(args.azure_text_output), result)
            results.append(result)

    ok = [
        item
        for item in results
        if isinstance(item.get("median_tokens_per_s", item.get("tokens_per_s")), (int, float))
    ]
    if ok:
        print("summary")
        for item in sorted(ok, key=lambda row: (str(row["mode"]), int(row["batch_size"]))):
            print(
                f"{item['mode']:10s} B{int(item['batch_size']):<3d} "
                f"{float(item.get('median_tokens_per_s', item['tokens_per_s'])):9.0f} tok/s "
                f"{float(item.get('median_step_s', item['step_s'])):.4f}s/step "
                f"{float(item['peak_gb']):.2f}GB"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-step throughput probe for the Terminus Flagship 30B Transformer++ block.")
    parser.add_argument("--modes", default="transformerpp")
    parser.add_argument("--batch-sizes", default="14")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-cuda-amp", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "muon", "none"], default="muon")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--loss-chunk-size", type=int, default=2048)
    alpha_modes = [
        "deterministic_absmean",
        "deterministic_grouped_absmean",
        "deterministic_grouped_lloyd",
        "fixed_init_absmean",
        "learned_log_gain",
        "learned_grouped_log_gain",
    ]
    parser.add_argument("--alpha-mode", choices=alpha_modes, default="deterministic_grouped_absmean")
    parser.add_argument("--transformer-alpha-mode", choices=alpha_modes, default="deterministic_grouped_absmean")
    parser.add_argument("--alpha-group-size", type=int, default=128)
    parser.add_argument(
        "--attention-backend",
        choices=["fa3", "flex", "sdpa_reference"],
        default="fa3",
    )
    parser.add_argument("--transformer-layer-indices", default="all")
    parser.add_argument("--local-attention-window", type=int, default=1024)
    parser.add_argument("--global-attention-every", type=int, default=5)
    parser.add_argument("--qk-norm-mode", choices=["unit_rms"], default="unit_rms")
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
    parser.add_argument("--jsonl-output")
    parser.add_argument("--azure-text-output")
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
