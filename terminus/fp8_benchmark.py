from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import validate_finite_state
from .fp8_validation import projection_shapes
from .integrity import fingerprint, source_hashes, write_json_atomic
from .model import BitLinear
from .preflight import validate_preflight_report
from .train import load_config, runtime_contract


VARIANTS = ("torch_bf16", "torch_fp8_reference", "deepgemm_bf16_backward", "deepgemm_dgrad", "deepgemm_full")


def variant_options(name, config):
    if name not in VARIANTS:
        raise ValueError(f"unknown benchmark variant: {name}")
    options = {"weight_scale_layout": "blockwise", "quantize_activations": name != "torch_bf16"}
    if name.startswith("deepgemm"):
        modes = {"deepgemm_bf16_backward": "bf16", "deepgemm_dgrad": "dgrad", "deepgemm_full": "full"}
        options.update(linear_backend="deepgemm", fp8_backward=modes[name],
                       fp8_quantizer=config.fp8_quantizer, fp8_cache_weights=config.fp8_cache_weights)
    return options


def benchmark_variant(config, training, *, variant, rows, outputs, width, warmup, steps, accumulation, optimizer_config, profile_path=None):
    torch.manual_seed(training.seed)
    layer = BitLinear(width, outputs, **variant_options(variant, config)).cuda()
    inputs = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gradient = (torch.randn(rows, outputs, device="cuda", dtype=torch.bfloat16) /
                (math.sqrt(rows * outputs) * accumulation))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=optimizer_config.adamw_lr, betas=optimizer_config.betas,
                                  weight_decay=optimizer_config.weight_decay, eps=optimizer_config.eps, fused=True)

    def update():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            inputs.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = layer(inputs)
            output.backward(gradient)
        norm = torch.nn.utils.clip_grad_norm_(layer.parameters(), training.grad_clip or float("inf"), error_if_nonfinite=True, foreach=True)
        optimizer.step()
        layer.clear_fp8_weight_cache()
        return norm

    durations = []
    for step in range(warmup + steps):
        torch.cuda.synchronize()
        if step == warmup:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        norm = update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if step >= warmup:
            durations.append(elapsed)
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    validate_finite_state((layer.state_dict(), optimizer.state_dict()))
    if profile_path is not None:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profiler:
            update()
            torch.cuda.synchronize()
        Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(profile_path))
    return {"variant": variant, "rows": rows, "outputs": outputs, "width": width,
            "accumulation": accumulation, "step_seconds": durations, "median_step_seconds": statistics.median(durations),
            "token_rows_per_second": rows * accumulation * steps / sum(durations),
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "last_gradient_norm": float(norm), "profile": str(profile_path) if profile_path is not None else None}


def run_benchmark(config_path, preflight_report, *, rows=None, warmup=2, steps=5, accumulation=None,
                  variants=VARIANTS, profile_dir=None):
    config, training, optimizer_config = load_config(config_path)
    if optimizer_config.kind != "adamw":
        raise ValueError("this matched projection benchmark measures AdamW updates only")
    rows = config.max_seq_len if rows is None else rows
    accumulation = training.gradient_accumulation_steps if accumulation is None else accumulation
    if rows <= 0 or warmup < 1 or steps < 1 or accumulation < 1 or not variants or set(variants) - set(VARIANTS):
        raise ValueError("invalid benchmark dimensions, variants, warmup or steps")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("this comparison requires the actual Hopper GPU; CPU emulation is not a speed benchmark")
    if config.linear_backend != "deepgemm" or config.fp8_backward != "full":
        raise ValueError("use the full FP8 configuration/preflight to exercise all benchmark variants")
    validate_preflight_report(preflight_report, config, torch.device("cuda", 0))
    results = []
    for role, (outputs, width) in projection_shapes(config).items():
        role_results = []
        for variant in variants:
            torch.cuda.empty_cache()
            profile_path = Path(profile_dir) / f"{role}_{variant}.json" if profile_dir else None
            result = benchmark_variant(config, training, variant=variant, rows=rows, outputs=outputs, width=width,
                                       warmup=warmup, steps=steps, accumulation=accumulation,
                                       optimizer_config=optimizer_config, profile_path=profile_path)
            role_results.append({"role": role, **result})
        baseline = next((item for item in role_results if item["variant"] == "torch_bf16"), None)
        if baseline:
            for item in role_results:
                speedup = baseline["median_step_seconds"] / item["median_step_seconds"]
                item["speedup_vs_matched_bf16"] = speedup
                item["fixed_work_time_reduction_fraction"] = 1 - 1 / speedup
        results.extend(role_results)
    return {"model": asdict(config), "model_sha256": fingerprint(asdict(config)), "source": source_hashes(),
            "runtime": runtime_contract(torch.device("cuda", 0)), "warmup_steps": warmup, "measured_steps": steps,
            "optimizer": asdict(optimizer_config), "grad_clip": training.grad_clip,
            "results": results,
            "scope": "Matched blockwise-ternary projection microbenchmarks with forward, both gradients, clipping, AdamW and accumulation. Weight cache is invalidated every update; timing includes its refill. Excludes full-network attention, output head, data, communication and checkpoints. No language-model quality or end-to-end cost guarantee."}


def main():
    parser = argparse.ArgumentParser(description="Measure matched BF16/FP8 projection optimizer updates on Hopper")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--profile-dir")
    args = parser.parse_args()
    result = run_benchmark(args.config, args.preflight_report, rows=args.rows, warmup=args.warmup,
                           steps=args.steps, accumulation=args.accumulation, variants=tuple(args.variants.split(",")),
                           profile_dir=args.profile_dir)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
