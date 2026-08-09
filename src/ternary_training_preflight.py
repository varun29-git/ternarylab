#!/usr/bin/env python3
"""Numerical and timing gate for packed BitLinear training on one H100.

This is a kernel qualification, not a training run.  It checks every projection
role at the requested model width, includes pack/cast overhead in timings, and
refuses FP8 unless the caller explicitly opts into the qualification path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from deepgemm_integration import deepgemm_bf16_wgrad, deepgemm_fp8_wgrad
from model import BitLinear
from packed_ternary import pack_ternary_weight
from ternary_training import require_hopper
from ternary_training_kernels import (
    pack_grouped_absmean_triton,
    packed_ternary_dgrad,
    packed_ternary_fprop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--ffn-dim", type=int, required=True)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--n-kv-heads", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--correctness-rows", type=int, default=128)
    parser.add_argument("--timing-rows", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--allow-fp8-wgrad", action="store_true")
    parser.add_argument(
        "--wgrad-backend",
        choices=["torch_bf16", "deepgemm_bf16", "deepgemm_fp8"],
        required=True,
    )
    parser.add_argument("--fp8-wgrad-roles", default="ffn_gate,ffn_up,ffn_down")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-rmse", type=float, default=0.02)
    parser.add_argument("--json-output", required=True)
    return parser.parse_args()


def projection_shapes(args: argparse.Namespace) -> Iterable[Tuple[str, int, int]]:
    head_dim = args.dim // args.n_heads
    kv_dim = args.n_kv_heads * head_dim
    return (
        ("q", args.dim, args.dim),
        ("k", kv_dim, args.dim),
        ("v", kv_dim, args.dim),
        ("o", args.dim, args.dim),
        ("ffn_gate", args.ffn_dim, args.dim),
        ("ffn_up", args.ffn_dim, args.dim),
        ("ffn_down", args.dim, args.ffn_dim),
    )


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    actual_float = actual.float()
    reference_float = reference.float()
    error = actual_float - reference_float
    reference_rms = torch.sqrt(reference_float.square().mean()).clamp_min(1.0e-12)
    relative_rmse = torch.sqrt(error.square().mean()) / reference_rms
    flat_actual = actual_float.reshape(-1)
    flat_reference = reference_float.reshape(-1)
    cosine = F.cosine_similarity(flat_actual, flat_reference, dim=0)
    return {
        "max_abs": float(error.abs().max()),
        "mean_abs": float(error.abs().mean()),
        "relative_rmse": float(relative_rmse),
        "cosine": float(cosine),
    }


def benchmark_ms(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


def qualify_shape(
    role: str,
    out_features: int,
    in_features: int,
    args: argparse.Namespace,
    fp8_roles: set[str],
) -> Dict[str, object]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(17 + out_features + in_features)
    weight = torch.randn(
        (out_features, in_features),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ) / math.sqrt(in_features)
    activations = torch.randn(
        (args.correctness_rows, in_features),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    grad_output = torch.randn(
        (args.correctness_rows, out_features),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    reference_packed = pack_ternary_weight(weight, group_size=args.group_size)
    codes, scales = pack_grouped_absmean_triton(weight, group_size=args.group_size)
    if not torch.equal(codes, reference_packed.codes):
        raise RuntimeError(f"{role}: Triton pack codes do not match the reference")
    if not torch.allclose(scales, reference_packed.scales, rtol=1.0e-6, atol=1.0e-8):
        scale_error = float((scales - reference_packed.scales).abs().max())
        raise RuntimeError(f"{role}: Triton scale mismatch, max_abs={scale_error}")

    effective = reference_packed.dequantize(torch.bfloat16)
    fprop_reference = F.linear(activations, effective)
    fprop_actual = packed_ternary_fprop(
        activations,
        codes,
        scales,
        in_features=in_features,
        group_size=args.group_size,
    )
    dgrad_reference = torch.mm(grad_output, effective)
    dgrad_actual = packed_ternary_dgrad(
        grad_output,
        codes,
        scales,
        in_features=in_features,
        group_size=args.group_size,
    )
    fprop_error = error_metrics(fprop_actual, fprop_reference)
    dgrad_error = error_metrics(dgrad_actual, dgrad_reference)

    wgrad_reference = torch.mm(grad_output.float().transpose(0, 1), activations.float())
    wgrad_bf16 = (
        torch.mm(
            grad_output.transpose(0, 1).contiguous(),
            activations,
            out_dtype=torch.float32,
        )
        if args.wgrad_backend == "torch_bf16"
        else deepgemm_bf16_wgrad(grad_output, activations)
    )
    wgrad_bf16_error = error_metrics(wgrad_bf16, wgrad_reference)
    fp8_error = None
    if args.allow_fp8_wgrad and role in fp8_roles:
        fp8_error = error_metrics(deepgemm_fp8_wgrad(grad_output, activations), wgrad_reference)

    for name, metrics in (
        ("fprop", fprop_error),
        ("dgrad", dgrad_error),
        ("bf16_wgrad", wgrad_bf16_error),
        ("fp8_wgrad", fp8_error),
    ):
        if metrics is None:
            continue
        if metrics["cosine"] < args.min_cosine or metrics["relative_rmse"] > args.max_relative_rmse:
            raise RuntimeError(f"{role} {name} failed numerical gate: {metrics}")

    timing_activations = torch.randn(
        (args.timing_rows, in_features),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    timing_grad = torch.randn(
        (args.timing_rows, out_features),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    def packed_forward_end_to_end():
        local_codes, local_scales = pack_grouped_absmean_triton(weight, group_size=args.group_size)
        return packed_ternary_fprop(
            timing_activations,
            local_codes,
            local_scales,
            in_features=in_features,
            group_size=args.group_size,
        )

    pack_codes = torch.empty_like(codes)
    pack_scales = torch.empty_like(scales)

    def production_pack_in_place():
        return pack_grouped_absmean_triton(
            weight,
            group_size=args.group_size,
            codes_out=pack_codes,
            scales_out=pack_scales,
        )

    timings = {
        "pack_once_per_optimizer_update_ms": benchmark_ms(
            production_pack_in_place,
            args.warmup,
            args.iterations,
        ),
        "packed_fprop_cached_ms": benchmark_ms(
            lambda: packed_ternary_fprop(
                timing_activations,
                codes,
                scales,
                in_features=in_features,
                group_size=args.group_size,
            ),
            args.warmup,
            args.iterations,
        ),
        "packed_fprop_including_pack_ms": benchmark_ms(
            packed_forward_end_to_end, args.warmup, args.iterations
        ),
        "packed_dgrad_ms": benchmark_ms(
            lambda: packed_ternary_dgrad(
                timing_grad,
                codes,
                scales,
                in_features=in_features,
                group_size=args.group_size,
            ),
            args.warmup,
            args.iterations,
        ),
        f"{args.wgrad_backend}_safe_wgrad_ms": benchmark_ms(
            (
                lambda: torch.mm(
                    timing_grad.transpose(0, 1).contiguous(),
                    timing_activations,
                    out_dtype=torch.float32,
                )
            )
            if args.wgrad_backend == "torch_bf16"
            else lambda: deepgemm_bf16_wgrad(timing_grad, timing_activations),
            args.warmup,
            args.iterations,
        ),
    }
    if args.allow_fp8_wgrad and role in fp8_roles:
        timings["deepgemm_fp8_wgrad_including_cast_ms"] = benchmark_ms(
            lambda: deepgemm_fp8_wgrad(timing_grad, timing_activations),
            args.warmup,
            args.iterations,
        )

    return {
        "role": role,
        "shape": {"m_correctness": args.correctness_rows, "n": out_features, "k": in_features},
        "packing": "exact",
        "errors": {
            "fprop": fprop_error,
            "dgrad": dgrad_error,
            "bf16_wgrad": wgrad_bf16_error,
            "fp8_wgrad": fp8_error,
        },
        "timings": timings,
    }


def qualify_compiled_autograd(args: argparse.Namespace, fp8_roles: set[str]) -> Dict[str, object]:
    """Require the actual cached custom-autograd boundary to survive AOTAutograd."""

    role = "ffn_up"
    backend = args.wgrad_backend
    role_string = ",".join(sorted(fp8_roles)) if backend == "deepgemm_fp8" else ""
    layer = BitLinear(
        128,
        128,
        alpha_mode="deterministic_grouped_absmean",
        alpha_group_size=128,
        quantization_strength=1.0,
        quantize_activations=False,
        kernel_backend="triton_packed",
        wgrad_backend=backend,
        fp8_wgrad_roles=role_string,
        kernel_role=role,
    ).cuda()
    cache = layer.refresh_packed_training_cache_(force=True)
    compiled = torch.compile(layer, fullgraph=True)
    x = torch.randn(
        (128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    upstream = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    output = compiled(x)
    (output * upstream).sum().backward()
    torch.cuda.synchronize()
    if x.grad is None or layer.weight.grad is None:
        raise RuntimeError("compiled packed BitLinear did not produce both required gradients")
    return {
        "status": "passed",
        "fullgraph": True,
        "aot_autograd_backward": True,
        "wgrad_backend": backend,
        "cache": cache,
    }


def main() -> None:
    args = parse_args()
    if args.dim % args.n_heads:
        raise SystemExit("dim must be divisible by n_heads")
    require_hopper(torch.device("cuda"))
    if args.wgrad_backend == "deepgemm_fp8" and not args.allow_fp8_wgrad:
        raise SystemExit("deepgemm_fp8 preflight requires --allow-fp8-wgrad")
    torch.manual_seed(17)
    fp8_roles = {role.strip() for role in args.fp8_wgrad_roles.split(",") if role.strip()}
    allowed_roles = {shape[0] for shape in projection_shapes(args)}
    if fp8_roles - allowed_roles:
        raise SystemExit(f"unknown FP8 roles: {sorted(fp8_roles - allowed_roles)}")
    if fp8_roles and not args.allow_fp8_wgrad:
        # A default recommendation is not authorization to execute FP8.
        fp8_roles = set()

    started = time.time()
    records = [qualify_shape(*shape, args, fp8_roles) for shape in projection_shapes(args)]
    compiled_autograd = qualify_compiled_autograd(args, fp8_roles)
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "model_shape": {
            "dim": args.dim,
            "ffn_dim": args.ffn_dim,
            "n_heads": args.n_heads,
            "n_kv_heads": args.n_kv_heads,
        },
        "fp8_wgrad_executed": bool(args.allow_fp8_wgrad),
        "fp8_wgrad_roles": sorted(fp8_roles),
        "compiled_autograd": compiled_autograd,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_relative_rmse": args.max_relative_rmse,
        },
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
