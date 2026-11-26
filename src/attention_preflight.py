from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from kernels import (
    KernelBackendError,
    assert_backend_usable,
    clear_flex_block_mask_cache,
    flash_attention3,
    flex_attention_forward,
    flex_block_mask_cache_size,
    flex_block_mask_stats,
    sdpa_reference_attention,
)


def comma_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def attention_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def tensor_error(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    relative_rms_floor: float = 1e-12,
) -> Dict[str, float]:
    """Report an absolute error and a stable relative-RMS error.

    Gradients of a mean-reduced attention probe can legitimately have an RMS
    below 1e-6.  Dividing an otherwise negligible FP16 rounding error by that
    near-zero signal creates a false parity failure.  The absolute gate still
    applies in all cases; a non-trivial RMS floor only makes the relative gate
    meaningful in that numerical-noise regime.
    """
    if relative_rms_floor <= 0.0:
        raise ValueError("relative_rms_floor must be positive")
    delta = actual.float() - reference.float()
    reference_rms = torch.sqrt(reference.float().square().mean())
    normalizer = reference_rms.clamp_min(relative_rms_floor)
    return {
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(torch.sqrt(delta.square().mean()).item()),
        "reference_rms": float(reference_rms.item()),
        "relative_rms_floor": float(relative_rms_floor),
        "relative_l2": float(torch.sqrt(delta.square().mean()).div(normalizer).item()),
    }


def assert_error(
    label: str,
    error: Dict[str, float],
    *,
    max_abs: float,
    max_relative_l2: float,
) -> None:
    if error["max_abs"] > max_abs or error["relative_l2"] > max_relative_l2:
        raise AssertionError(
            f"{label} parity failed: {error}; limits: "
            f"max_abs={max_abs}, relative_l2={max_relative_l2}"
        )


def make_qkv(
    seq_len: int,
    *,
    batch: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    seed: int,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape_q = (batch, q_heads, seq_len, head_dim)
    shape_kv = (batch, kv_heads, seq_len, head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(shape_kv, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(shape_kv, device="cuda", dtype=dtype, generator=generator)
    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def preprocess_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    qk_norm: bool,
    rope_theta: float = 10_000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror Terminus Flagship 30B's optional head RMSNorm followed by RoPE."""

    if qk_norm:
        q = (
            q.float() * torch.rsqrt(q.float().square().mean(dim=-1, keepdim=True) + 1e-5)
        ).to(q.dtype)
        k = (
            k.float() * torch.rsqrt(k.float().square().mean(dim=-1, keepdim=True) + 1e-5)
        ).to(k.dtype)
    half = q.shape[-1] // 2
    positions = torch.arange(q.shape[2], device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(half, device=q.device, dtype=torch.float32) / float(half))
    )
    angles = torch.outer(positions, inv_freq)[None, None, :, :]
    cos = angles.cos()
    sin = angles.sin()

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        left, right = x_float[..., :half], x_float[..., half:]
        return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


def backend_output(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    local_window: Optional[int],
    softcap: float = 0.0,
) -> torch.Tensor:
    q_flash = q.transpose(1, 2).contiguous()
    k_flash = k.transpose(1, 2).contiguous()
    v_flash = v.transpose(1, 2).contiguous()
    assert_backend_usable(backend, q=q_flash, k=k_flash)
    if backend == "fa3":
        window_size = (-1, -1) if local_window is None else (local_window - 1, 0)
        return flash_attention3(
            q_flash,
            k_flash,
            v_flash,
            dropout_p=0.0,
            causal=True,
            window_size=window_size,
            softcap=softcap,
        ).transpose(1, 2)
    if backend == "flex":
        return flex_attention_forward(q, k, v, local_window=local_window, softcap=softcap)
    raise ValueError(f"unsupported production backend: {backend}")


@lru_cache(maxsize=8)
def _compiled_flex_runner(
    local_window: Optional[int],
    softcap: float,
):
    torch.set_float32_matmul_precision("high")

    def runner(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flex_attention_forward(q, k, v, local_window=local_window, softcap=softcap)

    return torch.compile(runner)


def boundary_positions(seq_len: int, window: int) -> List[int]:
    candidates = {0, 1, window - 2, window - 1, window, seq_len - 1}
    return sorted(position for position in candidates if 0 <= position < seq_len)


def forward_parity(args: argparse.Namespace, backend: str, seq_len: int) -> Dict[str, object]:
    q, k, v = make_qkv(
        seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        seed=args.seed + seq_len,
        dtype=attention_dtype(args.dtype),
    )
    patterns = {"global": None, "local": args.window}
    result: Dict[str, object] = {"backend": backend, "seq_len": seq_len, "patterns": {}}
    for pattern, local_window in patterns.items():
        prepared_q, prepared_k = preprocess_qk(q, k, qk_norm=True)
        prepared_q = prepared_q * float(args.temperature)
        reference = sdpa_reference_attention(
            prepared_q,
            prepared_k,
            v,
            local_window=local_window,
            softcap=args.softcap,
        )
        actual = backend_output(
            backend,
            prepared_q,
            prepared_k,
            v,
            local_window=local_window,
            softcap=args.softcap,
        )
        whole = tensor_error(actual, reference)
        assert_error(
            f"{backend}/{pattern}/L{seq_len}",
            whole,
            max_abs=args.forward_max_abs,
            max_relative_l2=args.forward_max_relative_l2,
        )
        pattern_result: Dict[str, object] = {"whole_tensor": whole}
        if pattern == "local":
            boundary: Dict[str, Dict[str, float]] = {}
            for position in boundary_positions(seq_len, args.window):
                error = tensor_error(actual[:, :, position, :], reference[:, :, position, :])
                assert_error(
                    f"{backend}/local/L{seq_len}/position{position}",
                    error,
                    max_abs=args.boundary_max_abs,
                    max_relative_l2=args.boundary_max_relative_l2,
                )
                boundary[str(position)] = error
            pattern_result["boundary_positions"] = boundary
        result["patterns"][pattern] = pattern_result
    return result


def gradient_parity(args: argparse.Namespace, backend: str) -> Dict[str, object]:
    seq_len = args.gradient_length
    source = make_qkv(
        seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        seed=args.seed + 10_000 + seq_len,
        dtype=attention_dtype(args.dtype),
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 20_000 + seq_len)
    probe = torch.randn(
        (args.batch, args.q_heads, seq_len, args.head_dim),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    results: Dict[str, object] = {"backend": backend, "seq_len": seq_len, "patterns": {}}
    for pattern, local_window in (("global", None), ("local", args.window)):
        prepared_q, prepared_k = preprocess_qk(
            source[0], source[1], qk_norm=True
        )
        prepared_q = prepared_q * float(args.temperature)
        prepared_source = (prepared_q, prepared_k, source[2])
        reference_inputs = tuple(t.detach().clone().requires_grad_(True) for t in prepared_source)
        actual_inputs = tuple(t.detach().clone().requires_grad_(True) for t in prepared_source)
        reference = sdpa_reference_attention(
            *reference_inputs,
            local_window=local_window,
            softcap=args.softcap,
        )
        reference_loss = (reference.float() * probe).mean()
        reference_grads = torch.autograd.grad(reference_loss, reference_inputs)
        actual = backend_output(
            backend,
            *actual_inputs,
            local_window=local_window,
            softcap=args.softcap,
        )
        actual_loss = (actual.float() * probe).mean()
        actual_grads = torch.autograd.grad(actual_loss, actual_inputs)
        errors = {
            name: tensor_error(
                actual_grad,
                reference_grad,
                relative_rms_floor=args.gradient_relative_rms_floor,
            )
            for name, actual_grad, reference_grad in zip(
                ("q", "k", "v"), actual_grads, reference_grads
            )
        }
        for name, error in errors.items():
            assert_error(
                f"{backend}/{pattern}/gradient-{name}",
                error,
                max_abs=args.gradient_max_abs,
                max_relative_l2=args.gradient_max_relative_l2,
            )
        results["patterns"][pattern] = errors
    return results


def median_kernel_measurement(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: int,
    softcap: float,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    if backend == "flex":
        clear_flex_block_mask_cache()
        runner = _compiled_flex_runner(window, float(softcap))

        def invoke() -> torch.Tensor:
            return runner(q, k, v)
    else:
        def invoke() -> torch.Tensor:
            return backend_output(backend, q, k, v, local_window=window)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cold_baseline = torch.cuda.memory_allocated()
    cold_start = time.perf_counter()
    output = invoke()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1_000.0
    cold_peak_delta = max(0, torch.cuda.max_memory_allocated() - cold_baseline)
    persistent_after_cold = max(0, torch.cuda.memory_allocated() - cold_baseline)
    del output
    for _ in range(warmup):
        output = invoke()
        del output
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = invoke()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    peak_delta = max(0, torch.cuda.max_memory_allocated() - baseline)
    del output
    return {
        "median_ms": statistics.median(times) * 1_000.0,
        "peak_workspace_bytes": float(peak_delta),
        "cold_ms": cold_ms,
        "cold_peak_bytes": float(cold_peak_delta),
        "persistent_bytes_after_cold": float(persistent_after_cold),
        "flex_block_mask_cache_entries": float(flex_block_mask_cache_size()),
    }


def scaling_gate(args: argparse.Namespace, backend: str) -> Dict[str, object]:
    lengths = comma_ints(args.scaling_lengths)
    if len(lengths) < 2:
        raise ValueError("--scaling-lengths needs at least two lengths")
    rows = []
    for seq_len in lengths:
        q, k, v = make_qkv(
            seq_len,
            batch=args.scaling_batch,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            seed=args.seed + 30_000 + seq_len,
            dtype=attention_dtype(args.dtype),
        )
        row: Dict[str, object] = {
            "seq_len": seq_len,
            **median_kernel_measurement(
                backend,
                q,
                k,
                v,
                window=args.window,
                softcap=args.softcap,
                warmup=args.scaling_warmup,
                repeats=args.scaling_repeats,
            ),
        }
        if backend == "flex":
            row["block_mask"] = flex_block_mask_stats(
                batch=args.scaling_batch,
                heads=args.q_heads,
                seq_len=seq_len,
                device=q.device,
                local_window=args.window,
            )
        rows.append(row)
        del q, k, v
        torch.cuda.empty_cache()

    pairwise = []
    for lower, upper in zip(rows, rows[1:]):
        length_ratio = upper["seq_len"] / lower["seq_len"]
        time_ratio = upper["median_ms"] / max(lower["median_ms"], 1e-12)
        memory_ratio = upper["peak_workspace_bytes"] / max(lower["peak_workspace_bytes"], 1.0)
        time_exponent = math.log(max(time_ratio, 1e-12)) / math.log(length_ratio)
        memory_exponent = math.log(max(memory_ratio, 1e-12)) / math.log(length_ratio)
        pair = {
            "lower_length": lower["seq_len"],
            "upper_length": upper["seq_len"],
            "time_ratio": time_ratio,
            "memory_ratio": memory_ratio,
            "time_exponent": time_exponent,
            "memory_exponent": memory_exponent,
        }
        if time_exponent > args.max_scaling_exponent:
            raise AssertionError(f"{backend} local time scaling looks dense: {pair}")
        if memory_exponent > args.max_scaling_exponent:
            raise AssertionError(f"{backend} local memory scaling looks dense: {pair}")
        pairwise.append(pair)

    if backend == "flex":
        largest = rows[-1]["block_mask"]
        minimum_sparsity = 100.0 * max(0.0, 1.0 - (args.window + 256) / rows[-1]["seq_len"])
        if largest["sparsity_percent"] < minimum_sparsity:
            raise AssertionError(
                f"FlexAttention BlockMask is unexpectedly dense: {largest}; "
                f"minimum_sparsity_percent={minimum_sparsity:.2f}"
            )
        max_expected_blocks = math.ceil(args.window / 128) + 2
        if largest["max_active_kv_blocks_per_row"] > max_expected_blocks:
            raise AssertionError(
                f"FlexAttention local BlockMask activates too many blocks: {largest}; "
                f"maximum_expected={max_expected_blocks}"
            )
    return {"backend": backend, "rows": rows, "pairwise_scaling": pairwise}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard-gate FA3/Flex correctness and true local sparsity before Terminus Flagship 30B training"
    )
    parser.add_argument("--backends", default="fa3,flex")
    parser.add_argument("--lengths", default="1,1023,1024,1025,4096")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--softcap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--gradient-length", type=int, default=1025)
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--run-scaling", action="store_true")
    parser.add_argument("--scaling-lengths", default="2048,4096")
    parser.add_argument("--scaling-batch", type=int, default=1)
    parser.add_argument("--scaling-warmup", type=int, default=2)
    parser.add_argument("--scaling-repeats", type=int, default=5)
    parser.add_argument("--max-scaling-exponent", type=float, default=1.65)
    parser.add_argument("--forward-max-abs", type=float, default=0.10)
    parser.add_argument("--forward-max-relative-l2", type=float, default=0.04)
    parser.add_argument("--boundary-max-abs", type=float, default=0.10)
    parser.add_argument("--boundary-max-relative-l2", type=float, default=0.05)
    parser.add_argument("--gradient-max-abs", type=float, default=0.02)
    parser.add_argument("--gradient-max-relative-l2", type=float, default=0.08)
    parser.add_argument(
        "--gradient-relative-rms-floor",
        type=float,
        default=1e-5,
        help="Absolute RMS normalizer floor for gradient relative-error reporting.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run this preflight on the target CUDA machine")
    major, minor = torch.cuda.get_device_capability()
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.window <= 0:
        raise ValueError("window must be positive")
    if args.gradient_relative_rms_floor <= 0.0:
        raise ValueError("gradient-relative-rms-floor must be positive")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if args.softcap < 0.0:
        raise ValueError("softcap must be non-negative")
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    if not backends or any(item not in {"fa3", "flex"} for item in backends):
        raise ValueError("--backends must contain fa3 and/or flex")
    if "fa3" in backends and major < 9:
        raise SystemExit(
            f"FA3 preflight requires Hopper capability >= 9.0, got {major}.{minor}; "
            "use --backends flex on non-Hopper GPUs"
        )

    report: Dict[str, object] = {
        "status": "running",
        "device": torch.cuda.get_device_name(),
        "capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "gqa": {"q_heads": args.q_heads, "kv_heads": args.kv_heads, "head_dim": args.head_dim},
        "window": args.window,
        "temperature": args.temperature,
        "softcap": args.softcap,
        "qk_preprocessing": "all local/global layers RMSNorm(Q/K) -> RoPE",
        "forward_parity": [],
        "gradient_parity": [],
        "scaling": [],
    }
    try:
        for backend in backends:
            for seq_len in comma_ints(args.lengths):
                report["forward_parity"].append(forward_parity(args, backend, seq_len))
            if not args.skip_gradients:
                report["gradient_parity"].append(gradient_parity(args, backend))
            if args.run_scaling:
                report["scaling"].append(scaling_gate(args, backend))
        report["status"] = "pass"
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    try:
        main(parse_args())
    except KernelBackendError as exc:
        raise SystemExit(f"attention backend preflight failed: {exc}") from exc
