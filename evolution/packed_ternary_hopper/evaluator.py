"""Correctness-first OpenEvolve evaluator for the H100 packed ternary kernel."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from packed_ternary import PackedTernaryWeight, pack_ternary_weight

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # Allows local static checks before OpenEvolve is installed.
    EvaluationResult = None


GROUP_SIZE = 128
DEEPGEMM_VERSION = "2.6.1"
ALLOWED_IMPORT_ROOTS = {"__future__", "torch", "triton", "math"}
FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "__import__"}
FORBIDDEN_QUALIFIED_CALLS = {
    "torch.addmm",
    "torch.bmm",
    "torch.compile",
    "torch.load",
    "torch.matmul",
    "torch.mm",
    "torch.save",
    "torch.nn.functional.linear",
}

MODEL_DIM = int(os.environ.get("TERMINUS_EVOLVE_DIM", "768"))
FFN_DIM = int(os.environ.get("TERMINUS_EVOLVE_FFN_DIM", "2048"))
N_HEADS = int(os.environ.get("TERMINUS_EVOLVE_N_HEADS", "12"))
N_KV_HEADS = int(os.environ.get("TERMINUS_EVOLVE_N_KV_HEADS", "4"))
if min(MODEL_DIM, FFN_DIM, N_HEADS, N_KV_HEADS) <= 0:
    raise ValueError("OpenEvolve target dimensions must be positive")
if MODEL_DIM % N_HEADS or N_HEADS % N_KV_HEADS:
    raise ValueError("OpenEvolve target head dimensions are inconsistent")
KV_DIM = N_KV_HEADS * (MODEL_DIM // N_HEADS)

# Defaults are the completed 150M flagship shapes.  Environment overrides make
# the search target the exact 2.6B architecture once that architecture is
# frozen; training-sized M cases prevent a decode-only winner from being chosen.
BENCHMARK_CASES: Tuple[Tuple[str, int, int, int], ...] = (
    ("q_decode", 1, MODEL_DIM, MODEL_DIM),
    ("kv_decode", 1, KV_DIM, MODEL_DIM),
    ("ffn_up_decode", 1, FFN_DIM, MODEL_DIM),
    ("ffn_down_decode", 1, MODEL_DIM, FFN_DIM),
    ("q_prefill_128", 128, MODEL_DIM, MODEL_DIM),
    ("ffn_up_prefill_128", 128, FFN_DIM, MODEL_DIM),
    ("q_train_512", 512, MODEL_DIM, MODEL_DIM),
    ("kv_train_512", 512, KV_DIM, MODEL_DIM),
    ("ffn_up_train_2048", 2048, FFN_DIM, MODEL_DIM),
    ("ffn_down_train_2048", 2048, MODEL_DIM, FFN_DIM),
)

QUICK_CORRECTNESS_CASES: Tuple[Tuple[int, int, int], ...] = (
    (1, 33, 260),
    (17, 64, 128),
    (5, 127, 769),
)


def _evaluation_result(metrics: Dict[str, object], artifacts: Dict[str, object]):
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    return {**metrics, "artifacts": artifacts}


def _failure(error_type: str, message: str, detail: str = ""):
    return _evaluation_result(
        {
            "compile_success": 0.0,
            "correctness": 0.0,
            "combined_score": 0.0,
        },
        {
            "error_type": error_type,
            "error": message,
            "detail": detail[-8000:],
        },
    )


def _qualified_name(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _validate_candidate_source(program_path: Path) -> None:
    source = program_path.read_text()
    tree = ast.parse(source, filename=str(program_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
            if not roots <= ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"forbidden imports: {sorted(roots - ALLOWED_IMPORT_ROOTS)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"forbidden call: {node.func.id}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            qualified = _qualified_name(node.func)
            if qualified in FORBIDDEN_QUALIFIED_CALLS:
                raise ValueError(f"forbidden call: {qualified}")


def _load_candidate(program_path: Path) -> ModuleType:
    _validate_candidate_source(program_path)
    module_name = f"packed_ternary_candidate_{abs(hash(program_path.read_bytes()))}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import candidate {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "packed_ternary_matmul", None)):
        raise AttributeError("candidate must define packed_ternary_matmul")
    return module


def _require_h100() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires an NVIDIA H100 CUDA device")
    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(
            f"evaluation is pinned to Hopper compute capability 9.x, got {major}.{minor}"
        )


def _case_tensors(
    *,
    m: int,
    n: int,
    k: int,
    seed: int,
) -> Tuple[torch.Tensor, PackedTernaryWeight, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn((n, k), generator=generator, dtype=torch.float32) / math.sqrt(k)
    activations = torch.randn((m, k), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    packed_cpu = pack_ternary_weight(
        latent,
        group_size=GROUP_SIZE,
        scale_dtype=torch.float32,
    )
    reference = F.linear(activations.float(), packed_cpu.dequantize()).to(torch.bfloat16)
    packed_cuda = PackedTernaryWeight(
        codes=packed_cpu.codes.cuda(),
        scales=packed_cpu.scales.cuda(),
        in_features=packed_cpu.in_features,
        out_features=packed_cpu.out_features,
        group_size=packed_cpu.group_size,
    )
    return activations.cuda(), packed_cuda, reference.cuda()


def _run_candidate(
    module: ModuleType,
    activations: torch.Tensor,
    packed: PackedTernaryWeight,
) -> torch.Tensor:
    output = module.packed_ternary_matmul(
        activations,
        packed.codes,
        packed.scales,
        packed.group_size,
        packed.in_features,
    )
    torch.cuda.synchronize()
    return output


def _correctness_suite(
    module: ModuleType,
    cases: Iterable[Tuple[int, int, int]],
) -> Tuple[float, float]:
    worst_absolute = 0.0
    worst_relative = 0.0
    for index, (m, n, k) in enumerate(cases):
        activations, packed, reference = _case_tensors(m=m, n=n, k=k, seed=1100 + index)
        before = activations.clone()
        output = _run_candidate(module, activations, packed)
        if output.shape != reference.shape:
            raise AssertionError(f"wrong output shape {tuple(output.shape)}; expected {tuple(reference.shape)}")
        if output.dtype != torch.bfloat16:
            raise AssertionError(f"output must be BF16, got {output.dtype}")
        if not torch.isfinite(output).all():
            raise AssertionError("output contains NaN or Inf")
        if not torch.equal(activations, before):
            raise AssertionError("candidate modified its activation input")
        absolute = (output.float() - reference.float()).abs()
        relative = absolute / reference.float().abs().clamp_min(1e-3)
        worst_absolute = max(worst_absolute, float(absolute.max()))
        worst_relative = max(worst_relative, float(relative.max()))
        torch.testing.assert_close(output, reference, rtol=0.03, atol=0.03)
    return worst_absolute, worst_relative


def _measure_us(callable_, repeats: int, samples: int = 7) -> float:
    for _ in range(5):
        callable_()
    torch.cuda.synchronize()
    timings: List[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            callable_()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) * 1000.0 / repeats)
    return statistics.median(timings)


def _optional_deepgemm_backend():
    try:
        backend = importlib.import_module("deep_gemm")
    except ImportError:
        return None
    if getattr(backend, "__version__", "") != DEEPGEMM_VERSION:
        return None
    return backend if callable(getattr(backend, "bf16_gemm_nt", None)) else None


def _benchmark(module: ModuleType) -> Tuple[List[Dict[str, object]], float, float]:
    results: List[Dict[str, object]] = []
    speedups: List[float] = []
    candidate_latencies: List[float] = []
    deepgemm = _optional_deepgemm_backend()
    for index, (name, m, n, k) in enumerate(BENCHMARK_CASES):
        activations, packed, reference = _case_tensors(m=m, n=n, k=k, seed=2200 + index)
        dense_weight = packed.dequantize(dtype=torch.bfloat16)
        candidate_call = lambda: module.packed_ternary_matmul(
            activations,
            packed.codes,
            packed.scales,
            packed.group_size,
            packed.in_features,
        )
        dense_call = lambda: F.linear(activations, dense_weight)
        deepgemm_output = torch.empty((m, n), device=activations.device, dtype=torch.bfloat16)
        deepgemm_call = (
            None
            if deepgemm is None
            else lambda: deepgemm.bf16_gemm_nt(activations, dense_weight, deepgemm_output)
        )
        # Compile and correctness-check this exact production shape before timing.
        output = _run_candidate(module, activations, packed)
        torch.testing.assert_close(output, reference, rtol=0.03, atol=0.03)
        repeats = 100 if m == 1 else (20 if m <= 128 else 5)
        candidate_us = _measure_us(candidate_call, repeats)
        dense_us = _measure_us(dense_call, repeats)
        deepgemm_us = None
        deepgemm_error = None
        if deepgemm_call is not None:
            try:
                deepgemm_call()
                torch.cuda.synchronize()
                torch.testing.assert_close(deepgemm_output, reference, rtol=0.03, atol=0.03)
                deepgemm_us = _measure_us(deepgemm_call, repeats)
            except Exception as error:
                deepgemm_error = f"{type(error).__name__}: {error}"
        dense_candidates = [("torch_bf16", dense_us)]
        if deepgemm_us is not None:
            dense_candidates.append(("deepgemm_bf16", deepgemm_us))
        baseline_name, baseline_us = min(dense_candidates, key=lambda item: item[1])
        speedup = baseline_us / candidate_us
        candidate_latencies.append(candidate_us)
        speedups.append(speedup)
        results.append(
            {
                "name": name,
                "candidate_us": candidate_us,
                "torch_dense_bf16_us": dense_us,
                "deepgemm_dense_bf16_us": deepgemm_us,
                "deepgemm_error": deepgemm_error,
                "best_dense_baseline": baseline_name,
                "best_dense_bf16_us": baseline_us,
                "speedup_vs_best_dense": speedup,
            }
        )
    geometric_speedup = math.exp(sum(math.log(max(value, 1e-9)) for value in speedups) / len(speedups))
    geometric_latency = math.exp(
        sum(math.log(max(value, 1e-9)) for value in candidate_latencies) / len(candidate_latencies)
    )
    return results, geometric_speedup, geometric_latency


def evaluate_stage1(program_path: str):
    try:
        _require_h100()
        module = _load_candidate(Path(program_path).resolve())
        worst_abs, worst_rel = _correctness_suite(module, QUICK_CORRECTNESS_CASES)
        return _evaluation_result(
            {
                "compile_success": 1.0,
                "correctness": 1.0,
                "combined_score": 1.0,
                "worst_abs_error": worst_abs,
                "worst_relative_error": worst_rel,
            },
            {},
        )
    except Exception as error:
        return _failure(type(error).__name__, str(error), traceback.format_exc())


def evaluate_stage2(program_path: str):
    try:
        _require_h100()
        module = _load_candidate(Path(program_path).resolve())
        worst_abs, worst_rel = _correctness_suite(module, QUICK_CORRECTNESS_CASES)
        cases, geometric_speedup, geometric_latency = _benchmark(module)
        return _evaluation_result(
            {
                "compile_success": 1.0,
                "correctness": 1.0,
                "combined_score": geometric_speedup,
                "geometric_speedup_vs_dense": geometric_speedup,
                "geometric_speedup_vs_best_dense": geometric_speedup,
                "geometric_latency_us": geometric_latency,
                "worst_abs_error": worst_abs,
                "worst_relative_error": worst_rel,
            },
            {
                "benchmark_cases": json.dumps(cases, indent=2),
                "target_shape": json.dumps(
                    {
                        "dim": MODEL_DIM,
                        "ffn_dim": FFN_DIM,
                        "n_heads": N_HEADS,
                        "n_kv_heads": N_KV_HEADS,
                    },
                    sort_keys=True,
                ),
            },
        )
    except Exception as error:
        return _failure(type(error).__name__, str(error), traceback.format_exc())


def evaluate(program_path: str):
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    result = evaluate(str(EXPERIMENT_DIR / "initial_program.py"))
    if EvaluationResult is not None and isinstance(result, EvaluationResult):
        result = {"metrics": result.metrics, "artifacts": result.artifacts}
    print(json.dumps(result, indent=2))
