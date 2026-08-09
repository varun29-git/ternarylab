"""H100 system evaluator for the complete packed-BitLinear forward/backward path.

Candidates are data-only unified patches, never executable evaluator code.  A
patch is applied to an ephemeral repository copy, then the actual numerical
preflight and a compiled model training-step probe are run there.  The source
baseline, archive, reports, and evaluator stay untouched.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # Local static checks do not need OpenEvolve installed.
    EvaluationResult = None


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
BASELINE_ROOT = Path(os.environ.get("TERMINUS_SYSTEM_BASELINE", REPO_ROOT)).resolve()

# Only the fixed-datatype BitLinear compute substrate is evolvable.  The
# throughput benchmark, evaluator, architecture, attention semantics, archive,
# reports, data path, and cloud tooling are deliberately outside this list.
MUTABLE_PATHS = frozenset(
    {
        "src/ternary_training_kernels.py",
        "src/ternary_training.py",
        "src/deepgemm_integration.py",
        "src/packed_ternary.py",
        "src/model.py",  # BitLinear wiring only; the evaluator catches semantic regressions.
        "src/kernels.py",
    }
)

FORBIDDEN_PATCH_TOKENS = re.compile(
    r"(?i)\b(aws_access_key_id|aws_secret_access_key|azure_client_secret|"
    r"openai_api_key|anthropic_api_key|api[_-]?key|secret(?:[_-]?key)?|password|"
    r"credential|session[_-]?token|sas[_-]?token|authorization|bearer|"
    r"metadata\.google|169\.254\.169\.254|"
    r"subprocess|os\.system|socket|requests|urllib|httpx|curl|wget|eval\(|exec\(|open\()"
)
REDACTION = re.compile(
    r"(?i)((?:api[_-]?key|secret|password|token|authorization)\s*[=:]\s*)[^\s,;]+"
)

DIM = int(os.environ.get("TERMINUS_EVOLVE_DIM", "768"))
FFN_DIM = int(os.environ.get("TERMINUS_EVOLVE_FFN_DIM", "2048"))
N_HEADS = int(os.environ.get("TERMINUS_EVOLVE_N_HEADS", "12"))
N_KV_HEADS = int(os.environ.get("TERMINUS_EVOLVE_N_KV_HEADS", "4"))
SEQ_LEN = int(os.environ.get("TERMINUS_EVOLVE_SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("TERMINUS_EVOLVE_BATCH", "4"))


def _result(metrics: Dict[str, object], artifacts: Dict[str, object]):
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    return {**metrics, "artifacts": artifacts}


def _fail(kind: str, detail: str):
    return _result(
        {
            "patch_valid": 0.0,
            "static_contracts": 0.0,
            "numerical_correctness": 0.0,
            "compiled_training_step": 0.0,
            "combined_score": 0.0,
        },
        {"failure": kind, "detail": _redact(detail)[-8000:]},
    )


def _redact(text: str) -> str:
    return REDACTION.sub(r"\1[REDACTED]", str(text))


def _extract_patch(program_path: Path) -> str:
    tree = ast.parse(program_path.read_text(), filename=str(program_path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "SYSTEM_PATCH"
    ]
    if len(assignments) != 1:
        raise ValueError("candidate must contain exactly one SYSTEM_PATCH assignment")
    value = ast.literal_eval(assignments[0].value)
    if not isinstance(value, str):
        raise ValueError("SYSTEM_PATCH must be a literal string")
    if len(value) > 180_000:
        raise ValueError("SYSTEM_PATCH exceeds the 180k character candidate limit")
    return value.strip()


def _changed_paths(patch: str) -> Tuple[str, ...]:
    paths = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[6:])
        elif line.startswith("+++ /dev/null"):
            raise ValueError("new files are not allowed; place new kernels in the approved modules")
    if patch and not paths:
        raise ValueError("candidate patch contains no file changes")
    if len(set(paths)) != len(paths):
        paths = list(dict.fromkeys(paths))
    invalid = sorted(set(paths) - MUTABLE_PATHS)
    if invalid:
        raise ValueError(f"candidate changes non-BitLinear or protected paths: {invalid}")
    return tuple(paths)


def _validate_patch(patch: str) -> Tuple[str, ...]:
    if not patch or "EVOLVE-BLOCK" in patch:
        return ()
    if not patch.startswith("diff --git "):
        raise ValueError("SYSTEM_PATCH must be a unified git diff")
    paths = _changed_paths(patch)
    additions = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    forbidden = FORBIDDEN_PATCH_TOKENS.search(additions)
    if forbidden:
        raise ValueError(f"candidate contains prohibited secret/external-execution token: {forbidden.group(0)!r}")
    return paths


def _safe_env(source_root: Path, home: Path) -> Dict[str, str]:
    """Do not pass cloud/API credentials or the caller's home into candidate code."""

    home.mkdir(parents=True, exist_ok=True)
    allowed = {
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    }
    env = {key: os.environ[key] for key in allowed if os.environ.get(key)}
    env.update(
        {
            "HOME": str(home),
            "PYTHONPATH": str(source_root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "NO_PROXY": "*",
        }
    )
    return env


def _run(command: Iterable[str], *, cwd: Path, env: Dict[str, str], timeout: int) -> str:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = _redact(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {output[-5000:]}")
    return output


def _prepare_trial(patch: str) -> Tuple[tempfile.TemporaryDirectory, Path, Dict[str, str]]:
    temporary = tempfile.TemporaryDirectory(prefix="terminus-bitlinear-evolve-")
    trial = Path(temporary.name) / "trial"
    shutil.copytree(
        BASELINE_ROOT,
        trial,
        ignore=shutil.ignore_patterns(".git", "archive", "working", "provenance", "__pycache__", ".runtime_cache"),
    )
    env = _safe_env(trial, Path(temporary.name) / "home")
    if patch and "EVOLVE-BLOCK" not in patch:
        patch_path = Path(temporary.name) / "candidate.patch"
        patch_path.write_text(patch)
        _run(["git", "apply", "--check", "--unsafe-paths", str(patch_path)], cwd=trial, env=env, timeout=30)
        _run(["git", "apply", "--unsafe-paths", str(patch_path)], cwd=trial, env=env, timeout=30)
    return temporary, trial, env


def _require_h100() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the system evaluator requires CUDA on the target Azure H100")
    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(f"the evaluator requires Hopper SM90, got {major}.{minor}")


def _probe_result(output: str) -> Dict[str, object]:
    records = []
    for line in output.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("event") == "throughput_probe_result":
            records.append(candidate)
    if len(records) != 1 or "median_tokens_per_s" not in records[0]:
        raise RuntimeError("throughput probe did not emit exactly one successful result")
    return records[0]


def evaluate_stage1(program_path: str):
    try:
        patch = _extract_patch(Path(program_path))
        paths = _validate_patch(patch)
        temporary, trial, env = _prepare_trial(patch)
        try:
            contracts = _run(
                [sys.executable, "-m", "unittest", "discover", "-s", "src/tests", "-p", "test_*.py"],
                cwd=trial,
                env=env,
                timeout=180,
            )
        finally:
            temporary.cleanup()
        return _result(
            {"patch_valid": 1.0, "static_contracts": 1.0, "combined_score": 1.0},
            {"changed_paths": json.dumps(paths), "static_contracts_tail": contracts[-3000:]},
        )
    except Exception as exc:
        return _fail(type(exc).__name__, str(exc))


def evaluate_stage2(program_path: str):
    started = time.monotonic()
    try:
        _require_h100()
        patch = _extract_patch(Path(program_path))
        paths = _validate_patch(patch)
        temporary, trial, env = _prepare_trial(patch)
        try:
            contracts = _run(
                [sys.executable, "-m", "unittest", "discover", "-s", "src/tests", "-p", "test_*.py"],
                cwd=trial,
                env=env,
                timeout=180,
            )
            preflight = _run(
                [
                    sys.executable,
                    "src/ternary_training_preflight.py",
                    "--dim", str(DIM), "--ffn-dim", str(FFN_DIM),
                    "--n-heads", str(N_HEADS), "--n-kv-heads", str(N_KV_HEADS),
                    "--group-size", "128", "--wgrad-backend", "deepgemm_fp8",
                    "--fp8-wgrad-roles", "ffn_gate,ffn_up,ffn_down",
                    "--allow-fp8-wgrad", "--warmup", "2", "--iterations", "5",
                    "--json-output", "system_preflight.json",
                ],
                cwd=trial,
                env=env,
                timeout=600,
            )
            probe = _run(
                [
                    sys.executable,
                    "src/throughput_probe.py",
                    "--modes", "transformerpp", "--batch-sizes", str(BATCH_SIZE),
                    "--seq-len", str(SEQ_LEN), "--steps", "4", "--warmup-steps", "2",
                    "--compile", "--compile-mode", "max-autotune", "--optimizer", "muon",
                    "--attention-backend", "fa3", "--bitlinear-kernel-backend", "triton_packed",
                    "--bitlinear-wgrad-backend", "deepgemm_fp8",
                    "--fp8-wgrad-roles", "ffn_gate,ffn_up,ffn_down",
                ],
                cwd=trial,
                env=env,
                timeout=900,
            )
            record = _probe_result(probe)
        finally:
            temporary.cleanup()
        peak_gb = float(record["peak_gb"])
        device_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if peak_gb > device_gb * 0.95:
            raise RuntimeError(f"candidate exceeds 95% HBM safety limit: {peak_gb:.2f}/{device_gb:.2f} GB")
        throughput = float(record["median_tokens_per_s"])
        return _result(
            {
                "patch_valid": 1.0,
                "static_contracts": 1.0,
                "numerical_correctness": 1.0,
                "compiled_training_step": 1.0,
                "median_tokens_per_s": throughput,
                "median_step_s": float(record["median_step_s"]),
                "peak_gb": peak_gb,
                "combined_score": throughput,
            },
            {
                "changed_paths": json.dumps(paths),
                "device": torch.cuda.get_device_name(),
                "device_memory_gb": device_gb,
                "elapsed_s": time.monotonic() - started,
                "static_contracts_tail": contracts[-2000:],
                "preflight_tail": preflight[-3000:],
                "throughput": json.dumps(record, sort_keys=True),
            },
        )
    except Exception as exc:
        return _fail(type(exc).__name__, str(exc))


def evaluate(program_path: str):
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    payload = evaluate(str(EXPERIMENT_DIR / "initial_program.py"))
    if EvaluationResult is not None and isinstance(payload, EvaluationResult):
        payload = {"metrics": payload.metrics, "artifacts": payload.artifacts}
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
