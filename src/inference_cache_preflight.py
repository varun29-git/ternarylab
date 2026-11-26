from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from kernels import (
    KernelBackendError,
    clear_flex_block_mask_cache,
    flex_block_mask_cache_size,
)
from model import BitNetLlama, BitNetLlamaConfig


def tensor_error(actual: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    delta = actual.float() - reference.float()
    reference_rms = torch.sqrt(reference.float().square().mean()).clamp_min(1e-12)
    return {
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float(torch.sqrt(delta.square().mean()).div(reference_rms).item()),
    }


def assert_parity(label: str, error: Dict[str, float], args: argparse.Namespace) -> None:
    if error["max_abs"] > args.max_abs or error["relative_l2"] > args.max_relative_l2:
        raise AssertionError(
            f"{label} cache parity failed: {error}; "
            f"limits max_abs={args.max_abs}, relative_l2={args.max_relative_l2}"
        )


def build_model(
    backend: str,
    dtype: torch.dtype,
    *,
    max_seq_len: int,
    temperature_mode: str,
    softcap: float,
) -> BitNetLlama:
    # One full five-layer cycle exercises Terminus Flagship 30B's exact four-local/one-global
    # cache split. The 12:4, d=64 GQA geometry matches production.
    config = BitNetLlamaConfig(
        vocab_size=256,
        dim=768,
        n_layers=5,
        n_heads=12,
        n_kv_heads=4,
        ffn_dim=256,
        max_seq_len=max_seq_len,
        dropout=0.0,
        attention_backend=backend,
        local_attention_window=1024,
        global_attention_every=5,
        qk_norm_mode="unit_rms",
        attention_temperature_mode=temperature_mode,
        attention_softcap=softcap,
    )
    return BitNetLlama(config).cuda().to(dtype=dtype).eval()


@torch.inference_mode()
def run_backend(backend: str, args: argparse.Namespace) -> Dict[str, object]:
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)
    sustained_prompt_length = 64
    max_seq_len = max(1026, sustained_prompt_length + args.sustained_decode_steps)
    model = build_model(
        backend,
        dtype,
        max_seq_len=max_seq_len,
        temperature_mode=args.temperature_mode,
        softcap=args.softcap,
    )
    if backend == "flex":
        clear_flex_block_mask_cache()
    report: Dict[str, object] = {"backend": backend, "dtype": args.dtype, "lengths": []}
    for prefill_length in (1023, 1024, 1025):
        token_ids = torch.randint(
            1,
            model.config.vocab_size,
            (1, prefill_length + 1),
            device="cuda",
        )
        full = model(token_ids, return_logits=True)["logits"][:, -1, :]
        _, cache = model.forward_cached(token_ids[:, :prefill_length], return_logits=True)
        cached, cache = model.forward_cached(token_ids[:, prefill_length:], cache=cache, return_logits=True)
        cached_logits = cached["logits"][:, -1, :]
        error = tensor_error(cached_logits, full)
        assert_parity(f"{backend}/prefill{prefill_length}", error, args)
        local_caches = cache.layers[:4]
        global_cache = cache.layers[4]
        assert all(item is not None for item in local_caches) and global_cache is not None
        expected_local = min(1024, prefill_length + 1)
        expected_global = prefill_length + 1
        local_lengths = [item.length for item in local_caches if item is not None]
        if any(length != expected_local for length in local_lengths) or global_cache.length != expected_global:
            raise AssertionError(
                f"{backend}/prefill{prefill_length} wrong cache lengths: "
                f"local={local_lengths}/{expected_local}, global={global_cache.length}/{expected_global}"
            )
        report["lengths"].append(
            {
                "prefill_length": prefill_length,
                "decode_length": 1,
                "parity": error,
                "local_kv_tokens": local_lengths,
                "global_kv_tokens": global_cache.length,
            }
        )
    sustained_tokens = torch.randint(
        1,
        model.config.vocab_size,
        (1, sustained_prompt_length + args.sustained_decode_steps),
        device="cuda",
    )
    full_final = model(sustained_tokens, return_logits=True)["logits"][:, -1, :]
    cached, sustained_cache = model.forward_cached(
        sustained_tokens[:, :sustained_prompt_length],
        return_logits=True,
    )
    cache_entries_before_decode = flex_block_mask_cache_size()
    for position in range(sustained_prompt_length, sustained_tokens.shape[1]):
        cached, sustained_cache = model.forward_cached(
            sustained_tokens[:, position : position + 1],
            cache=sustained_cache,
            return_logits=True,
        )
    sustained_error = tensor_error(cached["logits"][:, -1, :], full_final)
    assert_parity(f"{backend}/sustained-decode", sustained_error, args)
    cache_entries_after_decode = flex_block_mask_cache_size()
    if backend == "flex" and cache_entries_after_decode != cache_entries_before_decode:
        raise AssertionError(
            "single-token Flex decode created shape-specific BlockMasks: "
            f"before={cache_entries_before_decode}, after={cache_entries_after_decode}"
        )
    report["sustained_decode"] = {
        "prompt_tokens": sustained_prompt_length,
        "decode_tokens": args.sustained_decode_steps,
        "parity": sustained_error,
        "flex_cache_entries_before": cache_entries_before_decode,
        "flex_cache_entries_after": cache_entries_after_decode,
        "final_local_kv_tokens": [
            item.length for item in sustained_cache.layers[:4] if item is not None
        ],
        "final_global_kv_tokens": sustained_cache.layers[4].length,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Terminus Flagship 30B cached local/global inference against full-context logits"
    )
    parser.add_argument("--backends", default="fa3,flex")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--temperature-mode",
        choices=["none", "bounded_per_head"],
        default="none",
    )
    parser.add_argument("--softcap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--max-abs", type=float, default=0.15)
    parser.add_argument("--max-relative-l2", type=float, default=0.05)
    parser.add_argument("--sustained-decode-steps", type=int, default=128)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run this on the target CUDA image")
    major, minor = torch.cuda.get_device_capability()
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    if not backends or any(item not in {"fa3", "flex"} for item in backends):
        raise ValueError("--backends must contain fa3 and/or flex")
    if "fa3" in backends and major < 9:
        raise SystemExit(
            f"FA3 cache preflight requires Hopper capability >= 9.0, got {major}.{minor}; "
            "use --backends flex on non-Hopper GPUs"
        )
    if args.softcap < 0.0:
        raise ValueError("softcap must be non-negative")
    report: Dict[str, object] = {
        "status": "running",
        "device": torch.cuda.get_device_name(),
        "capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "results": [],
    }
    try:
        for backend in backends:
            report["results"].append(run_backend(backend, args))
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
        raise SystemExit(f"inference cache preflight failed: {exc}") from exc
