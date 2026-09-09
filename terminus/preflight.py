from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .integrity import fingerprint, source_hashes, write_json_atomic
from .kernels import attention, sdpa_reference_attention
from .model import BitNetLlama
from .fp8_validation import relative_error, run_fp8_preflight
from .train import load_config, runtime_contract


def count_model(config):
    with torch.device("meta"):
        model = BitNetLlama(config)
        counts = model.count_parameters()
        cached = [module for _, module in model.bitlinear_named_modules() if module.fp8_cache_weights]
    directions = 2 if config.fp8_backward != "bf16" else 1
    cache_bytes = sum(module.weight.numel() * directions + module.alpha_values().numel() * 4 * directions for module in cached)
    return {**counts, "fp8_weight_cache_estimated_bytes": cache_bytes, "fp32_weights_and_gradients_bytes": counts["total"] * 8,
            "adamw_fp32_training_state_bytes_excluding_activations": counts["total"] * 16,
            "note": "Latent training state is not 1.58-bit; activations, workspaces and communication buffers are additional."}



def model_numerical_preflight(config, device):
    reduced = replace(config, vocab_size=257, dim=4 * config.head_dim, n_heads=4, n_kv_heads=2,
                      n_layers=config.global_attention_every, ffn_dim=((32 * config.head_dim // 3 + 127) // 128) * 128,
                      max_seq_len=min(35, config.max_seq_len), local_attention_window=min(11, config.local_attention_window),
                      pad_token_id=0, loss_chunk_size=7)
    optimized = BitNetLlama(reduced).to(device)
    reference = BitNetLlama(replace(reduced, gradient_checkpointing=False, elementwise_backend="torch")).to(device)
    reference.load_state_dict(optimized.state_dict())
    inputs = torch.randint(1, reduced.vocab_size, (1, reduced.max_seq_len), device=device)
    targets = torch.randint(1, reduced.vocab_size, inputs.shape, device=device)
    targets[:, 0] = -100
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        actual = optimized(inputs, targets, return_logits=False)["loss"]
        expected = reference(inputs, targets, return_logits=True)["loss"]
    actual.backward()
    expected.backward()
    gradient_errors = [relative_error(parameter.grad, reference_parameter.grad)
                       for parameter, reference_parameter in zip(optimized.parameters(), reference.parameters())]
    errors = {"loss_relative_error": relative_error(actual, expected), "max_parameter_gradient_relative_error": max(gradient_errors)}
    del reference
    optimized.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        full = optimized(inputs)["logits"]
        cache, chunks = None, []
        for start in range(0, inputs.shape[1], 3):
            output, cache = optimized.forward_cached(inputs[:, start:start + 3], cache=cache)
            chunks.append(output["logits"])
        errors["cached_logits_relative_error"] = relative_error(torch.cat(chunks, dim=1), full)
    if not all(error < 0.03 for error in errors.values()):
        raise AssertionError(f"reduced full-model loss/gradient/cache mismatch: {errors}")
    return {"model": asdict(reduced), "errors": errors,
            "scope": "Reduced full-model composition on the selected backend, not a 1.5B allocation or optimizer soak."}


def validate_preflight_report(path, config, device):
    if path is None:
        raise ValueError("production requires --preflight-report from this exact code, configuration and GPU runtime")
    report = json.loads(Path(path).read_text())
    expected = {"model_sha256": fingerprint(asdict(config)), "source": source_hashes(), "runtime": runtime_contract(device)}
    if report.get("identity") != expected or report.get("status") != "passed":
        raise ValueError("GPU preflight is missing, stale or incompatible")
    if not report.get("attention") or not report.get("model_numerics") or (config.linear_backend == "deepgemm" and not report.get("deepgemm")):
        raise ValueError("GPU preflight lacks required backend evidence")
    if config.linear_backend == "deepgemm":
        native = report["deepgemm"]
        if (not isinstance(native, dict) or not native.get("cases") or not native.get("quantizers")
                or native.get("backward_mode") != config.fp8_backward
                or native.get("quantizer") != config.fp8_quantizer
                or native.get("weight_cache") != config.fp8_cache_weights):
            raise ValueError("GPU preflight does not certify this FP8 backward/quantizer/cache policy")


def run_preflight(config, device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires CUDA; CPU tests do not certify production kernels")
    if config.attention_backend == "sdpa_reference":
        raise ValueError("reference attention cannot pass a production preflight")
    torch.manual_seed(1337)
    cases = []
    lengths = sorted({1, max(1, config.local_attention_window - 1), config.local_attention_window,
                      min(config.max_seq_len, config.local_attention_window + 1), config.max_seq_len})
    for length in lengths:
        for window in (None, config.local_attention_window):
            for query_length in sorted({1, min(3, length), length}):
                shape_query = (1, config.n_heads, query_length, config.head_dim)
                shape_key = (1, config.n_kv_heads, length, config.head_dim)
                operands = [torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
                            for shape in (shape_query, shape_key, shape_key)]
                reference_operands = [operand.detach().clone().requires_grad_() for operand in operands]
                offset = length - query_length
                result = attention(*operands, backend=config.attention_backend, local_window=window, query_position_offset=offset)
                expected = sdpa_reference_attention(*reference_operands, local_window=window, query_position_offset=offset)
                grad_output = torch.randn_like(result)
                gradients = torch.autograd.grad(result, operands, grad_output)
                reference_gradients = torch.autograd.grad(expected, reference_operands, grad_output)
                errors = [relative_error(result, expected)] + [relative_error(actual, target) for actual, target in zip(gradients, reference_gradients)]
                accepted = [error < 0.03 for error in errors]
                if length == 1:
                    # With one key softmax is identically one: dQ=dK=0.
                    # BF16 kernels/reference can leave sub-micro absolute
                    # residuals; a ratio between those residuals is undefined
                    # as an accuracy measure. Test the analytical zero instead.
                    for index in (0, 1):
                        accepted[index + 1] = max(float(gradients[index].abs().max()),
                                                  float(reference_gradients[index].abs().max())) < 1e-6
                if not all(accepted):
                    details = [{"reference_norm": float(reference.double().norm()),
                                "max_absolute_error": float((actual.double() - reference.double()).abs().max())}
                               for actual, reference in zip([result, *gradients], [expected, *reference_gradients])]
                    raise AssertionError(f"attention forward/backward mismatch: length={length}, query_length={query_length}, "
                                         f"window={window}, relative_errors={errors}, details={details}")
                cases.append({"length": length, "query_length": query_length, "window": window, "relative_errors": errors,
                              "qk_analytical_zero_atol": 1e-6 if length == 1 else None})
    deepgemm = run_fp8_preflight(config, device) if config.linear_backend == "deepgemm" else None
    model_numerics = model_numerical_preflight(config, device)
    torch.cuda.synchronize(device)
    query = torch.randn(1, config.n_heads, config.max_seq_len, config.head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, config.n_kv_heads, config.max_seq_len, config.head_dim, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        attention(query, key, key, backend=config.attention_backend, local_window=config.local_attention_window)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(10):
        attention(query, key, key, backend=config.attention_backend, local_window=config.local_attention_window)
    torch.cuda.synchronize(device)
    return {"status": "passed", "identity": {"model_sha256": fingerprint(asdict(config)), "source": source_hashes(), "runtime": runtime_contract(device)},
            "counts": count_model(config), "attention": cases, "deepgemm": deepgemm, "model_numerics": model_numerics,
            "local_attention_forward_ms": (time.perf_counter() - started) * 100,
            "scope": "Kernel and reduced-model numerical gate only; full-scale throughput, quality and multi-node soak tests still required."}


def main():
    parser = argparse.ArgumentParser(description="Count model parameters without allocation, or validate actual GPU kernels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config, _, _ = load_config(args.config)
    result = count_model(config) if args.count_only else run_preflight(config, torch.device("cuda", 0))
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
