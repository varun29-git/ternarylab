from __future__ import annotations

import torch

from .model import BitLinear
from .quantization import (
    dequantize_activation_fp8_e4m3, dequantize_weight_fp8_e4m3_2d,
    quantize_fp8_directions, quantize_ternary_directions,
)


@torch.no_grad()
def relative_error(actual, expected):
    return float((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-100))


@torch.no_grad()
def gradient_metrics(actual, expected):
    actual, expected = actual.double().flatten(), expected.double().flatten()
    actual_norm, expected_norm = actual.norm(), expected.norm()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise FloatingPointError("nonfinite FP8 validation operand or result")
    cosine = ((actual @ expected) / (actual_norm * expected_norm).clamp_min(1e-100)
              if actual_norm > 0 and expected_norm > 0 else actual_norm.new_tensor(float(actual_norm == expected_norm)))
    return {"relative_error": relative_error(actual, expected), "cosine": float(cosine)}


def reference_linear(values, weight, gradient, mode):
    input_rows, input_columns = quantize_fp8_directions(values, transposed=mode == "full")
    weights, _ = quantize_ternary_directions(weight)
    effective_inputs = dequantize_activation_fp8_e4m3(*input_rows)
    effective_weights = dequantize_weight_fp8_e4m3_2d(*weights)
    with torch.autocast(device_type=values.device.type, enabled=False):
        forward = (effective_inputs @ effective_weights.T).bfloat16()
        baseline_input = gradient.float() @ effective_weights
        baseline_weight = gradient.float().T @ effective_inputs
        if mode == "bf16":
            grad_input = gradient @ effective_weights.bfloat16()
        else:
            gradient_rows, _ = quantize_fp8_directions(gradient, eps=0)
            grad_input = (dequantize_activation_fp8_e4m3(*gradient_rows) @ effective_weights).bfloat16()
        if mode == "full":
            _, gradient_columns = quantize_fp8_directions(gradient, rowwise=False, transposed=True, eps=0)
            grad_weight = dequantize_activation_fp8_e4m3(*gradient_columns) @ dequantize_activation_fp8_e4m3(*input_columns).T
        else:
            grad_weight = (gradient.T @ effective_inputs.bfloat16()).float()
    return {"forward": forward, "input_gradient": grad_input, "weight_gradient": grad_weight,
            "ste_input_gradient": baseline_input, "ste_weight_gradient": baseline_weight}


def projection_shapes(config):
    return {"query_output": (config.dim, config.dim),
            "key_value": (config.n_kv_heads * config.head_dim, config.dim),
            "gate_up": (config.ffn_dim, config.dim), "down": (config.dim, config.ffn_dim)}


def validate_quantizers(backend, device):
    errors = []
    for rows in (7, 128, 257):
        for magnitude in (0., 1e-12, 1.):
            inputs = (torch.randn(rows, 256, device=device) * magnitude).bfloat16()
            actual = quantize_fp8_directions(inputs, transposed=True, eps=0, backend=backend)
            expected = quantize_fp8_directions(inputs, transposed=True, eps=0)
            for native, reference in zip(actual, expected):
                error = relative_error(dequantize_activation_fp8_e4m3(*native),
                                       dequantize_activation_fp8_e4m3(*reference))
                if not error < .002:
                    raise AssertionError(f"fused FP8 quantizer disagrees with its reference: {error}")
                errors.append(error)
    weights = torch.randn(256, 384, device=device)
    primary, secondary = quantize_ternary_directions(weights, transposed=True, backend=backend)
    expected, _ = quantize_ternary_directions(weights)
    weight_error = relative_error(dequantize_weight_fp8_e4m3_2d(*primary), dequantize_weight_fp8_e4m3_2d(*expected))
    if not weight_error < .002 or not torch.equal(primary[0].float().T, secondary[0].float()):
        raise AssertionError("ternary quantizer or exact transpose mismatch")
    for invalid in (float("nan"), float("inf")):
        inputs[0, 0] = invalid
        for operand in quantize_fp8_directions(inputs, transposed=True, eps=0, backend=backend):
            if torch.isfinite(dequantize_activation_fp8_e4m3(*operand)).all():
                raise AssertionError("FP8 quantization silently sanitized a nonfinite input")
        weights[0, 0] = invalid
        operand, _ = quantize_ternary_directions(weights, backend=backend)
        if torch.isfinite(dequantize_weight_fp8_e4m3_2d(*operand)).all():
            raise AssertionError("ternary quantization silently sanitized a nonfinite latent weight")
    return {"max_directional_error": max(errors), "ternary_error": weight_error,
            "tiny_gradients": "passed", "nonfinite_propagation": "passed"}


def validate_linear_case(config, device, rows, outputs, width, *, stress="normal"):
    layer = BitLinear(width, outputs, quantize_activations=True, weight_scale_layout="blockwise", linear_backend="deepgemm",
                      fp8_backward=config.fp8_backward, fp8_quantizer=config.fp8_quantizer,
                      fp8_cache_weights=config.fp8_cache_weights).to(device)
    inputs = torch.randn(rows, width, device=device, dtype=torch.bfloat16, requires_grad=True)
    gradient = torch.randn(rows, outputs, device=device, dtype=torch.bfloat16)
    if stress == "zero_gradient":
        gradient.zero_()
    elif stress == "tiny_gradient":
        gradient *= 1e-12
    elif stress == "token_outliers":
        gradient *= torch.logspace(-6, 6, rows, device=device)[:, None]
    output = layer(inputs)
    output.backward(gradient)
    with torch.no_grad():
        expected = reference_linear(inputs.detach(), layer.weight.detach(), gradient, config.fp8_backward)
    actual = {"forward": output, "input_gradient": inputs.grad, "weight_gradient": layer.weight.grad}
    implementation = {name: gradient_metrics(value, expected[name]) for name, value in actual.items()}
    if not all(result["relative_error"] < .01 for result in implementation.values()):
        raise AssertionError(f"native FP8 operation differs from its explicit FP8 reference: {implementation}")
    approximation = {name: gradient_metrics(actual[name], expected["ste_" + name])
                     for name in ("input_gradient", "weight_gradient")}
    if not all(result["relative_error"] < .08 and result["cosine"] > .995 for result in approximation.values()):
        raise AssertionError(f"FP8 gradient approximation exceeded the numerical gate: {approximation}")
    if config.fp8_backward != "bf16":
        token_errors = ((inputs.grad.double() - expected["ste_input_gradient"].double()).norm(dim=1) /
                        expected["ste_input_gradient"].double().norm(dim=1).clamp_min(1e-100))
        if not float(token_errors.max()) < .15:
            raise AssertionError("FP8 dgrad erased or corrupted a low-magnitude token gradient")
    if layer.weight.grad.dtype != torch.float32:
        raise AssertionError("latent weight gradients must remain FP32")
    return {"rows": rows, "outputs": outputs, "width": width, "stress": stress,
            "implementation": implementation, "gradient_approximation": approximation,
            "weight_gradient_dtype": str(layer.weight.grad.dtype)}


def run_fp8_preflight(config, device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9:
        raise RuntimeError("ternary-FP8 certification requires the actual Hopper device")
    quantizers = validate_quantizers(config.fp8_quantizer, device)
    cases = [validate_linear_case(config, device, rows, 256, 256)
             for rows in (1, 7, 128, 257)]
    for stress in ("zero_gradient", "tiny_gradient", "token_outliers"):
        cases.append(validate_linear_case(config, device, 257, 256, 256, stress=stress))
    for role, (outputs, width) in projection_shapes(config).items():
        case = validate_linear_case(config, device, config.max_seq_len, outputs, width)
        cases.append({"role": role, **case})
    return {"backward_mode": config.fp8_backward, "quantizer": config.fp8_quantizer,
            "weight_cache": config.fp8_cache_weights, "quantizers": quantizers, "cases": cases,
            "scope": "Real native GEMMs versus explicit per-operation FP8 references and full-precision STE products. Numerical gates do not establish language-model convergence or quality parity."}
