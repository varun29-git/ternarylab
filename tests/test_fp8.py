import contextlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import PropertyMock, patch

import torch
from torch.utils.checkpoint import checkpoint

from terminus.kernels import KernelBackendError, deep_gemm_fp8_gemm_nt
from terminus.model import BitLinear, BitNetLlama, BitNetLlamaConfig
from terminus.quantization import (
    DeepGEMMBitLinearFunction, deepgemm_bitlinear, dequantize_activation_fp8_e4m3,
    dequantize_weight_fp8_e4m3_2d, quantize_activation_fp8_e4m3,
    quantize_fp8_directions, quantize_ternary_directions,
)
from tests.helpers import tiny_config


def emulated_fp8_gemm(lhs, rhs, *, rhs_group_size=128, out_dtype=torch.bfloat16):
    values = dequantize_activation_fp8_e4m3(*lhs)
    weights = (dequantize_activation_fp8_e4m3(*rhs) if rhs_group_size == 1 else
               dequantize_weight_fp8_e4m3_2d(*rhs))
    with torch.autocast(device_type=values.device.type, enabled=False):
        return (values @ weights.T).to(out_dtype)


def error_ratio(actual, expected):
    difference = (actual.double() - expected.double()).norm()
    return float(difference / expected.double().norm().clamp_min(1e-100))


class FP8PathTests(unittest.TestCase):
    def test_numerical_gate_against_independent_fp8_equations(self):
        from terminus.fp8_validation import validate_linear_case, validate_quantizers

        self.assertEqual(validate_quantizers("torch", torch.device("cpu"))["tiny_gradients"], "passed")
        for mode in ("bf16", "dgrad", "full"):
            config = tiny_config(dim=256, n_heads=4, n_kv_heads=2, ffn_dim=512,
                                 linear_backend="deepgemm", weight_scale_layout="blockwise",
                                 quantize_activations=True, fp8_backward=mode, fp8_cache_weights=True)
            with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
                for stress in ("normal", "zero_gradient", "tiny_gradient", "token_outliers"):
                    result = validate_linear_case(config, torch.device("cpu"), 17, 128, 128, stress=stress)
                    self.assertEqual(result["weight_gradient_dtype"], "torch.float32")

    def test_dynamic_gradient_scales_preserve_tiny_gradients(self):
        torch.manual_seed(17)
        for magnitude in (0.0, 1e-20, 1e-12, 1e-6, 1.0, 1e12):
            values = (torch.randn(17, 128) * magnitude).bfloat16()
            rows, columns = quantize_fp8_directions(values, transposed=True, eps=0)
            actual = dequantize_activation_fp8_e4m3(*rows)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertLess(error_ratio(actual, values), .04)
            transposed = dequantize_activation_fp8_e4m3(*columns)
            self.assertTrue(torch.equal(transposed[:, 17:], torch.zeros_like(transposed[:, 17:])))
            self.assertLess(error_ratio(transposed[:, :17], values.T), .04)

    def test_directional_inputs_are_quantized_from_original_values(self):
        torch.manual_seed(23)
        values = torch.randn(17, 256).bfloat16()
        rows, columns = quantize_fp8_directions(values, transposed=True)
        expected = quantize_activation_fp8_e4m3(torch.nn.functional.pad(values.T, (0, 111)).contiguous())
        self.assertTrue(torch.equal(columns[0].float(), expected[0].float()))
        self.assertTrue(torch.equal(columns[1], expected[1]))
        rounded_input = dequantize_activation_fp8_e4m3(*rows, out_dtype=values.dtype)
        _, requantized = quantize_fp8_directions(rounded_input, transposed=True)
        self.assertFalse(torch.equal(columns[0].float(), requantized[0].float()))

    def test_ternary_transpose_preserves_codes_and_scales(self):
        weights = torch.randn(256, 128)
        primary, secondary = quantize_ternary_directions(weights, transposed=True)
        self.assertTrue(set(primary[0].float().unique().tolist()) <= {-1., 0., 1.})
        self.assertTrue(torch.equal(primary[0].float().T, secondary[0].float()))
        torch.testing.assert_close(dequantize_weight_fp8_e4m3_2d(*primary).T,
                                   dequantize_weight_fp8_e4m3_2d(*secondary), rtol=0, atol=0)

    def test_all_backward_modes_and_tail_shapes(self):
        for mode in ("bf16", "dgrad", "full"):
            for rows in (1, 7, 128, 257):
                with self.subTest(mode=mode, rows=rows):
                    torch.manual_seed(rows)
                    values = torch.randn(rows, 128, dtype=torch.bfloat16, requires_grad=True)
                    weight = torch.randn(256, 128, requires_grad=True)
                    gradient = torch.randn(rows, 256, dtype=torch.bfloat16) * 1e-8
                    with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm) as gemm:
                        output = DeepGEMMBitLinearFunction.apply(values, weight, mode, "torch")
                        output.backward(gradient)
                    self.assertEqual(gemm.call_count, {"bf16": 1, "dgrad": 2, "full": 3}[mode])
                    input_rows, input_columns = quantize_fp8_directions(values.detach(), transposed=mode == "full")
                    weights, weights_transposed = quantize_ternary_directions(weight.detach(), transposed=mode != "bf16")
                    if mode == "bf16":
                        expected_input = gradient @ dequantize_weight_fp8_e4m3_2d(*weights, out_dtype=torch.bfloat16)
                    else:
                        gradient_rows, _ = quantize_fp8_directions(gradient, eps=0)
                        expected_input = emulated_fp8_gemm(gradient_rows, weights_transposed)
                    if mode == "full":
                        _, gradient_columns = quantize_fp8_directions(gradient, rowwise=False, transposed=True, eps=0)
                        expected_weight = emulated_fp8_gemm(gradient_columns, input_columns, rhs_group_size=1, out_dtype=torch.float32)
                        self.assertEqual(gemm.call_args.kwargs, {"rhs_group_size": 1, "out_dtype": torch.float32})
                        self.assertTrue((weight.grad != weight.grad.bfloat16().float()).any())
                    else:
                        expected_weight = (gradient.T @ dequantize_activation_fp8_e4m3(*input_rows, out_dtype=torch.bfloat16)).float()
                    torch.testing.assert_close(values.grad, expected_input, rtol=0, atol=0)
                    torch.testing.assert_close(weight.grad, expected_weight, rtol=0, atol=0)
                    self.assertEqual(weight.grad.dtype, torch.float32)
                    self.assertGreater(weight.grad.unique().numel(), 3)

    def test_only_requested_gradients_are_computed(self):
        for need_input, need_weight in ((False, True), (True, False)):
            values = torch.randn(7, 128, dtype=torch.bfloat16, requires_grad=need_input)
            weight = torch.randn(128, 128, requires_grad=need_weight)
            with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm) as gemm:
                deepgemm_bitlinear(values, weight, backward_mode="full").sum().backward()
            self.assertEqual(gemm.call_count, 2)
            self.assertEqual(values.grad is not None, need_input)
            self.assertEqual(weight.grad is not None, need_weight)

    def test_inference_does_not_build_backward_operands(self):
        values, weights = torch.randn(7, 128).bfloat16(), torch.randn(128, 128, requires_grad=True)
        with torch.inference_mode(), patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm), \
             patch("terminus.quantization.quantize_fp8_directions", wraps=quantize_fp8_directions) as quantize:
            output = deepgemm_bitlinear(values, weights, backward_mode="full")
        self.assertFalse(output.requires_grad)
        self.assertFalse(quantize.call_args.kwargs.get("transposed", False))

    def test_versioned_cache_reuses_and_invalidates_weights(self):
        layer = BitLinear(128, 128, quantize_activations=True, weight_scale_layout="blockwise",
                          linear_backend="deepgemm", fp8_backward="full", fp8_cache_weights=True)
        with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm), \
             patch("terminus.model.quantize_ternary_directions", wraps=quantize_ternary_directions) as quantize:
            for _ in range(3):
                values = torch.randn(7, 128, dtype=torch.bfloat16, requires_grad=True)
                layer(values).float().square().mean().backward()
            self.assertEqual(quantize.call_count, 1)
            with torch.no_grad():
                layer.weight.add_(.1)
            layer(values).sum().backward()
            self.assertEqual(quantize.call_count, 2)
            layer.load_state_dict(layer.state_dict())
            layer(values).sum().backward()
            self.assertEqual(quantize.call_count, 3)
            layer.clear_fp8_weight_cache()
            layer(values).sum().backward()
            self.assertEqual(quantize.call_count, 4)
        self.assertEqual(set(layer.state_dict()), {"weight", "quantization_strength"})

    def test_cached_checkpoint_recomputation_preserves_gradients(self):
        layer = BitLinear(128, 128, quantize_activations=True, weight_scale_layout="blockwise",
                          linear_backend="deepgemm", fp8_backward="full", fp8_cache_weights=True)
        values = torch.randn(7, 128, dtype=torch.bfloat16, requires_grad=True)
        with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
            layer(values).float().square().sum().backward()
            expected = values.grad.clone(), layer.weight.grad.clone()
            values.grad = layer.weight.grad = None
            checkpoint(layer, values, use_reentrant=False).float().square().sum().backward()
        torch.testing.assert_close(values.grad, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(layer.weight.grad, expected[1], rtol=0, atol=0)

    def test_inference_cache_is_not_reused_by_training(self):
        for mode in ("bf16", "dgrad", "full"):
            layer = BitLinear(128, 128, quantize_activations=True, weight_scale_layout="blockwise",
                              linear_backend="deepgemm", fp8_backward=mode, fp8_cache_weights=True)
            values = torch.randn(7, 128, dtype=torch.bfloat16, requires_grad=True)
            with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
                with torch.inference_mode():
                    layer(values)
                self.assertTrue(layer._fp8_weight_cache[1][0][0].is_inference())
                layer(values).float().square().sum().backward()
                self.assertFalse(layer._fp8_weight_cache[1][0][0].is_inference())
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertTrue(torch.isfinite(values.grad).all())

    def test_complete_model_accumulation_and_optimizer_updates(self):
        config = tiny_config(dim=256, n_heads=4, n_kv_heads=2, ffn_dim=512, gradient_checkpointing=True,
                             linear_backend="deepgemm", weight_scale_layout="blockwise", quantize_activations=True,
                             fp8_backward="full", fp8_cache_weights=True)
        model = BitNetLlama(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randint(1, config.vocab_size, (2, 7))
        targets = torch.randint(1, config.vocab_size, inputs.shape)
        with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                for _ in range(2):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        loss = model(inputs, targets, return_logits=False)["loss"] / 2
                    loss.backward()
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    and parameter.grad.dtype == torch.float32 for parameter in model.parameters()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                model.clear_fp8_weight_cache()
                self.assertTrue(all(layer._fp8_weight_cache is None for _, layer in model.bitlinear_named_modules()))
        self.assertTrue(all(set(layer.ternary_weight().unique().tolist()) <= {-1, 0, 1}
                            for _, layer in model.bitlinear_named_modules()))
        self.assertFalse(any("cache" in name for name in model.state_dict()))

    def test_h100_configs_counts_and_matched_benchmark_options(self):
        from terminus.fp8_benchmark import VARIANTS, run_benchmark, variant_options
        from terminus.preflight import count_model
        from terminus.train import load_config

        root = Path(__file__).resolve().parents[1]
        full_path = root / "configs/1.5b_h100_ternary_fp8.json"
        full, training, optimizer = load_config(full_path)
        conservative, _, _ = load_config(root / "configs/1.5b_h100_ternary_fp8_dgrad.json")
        reference, _, _ = load_config(root / "configs/1.5b_h100_ternary_bf16_reference.json")
        self.assertEqual(full.fp8_backward, "full")
        self.assertEqual(conservative.fp8_backward, "dgrad")
        self.assertEqual(reference.linear_backend, "torch")
        for config in (full, conservative, reference):
            self.assertEqual(config.weight_scale_layout, "blockwise")
            self.assertEqual(count_model(config)["total"], 1502210108)
        self.assertEqual(count_model(full)["fp8_weight_cache_estimated_bytes"], 2611591680)
        self.assertEqual(count_model(reference)["fp8_weight_cache_estimated_bytes"], 0)
        for variant in VARIANTS:
            self.assertEqual(variant_options(variant, full)["weight_scale_layout"], "blockwise")
        with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "actual Hopper"):
            run_benchmark(full_path, None)
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "compiled.json"
            invalid.write_text(json.dumps({"model": asdict(full), "training": {**asdict(training), "compile": True},
                                           "optimizer": asdict(optimizer)}))
            with self.assertRaisesRegex(ValueError, "torch.compile"):
                load_config(invalid)

    def test_preflight_requires_exact_fp8_policy(self):
        from terminus.integrity import fingerprint
        from terminus.preflight import validate_preflight_report

        config = tiny_config(dim=256, n_heads=4, n_kv_heads=2, ffn_dim=512,
                             linear_backend="deepgemm", weight_scale_layout="blockwise", quantize_activations=True,
                             fp8_backward="full", fp8_quantizer="triton", fp8_cache_weights=True)
        native = {"backward_mode": "full", "quantizer": "triton", "weight_cache": True,
                  "cases": [{"test_only": True}], "quantizers": {"test_only": True}}
        report = {"identity": {"model_sha256": fingerprint(asdict(config)), "source": {}, "runtime": {}},
                  "status": "passed", "attention": [True], "model_numerics": {"test_only": True}, "deepgemm": native}
        with tempfile.TemporaryDirectory() as directory, patch("terminus.preflight.source_hashes", return_value={}), \
             patch("terminus.preflight.runtime_contract", return_value={}):
            path = Path(directory) / "test_receipt.json"
            path.write_text(json.dumps(report))
            validate_preflight_report(path, config, torch.device("cpu"))
            for change in ({"backward_mode": "bf16"}, {"quantizer": "torch"}, {"weight_cache": False},
                           {"cases": []}, {"quantizers": {}}):
                path.write_text(json.dumps({**report, "deepgemm": {**native, **change}}))
                with self.assertRaisesRegex(ValueError, "FP8 backward/quantizer/cache policy"):
                    validate_preflight_report(path, config, torch.device("cpu"))

    def test_1d1d_api_requires_zero_initialized_fp32_accumulation(self):
        calls = []

        def native(lhs, rhs, output, **options):
            self.assertIs(options["c"], output)
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            self.assertEqual(options["recipe"], (1, 1, 128))
            self.assertTrue(options["disable_ue8m0_cast"])
            calls.append(output.dtype)
            output.copy_(emulated_fp8_gemm(lhs, rhs, rhs_group_size=1, out_dtype=output.dtype))

        module = types.SimpleNamespace(fp8_gemm_nt=native, get_mn_major_tma_aligned_tensor=lambda value: value)
        operand = quantize_activation_fp8_e4m3(torch.randn(128, 128))
        with patch.dict(sys.modules, {"deep_gemm": module}), \
             patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            output = deep_gemm_fp8_gemm_nt(operand, operand, rhs_group_size=1, out_dtype=torch.float32)
            with self.assertRaisesRegex(KernelBackendError, "FP32"):
                deep_gemm_fp8_gemm_nt(operand, operand, rhs_group_size=1)
        self.assertEqual(calls, [torch.float32])
        self.assertTrue(torch.isfinite(output).all())

    def test_invalid_options_fail_closed(self):
        for options in ({"fp8_backward": "automatic"}, {"fp8_quantizer": "automatic"}, {"fp8_backward": "full"},
                        {"fp8_quantizer": "triton"}, {"fp8_cache_weights": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                tiny_config(**options)
        with self.assertRaisesRegex(ValueError, "FP32"):
            deepgemm_bitlinear(torch.ones(1, 128).bfloat16(), torch.ones(128, 128).bfloat16())

    def test_nonfinite_inputs_are_not_sanitized(self):
        for invalid in (float("nan"), float("inf")):
            values = torch.ones(7, 128)
            values[0, 0] = invalid
            primary, secondary = quantize_fp8_directions(values, transposed=True, eps=0)
            self.assertFalse(torch.isfinite(dequantize_activation_fp8_e4m3(*primary)).all())
            self.assertFalse(torch.isfinite(dequantize_activation_fp8_e4m3(*secondary)).all())


@unittest.skipUnless(torch.cuda.is_available(), "Hopper CUDA unavailable: fused FP8 kernels are not certified locally")
class GPUFP8QuantizerTests(unittest.TestCase):
    def test_fused_directional_quantization_and_layout(self):
        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("SM90 required")
        for rows in (1, 7, 128, 257):
            for magnitude in (0., 1e-12, 1.):
                values = (torch.randn(rows, 512, device="cuda") * magnitude).bfloat16()[:, ::2]
                expected = quantize_fp8_directions(values, transposed=True, eps=0)
                actual = quantize_fp8_directions(values, transposed=True, eps=0, backend="triton")
                for native, reference in zip(actual, expected):
                    self.assertEqual(native[1].stride(), (1, ((native[1].shape[0] + 3) // 4) * 4))
                    self.assertLess(error_ratio(dequantize_activation_fp8_e4m3(*native),
                                               dequantize_activation_fp8_e4m3(*reference)), .002)
        values[0, 0] = float("nan")
        for operand in quantize_fp8_directions(values, transposed=True, eps=0, backend="triton"):
            self.assertFalse(torch.isfinite(dequantize_activation_fp8_e4m3(*operand)).all())

    def test_fused_ternary_scales_and_exact_transpose(self):
        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("SM90 required")
        weights = torch.randn(256, 384, device="cuda")
        primary, secondary = quantize_ternary_directions(weights, transposed=True, backend="triton")
        expected, _ = quantize_ternary_directions(weights)
        self.assertLess(error_ratio(dequantize_weight_fp8_e4m3_2d(*primary),
                                   dequantize_weight_fp8_e4m3_2d(*expected)), .002)
        self.assertTrue(torch.equal(primary[0].float().T, secondary[0].float()))
        self.assertTrue(torch.equal(primary[1].T, secondary[1]))
        weights[0, 0] = float("nan")
        primary, _ = quantize_ternary_directions(weights, backend="triton")
        self.assertFalse(torch.isfinite(dequantize_weight_fp8_e4m3_2d(*primary)).all())
