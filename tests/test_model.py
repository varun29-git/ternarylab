import copy
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from terminus.generate import generate, sample_next_token
from terminus.model import BitLinear, BitNetLlama, BitNetLlamaConfig, RotaryEmbedding, RMSNorm, chunked_cross_entropy_sum
from terminus.quantization import (dequantize_activation_fp8_e4m3, quantize_activation_fp8_e4m3,
                                   weight_scales, DeepGEMMBitLinearFunction, quantize_weight_fp8_e4m3_2d,
                                   dequantize_weight_fp8_e4m3_2d)
from terminus.train import load_config
from tests.helpers import tiny_config


class QuantizationTests(unittest.TestCase):
    def test_partial_groups_exclude_padding(self):
        values = torch.full((3, 5), 0.1)
        for layout in ("rowwise", "blockwise"):
            torch.testing.assert_close(weight_scales(values, 4, layout=layout), torch.full((3 if layout == "rowwise" else 1, 2), .1))

    def test_tiny_scales_not_rounded_to_fp8_zero(self):
        layer = BitLinear(4, 3, alpha_group_size=4)
        with torch.no_grad():
            layer.weight.fill_(0.0002)
        torch.testing.assert_close(layer.quantized_weight(), layer.weight)

    def test_activation_roundtrip_noncontiguous_and_zero(self):
        for values in (torch.zeros(2, 7), torch.randn(3, 18)[:, ::2], torch.randn(2, 3, 11) * 1e4):
            codes, scales = quantize_activation_fp8_e4m3(values, 4)
            result = dequantize_activation_fp8_e4m3(codes, scales, 4)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result, values, rtol=.065, atol=1e-5)

    def test_activation_setting_does_not_change_weight_layout(self):
        layer = BitLinear(9, 7, alpha_group_size=4)
        before = layer.quantized_weight()
        layer.quantize_activations = True
        torch.testing.assert_close(before, layer.quantized_weight(), rtol=0, atol=0)

    def test_identity_ste_all_strengths_and_layouts(self):
        for layout in ("rowwise", "blockwise"):
            for strength in (0.0, 0.3, 1.0):
                layer = BitLinear(9, 7, alpha_group_size=4, weight_scale_layout=layout,
                                  quantize_activations=True, quantization_strength=strength)
                inputs = torch.randn(2, 3, 9, requires_grad=True)
                gradient = torch.randn(2, 3, 7)
                expected_weight = layer.quantized_weight().detach()
                expected_input = layer.quantized_activation(inputs)[0].detach()
                actual = layer(inputs)
                torch.testing.assert_close(actual, functional.linear(expected_input, expected_weight))
                actual.backward(gradient)
                torch.testing.assert_close(inputs.grad, gradient @ expected_weight)
                torch.testing.assert_close(layer.weight.grad, gradient.reshape(-1, 7).T @ expected_input.reshape(-1, 9))

    def test_native_autograd_matches_dequantized_operands(self):
        def reference_gemm(lhs, rhs):
            values = dequantize_activation_fp8_e4m3(*lhs, out_dtype=torch.bfloat16)
            weights = dequantize_weight_fp8_e4m3_2d(*rhs, out_dtype=torch.bfloat16)
            return functional.linear(values, weights)

        inputs = torch.randn(3, 128, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(128, 128, requires_grad=True)
        with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=reference_gemm):
            result = DeepGEMMBitLinearFunction.apply(inputs, weight)
        gradient = torch.randn_like(result)
        result.backward(gradient)
        input_codes = quantize_activation_fp8_e4m3(inputs)
        weight_codes = quantize_weight_fp8_e4m3_2d(weight)
        expected_input = dequantize_activation_fp8_e4m3(*input_codes, out_dtype=torch.bfloat16)
        expected_weight = dequantize_weight_fp8_e4m3_2d(*weight_codes, out_dtype=torch.bfloat16)
        torch.testing.assert_close(inputs.grad, gradient @ expected_weight)
        torch.testing.assert_close(weight.grad, (gradient.T @ expected_input).float())

    def test_strength_checkpoint_and_no_detached_gradient(self):
        model = BitNetLlama(tiny_config())
        model.set_quantization_strength_(0.25)
        restored = BitNetLlama(tiny_config())
        restored.load_state_dict(model.state_dict())
        for _, module in restored.bitlinear_named_modules():
            self.assertEqual(float(module.quantization_strength), .25)

    def test_invalid_native_config_fails(self):
        with self.assertRaises(ValueError):
            BitLinear(127, 128, linear_backend="deepgemm")


class ModelTests(unittest.TestCase):
    def test_cached_last_logits_avoids_prompt_vocabulary_tensor(self):
        model = BitNetLlama(tiny_config()).eval()
        inputs = torch.randint(1, 32, (1, 9))
        full, _ = model.forward_cached(inputs)
        last, cache = model.forward_cached(inputs, last_logits_only=True)
        self.assertEqual(last["logits"].shape, (1, 1, 32))
        self.assertEqual(cache.position, 9)
        torch.testing.assert_close(last["logits"], full["logits"][:, -1:], rtol=1e-5, atol=1e-5)

    def test_reduced_model_numerical_preflight_reference(self):
        from terminus.preflight import model_numerical_preflight

        result = model_numerical_preflight(tiny_config(), torch.device("cpu"))
        self.assertLess(max(result["errors"].values()), 1e-4)

    def test_loss_retention_benchmark(self):
        from terminus.benchmark import loss_memory_benchmark

        report = loss_memory_benchmark(tokens=37, vocabulary=128, width=16, chunk_size=7)
        self.assertLess(report["cases"]["chunked"]["retained_extra_forward_storage_bytes"],
                        report["cases"]["full"]["retained_extra_forward_storage_bytes"])

    def test_tiny_sampling_temperature_stays_finite(self):
        from terminus.generate import sample_next_token

        self.assertEqual(sample_next_token(torch.tensor([1., 2., 0.]), temperature=1e-300), 1)

    def setUp(self):
        torch.manual_seed(33)

    def test_parameter_counts(self):
        with torch.device("meta"):
            self.assertEqual(BitNetLlama().count_parameters()["total"], 150436648)
            candidate, _, _ = load_config("configs/1.5b_candidate.json")
            self.assertEqual(BitNetLlama(candidate).count_parameters()["total"], 1445062716)
            self.assertEqual(candidate.vocab_size, 68096)
            self.assertEqual(candidate.local_attention_window, 512)

    def test_bad_configuration(self):
        for options in ({"n_heads": 0}, {"n_layers": 3}, {"dim": 30}, {"rms_norm_eps": math.nan},
                        {"rope_theta": math.inf}, {"dropout": 1.0}, {"pad_token_id": 33}, {"loss_chunk_size": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                tiny_config(**options)

    def test_cache_chunking_and_storage(self):
        model = BitNetLlama(tiny_config()).eval()
        inputs = torch.randint(0, 32, (2, 16))
        expected = model(inputs)["logits"]
        for chunks in ((7, 3, 1, 5), (1,) * 16, (16,)):
            cache, outputs, position = None, [], 0
            for length in chunks:
                output, cache = model.forward_cached(inputs[:, position:position + length], cache=cache)
                outputs.append(output["logits"])
                position += length
            torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=2e-5, atol=2e-6)
            local = cache.layers[0]
            self.assertEqual(local.keys.shape[2], 4)
            self.assertEqual(local.keys.untyped_storage().nbytes(), local.keys.numel() * local.keys.element_size())
            with self.assertRaises(ValueError):
                model.forward_cached(inputs[:, :1], cache=cache)

    def test_cache_requires_eval_and_fixed_batch(self):
        model = BitNetLlama(tiny_config())
        inputs = torch.ones(1, 2, dtype=torch.long)
        with self.assertRaises(RuntimeError):
            model.forward_cached(inputs)
        model.eval()
        _, cache = model.forward_cached(inputs)
        with self.assertRaises(ValueError):
            model.forward_cached(inputs.expand(2, 2), cache=cache)

    def test_no_future_or_masked_key_leak(self):
        model = BitNetLlama(tiny_config()).eval()
        inputs = torch.randint(0, 32, (1, 12))
        changed = inputs.clone()
        changed[:, 7:] = (changed[:, 7:] + 1) % 32
        torch.testing.assert_close(model(inputs)["logits"][:, :7], model(changed)["logits"][:, :7])
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[:, :3] = False
        changed = inputs.clone()
        changed[:, :3] = (changed[:, :3] + 1) % 32
        torch.testing.assert_close(model(inputs, attention_mask=mask)["logits"][:, 3:],
                                   model(changed, attention_mask=mask)["logits"][:, 3:], rtol=0, atol=0)

    def test_mask_not_silently_accepted_by_production(self):
        config = tiny_config(dim=128, n_heads=2, n_kv_heads=1, attention_backend="fa3")
        model = BitNetLlama(config)
        with self.assertRaises(ValueError):
            model(torch.ones(1, 2, dtype=torch.long), attention_mask=torch.ones(1, 2, dtype=torch.bool))

    def test_chunk_loss_and_checkpointed_blocks_match_full_loss(self):
        inputs = torch.randint(0, 32, (2, 11))
        targets = torch.randint(0, 32, inputs.shape)
        targets[0, :5] = -100
        original = BitNetLlama(tiny_config())
        candidate = BitNetLlama(tiny_config(gradient_checkpointing=True))
        candidate.load_state_dict(original.state_dict())
        full = original(inputs, targets)
        chunked = candidate(inputs, targets, return_logits=False)
        torch.testing.assert_close(full["loss"], chunked["loss"])
        self.assertEqual(int(chunked["loss_tokens"]), 17)
        full["loss"].backward()
        chunked["loss"].backward()
        for expected, actual in zip(original.parameters(), candidate.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6)

    def test_all_ignored_loss_has_zero_finite_gradients(self):
        model = BitNetLlama(tiny_config())
        inputs = torch.ones(1, 7, dtype=torch.long)
        output = model(inputs, torch.full_like(inputs, -100), return_logits=False)
        self.assertEqual(float(output["loss"].detach()), 0)
        output["loss"].backward()
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertEqual(float(parameter.grad.abs().sum()), 0)

    def test_no_head_materialization_without_logits_or_targets(self):
        model = BitNetLlama(tiny_config())
        with patch("terminus.model.functional.linear", wraps=functional.linear) as linear:
            model(torch.ones(1, 3, dtype=torch.long), return_logits=False)
            self.assertTrue(all(call.args[1] is not model.tok_embeddings.weight for call in linear.call_args_list))

    def test_rope_preserves_fp32_frequencies_after_cast(self):
        rotary = RotaryEmbedding(64, 16384)
        expected = rotary(100, device=torch.device("cpu"), dtype=torch.float32, offset=8192)
        rotary.bfloat16()
        actual = rotary(100, device=torch.device("cpu"), dtype=torch.float32, offset=8192)
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_norm_large_bfloat16_values_are_finite(self):
        values = (torch.randn(2, 3, 32) * 1e8).bfloat16().requires_grad_()
        output = RMSNorm(32)(values)
        output.float().sum().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_nucleus_includes_crossing_token(self):
        logits = torch.tensor([.6, .3, .1]).log()
        with patch("torch.multinomial", return_value=torch.tensor([0])) as sampler:
            sample_next_token(logits, top_p=.8)
        torch.testing.assert_close(sampler.call_args.args[0], torch.tensor([2/3, 1/3, 0.]))
        with self.assertRaises(ValueError):
            sample_next_token(torch.tensor([float("nan")]))

    def test_generation_limits_and_mode_restoration(self):
        model = BitNetLlama(tiny_config())
        inputs = torch.ones(1, 3, dtype=torch.long)
        self.assertEqual(generate(model, inputs, max_new_tokens=3, temperature=0).shape, (1, 6))
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            generate(model, inputs, max_new_tokens=20)

    def test_tiny_overfit(self):
        model = BitNetLlama(tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        inputs = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        targets = (inputs + 1) % 4
        initial = float(model(inputs, targets, return_logits=False)["loss"].detach())
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs, targets, return_logits=False)["loss"]
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), initial * .5)

    def test_compile_strength_change_does_not_recompile(self):
        compilations = []

        def backend(graph, inputs):
            compilations.append(graph)
            return graph.forward

        layer = BitLinear(8, 8, alpha_group_size=4)
        compiled = torch.compile(layer, backend=backend, fullgraph=True)
        inputs = torch.randn(2, 8)
        for strength in (0., .25, .5, 1.):
            layer.quantization_strength.fill_(strength)
            torch.testing.assert_close(compiled(inputs), layer(inputs))
        self.assertEqual(len(compilations), 1)
