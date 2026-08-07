from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ImportError:  # Local macOS checkout does not necessarily carry the GPU stack.
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT))
    from model import BitNetLlama, BitNetLlamaConfig, diagnostic_quantile
    from train import (
        bitlinear_effective_update_stats,
        bitlinear_effective_value_sample,
        optimizer_group_stats,
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable in this local environment")
class Test16ClosureDiagnosticsTest(unittest.TestCase):
    def _model(self):
        config = BitNetLlamaConfig(
            vocab_size=64,
            dim=32,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            ffn_dim=64,
            max_seq_len=8,
            attention_backend="sdpa_reference",
            local_attention_window=4,
            global_attention_every=2,
            weight_init="normal",
        )
        return BitNetLlama(config).eval()

    def test_attention_and_activation_tail_payloads_cover_local_and_global(self) -> None:
        model = self._model()
        input_ids = torch.arange(8).remainder(64).view(1, 8)
        with torch.no_grad():
            output = model(
                input_ids,
                return_activation_stats=True,
                return_attention_stats=True,
            )
        activation = output["activation_stats"]
        for field in (
            "p99_abs",
            "p999_abs",
            "per_channel_rms_max",
            "per_channel_rms_p99",
            "per_channel_max_abs_p99",
        ):
            self.assertIn(field, activation)

        attention = output["attention_stats"]
        self.assertEqual([layer["attention_kind"] for layer in attention], ["local", "global"])
        for layer in attention:
            for field in (
                "p99_logit",
                "p999_logit",
                "near_512_mass",
                "distant_mass",
                "mean_head_attention_cosine_similarity",
                "pre_qknorm_q_vector_rms_max",
                "post_qknorm_q_vector_rms_max",
                "q_rope_rms_max_abs_delta",
                "post_rope_q_vector_rms_max",
                "heads",
            ):
                self.assertIn(field, layer)
        self.assertEqual(attention[0]["distant_mass"], 0.0)

    def test_diagnostic_quantile_accepts_all_training_float_dtypes(self) -> None:
        expected = 2.5
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
            actual = diagnostic_quantile(values, 0.5)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertAlmostEqual(float(actual), expected, places=5)

    def test_diagnostic_quantile_ignores_nonfinite_and_defaults_if_empty(self) -> None:
        mixed = torch.tensor(
            [float("nan"), 1.0, 3.0, float("inf"), float("-inf")],
            dtype=torch.float16,
        )
        self.assertEqual(float(diagnostic_quantile(mixed, 0.5)), 2.0)
        nonfinite = torch.tensor([float("nan"), float("inf")], dtype=torch.bfloat16)
        self.assertEqual(float(diagnostic_quantile(nonfinite, 0.99, default=-7.0)), -7.0)

    def test_attention_stats_are_dtype_safe_and_report_nonfinite_logits(self) -> None:
        attention = self._model().layers[0].attn
        q = torch.randn(1, attention.n_heads, 4, attention.head_dim, dtype=torch.float16)
        k = torch.randn(1, attention.n_kv_heads, 4, attention.head_dim, dtype=torch.float16)
        q[0, 0, 0, 0] = float("nan")
        stats = attention._attention_logit_stats(
            q,
            k,
            layer_index=0,
            backend="sdpa_reference",
            attention_kind="local",
            sample_seq_len=4,
        )
        self.assertGreater(stats["nonfinite_logit_count"], 0)
        self.assertLess(stats["finite_logit_fraction"], 1.0)
        self.assertTrue(math.isfinite(stats["p99_logit"]))
        self.assertTrue(math.isfinite(stats["mean_entropy"]))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is required to reproduce training autocast",
    )
    def test_attention_stats_survive_cuda_fp16_autocast(self) -> None:
        attention = self._model().layers[0].attn.cuda()
        q = torch.randn(
            1,
            attention.n_heads,
            8,
            attention.head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        k = torch.randn(
            1,
            attention.n_kv_heads,
            8,
            attention.head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            stats = attention._attention_logit_stats(
                q,
                k,
                layer_index=0,
                backend="sdpa_reference",
                attention_kind="local",
                sample_seq_len=8,
            )
        self.assertTrue(math.isfinite(stats["p99_logit"]))
        self.assertEqual(stats["nonfinite_logit_count"], 0)
        self.assertEqual(stats["finite_logit_fraction"], 1.0)

    def test_quantization_payload_keeps_every_module_and_tail_metrics(self) -> None:
        model = self._model()
        quantization = model.quantization_error_report(top_k=2)
        decomposition = model.quantization_error_decomposition_report(top_k=2)
        health = model.module_health_report(top_k=2)
        module_count = sum(1 for _ in model.bitlinear_named_modules())
        self.assertEqual(len(quantization["layers"]), module_count)
        self.assertEqual(len(decomposition["layers"]), module_count)
        self.assertEqual(len(health["layers"]), module_count)
        self.assertIn("by_attention_block_kind", quantization)
        self.assertIn("by_attention_block_kind", decomposition)
        for field in ("mean_relative_mse", "mean_abs_error", "mean_relative_mae"):
            self.assertIn(field, quantization)
        for field in ("relative_mse", "mean_abs_error", "relative_mae"):
            self.assertIn(field, quantization["layers"][0])
        for field in (
            "current_relative_mse",
            "oracle_relative_mse",
            "scale_gap_relative_mse",
            "support_gap_relative_mse",
            "irreducible_relative_mse",
            "scale_recovery_fraction",
            "support_recovery_fraction",
            "dominant_component",
            "top_scale_gap_layers",
            "top_support_gap_layers",
            "top_irreducible_layers",
        ):
            self.assertIn(field, decomposition)
        for field in (
            "mean_abs_over_alpha",
            "max_abs_over_alpha",
            "p99_abs_over_alpha",
            "normalized_abs_kurtosis",
            "p01_boundary_distance_over_alpha",
            "boundary_band_fraction_0p05",
            "saturated_tail_fraction_1p5",
            "zero_cell_error_fraction_of_total",
            "saturated_tail_error_fraction_of_total",
            "ternary_entropy",
        ):
            self.assertIn(field, health["layers"][0])
        self.assertIn("summary", health)
        self.assertIn("highest_saturated_tail_error_fraction", health)
        self.assertIn("highest_boundary_band_fraction", health)

    def test_functional_quantization_probe_reports_activation_weighted_error(self) -> None:
        model = self._model()
        model.enable_functional_quantization_probes(max_rows=4)
        input_ids = torch.arange(8).remainder(64).view(1, 8)
        with torch.no_grad():
            model(input_ids, return_logits=True)
        report = model.functional_quantization_error_report(top_k=2)
        self.assertGreater(report["module_count"], 0)
        self.assertEqual(len(report["layers"]), report["module_count"])
        self.assertIn("mean_relative_mse", report)
        self.assertIn("top_layers", report)
        self.assertTrue(all("attention_kind" in layer for layer in report["layers"]))
        model.clear_functional_quantization_probes()

    def test_optimizer_state_and_effective_update_payloads_are_nonempty(self) -> None:
        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.arange(8).remainder(64).view(1, 8)
        output = model(input_ids, targets=input_ids)
        output["loss"].backward()
        optimizer.step()
        state = optimizer_group_stats(optimizer)
        only_group = state["0"]
        self.assertGreater(only_group["state_tensor_count"], 0.0)
        self.assertIn("exp_avg", only_group["state_by_key"])

        before = bitlinear_effective_value_sample(model, 16)
        first_module = next(model.bitlinear_named_modules())[1]
        with torch.no_grad():
            first_module.weight.view(-1)[0].add_(0.25)
        after = bitlinear_effective_value_sample(model, 16)
        updates = bitlinear_effective_update_stats(before, after)
        self.assertEqual(len(updates["layers"]), len(before))
        self.assertTrue(
            any(layer["latent_update_rms"] > 0.0 for layer in updates["layers"])
        )
        self.assertIn("top_latent_update_over_alpha", updates)
        self.assertIn("top_half_cell_motion_rate", updates)
        self.assertIn("latent_update_over_alpha_rms", updates["layers"][0])
        self.assertIn("sample_half_cell_motion_rate", updates["layers"][0])


if __name__ == "__main__":
    unittest.main()
