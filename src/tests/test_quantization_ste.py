from __future__ import annotations

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
    from model import BitLinear


@unittest.skipIf(torch is None, "PyTorch is unavailable in this local environment")
class QuantizationStraightThroughEstimatorTest(unittest.TestCase):
    def _deterministic_layer(self):
        layer = BitLinear(
            4,
            2,
            weight_init="normal",
            alpha_mode="deterministic_grouped_absmean",
            alpha_group_size=4,
            quantization_strength=1.0,
            quantize_activations=False,
        )
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [
                        [0.10, 0.25, 0.80, -0.50],
                        [-0.15, 0.35, -0.90, 0.60],
                    ],
                    dtype=layer.weight.dtype,
                )
            )
        return layer

    def test_forward_is_exact_grouped_alpha_times_ternary(self) -> None:
        layer = self._deterministic_layer()
        actual = layer.quantized_weight().detach()
        alpha = layer.alpha_scale().detach()
        expected = alpha * torch.round(layer.weight.detach().float() / alpha).clamp(-1, 1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_deterministic_weight_backward_is_identity(self) -> None:
        layer = self._deterministic_layer()
        upstream = torch.tensor(
            [[0.30, -0.70, 1.10, 0.20], [-0.40, 0.90, 0.60, -1.20]],
            dtype=torch.float32,
        )
        loss = torch.sum(layer.quantized_weight() * upstream)
        loss.backward()
        torch.testing.assert_close(layer.weight.grad, upstream, rtol=0.0, atol=0.0)

    def test_learned_scale_retains_an_explicit_gradient_path(self) -> None:
        layer = BitLinear(
            4,
            2,
            weight_init="normal",
            alpha_mode="learned_grouped_log_gain",
            alpha_group_size=4,
            quantization_strength=1.0,
            quantize_activations=False,
        )
        upstream = torch.ones_like(layer.weight, dtype=torch.float32)
        loss = torch.sum(layer.quantized_weight() * upstream)
        loss.backward()
        torch.testing.assert_close(layer.weight.grad, upstream, rtol=0.0, atol=0.0)
        self.assertIsNotNone(layer.log_gain.grad)
        self.assertTrue(bool(torch.isfinite(layer.log_gain.grad).all()))

    def test_activation_forward_and_backward_use_full_dequantized_target(self) -> None:
        layer = BitLinear(
            4,
            2,
            weight_init="normal",
            alpha_mode="deterministic_grouped_absmean",
            alpha_group_size=4,
            quantization_strength=1.0,
            quantize_activations=True,
        )
        x = torch.tensor(
            [[[0.1, -0.4, 0.7, 1.2]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        quantized, scale = layer.quantized_activation(x)
        integer = torch.round(x.detach() / scale.detach()).clamp(
            -layer.activation_qmax,
            layer.activation_qmax,
        )
        torch.testing.assert_close(quantized.detach(), integer * scale.detach(), rtol=0.0, atol=0.0)
        upstream = torch.tensor([[[0.3, -0.7, 1.1, 0.2]]])
        torch.sum(quantized * upstream).backward()
        torch.testing.assert_close(x.grad, upstream, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
