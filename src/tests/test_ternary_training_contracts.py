from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from model import BitLinear, BitNetLlamaConfig


class TernaryTrainingStaticContractTest(unittest.TestCase):
    def test_packed_backend_rejects_precision_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha_group_size=128"):
            BitNetLlamaConfig(
                attention_backend="sdpa_reference",
                bitlinear_kernel_backend="triton_packed",
                alpha_group_size=64,
            )
        with self.assertRaisesRegex(ValueError, "activation QAT"):
            BitNetLlamaConfig(
                attention_backend="sdpa_reference",
                bitlinear_kernel_backend="triton_packed",
                quantize_activations=True,
            )

    def test_fp8_requires_an_explicit_deepgemm_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "require"):
            BitNetLlamaConfig(
                attention_backend="sdpa_reference",
                fp8_wgrad_roles="ffn_up",
            )

    def test_launcher_defaults_to_safe_deepgemm_bf16(self) -> None:
        source = (ROOT / "run_h100_e2e.sh").read_text()
        self.assertIn('BITLINEAR_KERNEL_BACKEND="${BITLINEAR_KERNEL_BACKEND:-triton_packed}"', source)
        self.assertIn('BITLINEAR_WGRAD_BACKEND="${BITLINEAR_WGRAD_BACKEND:-deepgemm_bf16}"', source)
        self.assertIn("ALLOW_FP8_WGRAD=1", source)
        self.assertIn("ternary_training_preflight.py", source)

    def test_custom_kernel_ownership_is_separated_by_operand_type(self) -> None:
        source = (ROOT / "ternary_training.py").read_text()
        self.assertIn("packed_ternary_fprop", source)
        self.assertIn("packed_ternary_dgrad", source)
        self.assertIn("deepgemm_fp8_wgrad", source)
        self.assertNotIn("deepgemm_fp8_fprop", source)
        self.assertNotIn("deepgemm_fp8_dgrad", source)

    def test_archive_is_not_a_kernel_dependency(self) -> None:
        for relative in (
            "src/model.py",
            "src/ternary_training.py",
            "src/ternary_training_kernels.py",
            "src/deepgemm_integration.py",
            "src/run_h100_e2e.sh",
        ):
            source = (REPO_ROOT / relative).read_text()
            self.assertNotIn("archive/30b-run-original", source)

    def test_packed_cache_is_derived_not_checkpoint_state(self) -> None:
        layer = BitLinear(128, 64)
        state = layer.state_dict()
        self.assertNotIn("_packed_codes_cache", state)
        self.assertNotIn("_packed_scales_cache", state)
        self.assertEqual(layer.weight.dtype, torch.float32)

    def test_trainer_refreshes_cache_after_optimizer_update(self) -> None:
        source = (ROOT / "train.py").read_text()
        self.assertIn("model.refresh_packed_training_cache_(force=True)", source)
        self.assertIn('"refresh_contract": "after_each_applied_optimizer_update"', source)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "Hopper numerical kernel contract",
)
class TernaryTrainingHopperContractTest(unittest.TestCase):
    def test_pack_fprop_and_dgrad_match_reference(self) -> None:
        from packed_ternary import pack_ternary_weight
        from ternary_training_kernels import (
            pack_grouped_absmean_triton,
            packed_ternary_dgrad,
            packed_ternary_fprop,
        )

        torch.manual_seed(123)
        m, n, k = 64, 64, 128
        weight = torch.randn((n, k), device="cuda", dtype=torch.float32) / (k**0.5)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
        reference = pack_ternary_weight(weight, group_size=128)
        codes, scales = pack_grouped_absmean_triton(weight)
        self.assertTrue(torch.equal(codes, reference.codes))
        torch.testing.assert_close(scales, reference.scales, rtol=1.0e-6, atol=1.0e-8)
        effective = reference.dequantize(torch.bfloat16)

        actual_y = packed_ternary_fprop(x, codes, scales, in_features=k)
        reference_y = torch.nn.functional.linear(x, effective)
        torch.testing.assert_close(actual_y, reference_y, rtol=2.0e-2, atol=2.0e-2)

        actual_dx = packed_ternary_dgrad(dy, codes, scales, in_features=k)
        reference_dx = torch.mm(dy, effective)
        torch.testing.assert_close(actual_dx, reference_dx, rtol=2.0e-2, atol=2.0e-2)

    def test_custom_autograd_preserves_identity_ste(self) -> None:
        from packed_ternary import pack_ternary_weight
        from ternary_training import packed_ternary_linear
        from ternary_training_kernels import pack_grouped_absmean_triton

        torch.manual_seed(456)
        m, n, k = 64, 64, 128
        weight = (
            torch.randn((n, k), device="cuda", dtype=torch.float32) / (k**0.5)
        ).requires_grad_(True)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        upstream = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
        codes, scales = pack_grouped_absmean_triton(weight.detach())
        output = packed_ternary_linear(
            x,
            weight,
            codes,
            scales,
            group_size=128,
            wgrad_backend="torch_bf16",
            role="ffn_up",
            fp8_wgrad_roles="",
        )
        (output * upstream).sum().backward()

        effective = pack_ternary_weight(weight.detach()).dequantize(torch.bfloat16)
        reference_dx = torch.mm(upstream, effective)
        reference_dw = torch.mm(
            upstream.transpose(0, 1).contiguous(),
            x.detach(),
            out_dtype=torch.float32,
        )
        torch.testing.assert_close(x.grad, reference_dx, rtol=2.0e-2, atol=2.0e-2)
        torch.testing.assert_close(weight.grad, reference_dw, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
