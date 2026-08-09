from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepgemm_integration import (
    applicability_report,
    deepgemm_bf16_linear,
    deepgemm_bf16_wgrad,
)


class DeepGEMMIntegrationTest(unittest.TestCase):
    def test_applicability_keeps_packed_ternary_on_custom_kernel(self) -> None:
        report = applicability_report()
        self.assertEqual(report["packed_ternary_fprop"], "custom_triton")
        self.assertEqual(report["packed_ternary_dgrad"], "custom_triton_transpose")
        self.assertFalse(report["fp8_output_head"])
        self.assertEqual(report["bitlinear_fp8_wgrad"], "qualification_only_role_gated")

    def test_backend_fails_closed_without_cuda(self) -> None:
        activations = torch.zeros((1, 8), dtype=torch.bfloat16)
        weight = torch.zeros((4, 8), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            deepgemm_bf16_linear(activations, weight)

    def test_wgrad_fails_closed_without_cuda(self) -> None:
        grad_output = torch.zeros((8, 4), dtype=torch.bfloat16)
        activations = torch.zeros((8, 16), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            deepgemm_bf16_wgrad(grad_output, activations)


if __name__ == "__main__":
    unittest.main()
