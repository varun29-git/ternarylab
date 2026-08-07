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
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from model import BitLinear, BitNetLlama, BitNetLlamaConfig, RMSNorm, SingleScaleRMSNorm, UnitRMSNorm
    from optim import OptimizerConfig, muon_aux_adam_groups, prepare_keller_muon_groups


@unittest.skipIf(torch is None, "PyTorch is unavailable in this local environment")
class ScaleQualificationContractsTest(unittest.TestCase):
    def test_unit_qk_norm_has_no_parameters_and_bounds_scaled_dot_product(self) -> None:
        head_dim = 64
        q_norm = UnitRMSNorm(head_dim)
        k_norm = UnitRMSNorm(head_dim)
        self.assertEqual(sum(p.numel() for p in q_norm.parameters()), 0)
        self.assertEqual(sum(p.numel() for p in k_norm.parameters()), 0)

        q = q_norm(torch.randn(3, 5, head_dim, dtype=torch.float32))
        k = k_norm(torch.randn(3, 7, head_dim, dtype=torch.float32))
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        self.assertLessEqual(float(logits.abs().max()), math.sqrt(head_dim) + 1e-4)

    def test_row_local_grouping_is_explicit_and_exact(self) -> None:
        layer = BitLinear(8, 2, alpha_group_size=4, weight_init="normal")
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                        [-2.0, -2.0, -2.0, -2.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                )
            )
        expected = torch.tensor([2.5, 25.0, 2.0, 5.0])
        torch.testing.assert_close(layer._grouped_absmean_values(), expected)
        expanded = layer._expand_group_values(expected)
        torch.testing.assert_close(
            expanded,
            torch.tensor(
                [
                    [2.5, 2.5, 2.5, 2.5, 25.0, 25.0, 25.0, 25.0],
                    [2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0],
                ]
            ),
        )

    def test_misaligned_alpha_group_is_rejected_before_distributed_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            BitNetLlamaConfig(
                dim=12,
                n_layers=2,
                n_heads=2,
                n_kv_heads=1,
                ffn_dim=20,
                alpha_group_size=8,
                local_attention_window=8,
                global_attention_every=2,
                max_seq_len=8,
                require_shard_aligned_alpha_groups=True,
            )

    def test_model_uses_mixed_norm_policy(self) -> None:
        config = BitNetLlamaConfig(
            vocab_size=32,
            dim=8,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            ffn_dim=16,
            max_seq_len=16,
            alpha_group_size=4,
            local_attention_window=8,
            global_attention_every=2,
            attention_backend="sdpa_reference",
            qk_norm_mode="unit_rms",
        )
        model = BitNetLlama(config)
        self.assertIsInstance(model.layers[0].attn_norm, RMSNorm)
        self.assertIsInstance(model.layers[0].ffn_norm, RMSNorm)
        self.assertIsInstance(model.layers[0].attn.attn_sub_norm, SingleScaleRMSNorm)
        self.assertIsInstance(model.layers[0].ffn.ffn_sub_norm, SingleScaleRMSNorm)
        self.assertIsInstance(model.layers[0].attn.q_norm, UnitRMSNorm)
        self.assertIsInstance(model.layers[0].attn.k_norm, UnitRMSNorm)
        self.assertIsInstance(model.norm, RMSNorm)
        self.assertEqual(sum(p.numel() for p in model.layers[0].attn.q_norm.parameters()), 0)
        self.assertEqual(sum(p.numel() for p in model.layers[0].attn.k_norm.parameters()), 0)

    def test_muon_uses_per_matrix_rms_matching_without_changing_decay_rule(self) -> None:
        config = BitNetLlamaConfig(
            vocab_size=32,
            dim=8,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            ffn_dim=16,
            max_seq_len=16,
            alpha_group_size=4,
            local_attention_window=8,
            global_attention_every=2,
            attention_backend="sdpa_reference",
            qk_norm_mode="unit_rms",
        )
        model = BitNetLlama(config)
        opt_config = OptimizerConfig(
            kind="muon",
            muon_lr=8.0e-3,
            muon_update_scale="match_rms_adamw",
            muon_weight_decay=0.03,
        )
        groups = muon_aux_adam_groups(model, opt_config)
        muon_groups = [group for group in groups if group["use_muon"]]
        self.assertEqual(len(muon_groups), config.n_layers * 7)
        self.assertTrue(all(len(group["params"]) == 1 for group in muon_groups))

        q_group = next(group for group in muon_groups if group["name"] == "muon_transformer_l0_q_proj")
        q_param = q_group["params"][0]
        expected_scale = 0.2 * math.sqrt(max(q_param.shape[0], q_param.shape[1]))
        self.assertAlmostEqual(float(q_group["muon_update_scale"]), expected_scale)

        gate_group = next(group for group in muon_groups if group["name"] == "muon_transformer_l0_gate_proj")
        gate_param = gate_group["params"][0]
        gate_expected_scale = 0.2 * math.sqrt(max(gate_param.shape[0], gate_param.shape[1]))
        self.assertAlmostEqual(float(gate_group["muon_update_scale"]), gate_expected_scale)

        prepared, names = prepare_keller_muon_groups(groups)
        prepared_q_group = prepared[names.index("muon_transformer_l0_q_proj")]
        self.assertAlmostEqual(float(prepared_q_group["lr"]), opt_config.muon_lr * expected_scale)
        self.assertAlmostEqual(
            float(prepared_q_group["weight_decay"]),
            opt_config.muon_weight_decay / expected_scale,
        )


if __name__ == "__main__":
    unittest.main()
