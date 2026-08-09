from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT))
    from model import BitLinear, BitNetLlama, BitNetLlamaConfig
    from packed_ternary import (
        DENSE_POLICY_BF16_LARGE,
        PackedTernaryLinear,
        PackedTernaryWeight,
        build_packed_model_payload,
        pack_ternary_weight,
        verify_packed_model_payload,
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class PackedTernaryTest(unittest.TestCase):
    def test_known_code_mapping_and_padding(self) -> None:
        weight = torch.tensor([[0.0, 1.0, -1.0, 0.0, 3.0]], dtype=torch.float32)
        packed = pack_ternary_weight(weight, group_size=4)
        self.assertEqual(packed.codes.dtype, torch.uint8)
        self.assertEqual(tuple(packed.codes.shape), (2, 1))
        self.assertEqual(tuple(packed.scales.shape), (2, 1))
        self.assertEqual(packed.ternary_codes().tolist(), [[0, 1, -1, 0, 1]])
        restored = PackedTernaryWeight.from_payload(packed.to_payload())
        self.assertTrue(torch.equal(restored.codes, packed.codes))

    def test_matches_bitlinear_effective_weight(self) -> None:
        torch.manual_seed(7)
        layer = BitLinear(
            260,
            33,
            weight_init="normal",
            alpha_mode="deterministic_grouped_absmean",
            alpha_group_size=128,
        )
        packed = pack_ternary_weight(
            layer.weight,
            group_size=layer.alpha_group_size,
            eps=layer.eps,
        )
        torch.testing.assert_close(
            packed.scales.transpose(0, 1).reshape(-1),
            layer.alpha_values().detach(),
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(
            torch.equal(
                packed.ternary_codes(),
                layer.ternary_weight().detach().to(torch.int8),
            )
        )
        torch.testing.assert_close(
            packed.dequantize(),
            layer.quantized_weight().detach(),
            rtol=2e-7,
            atol=1e-8,
        )

    def test_reference_linear_matches_effective_linear(self) -> None:
        torch.manual_seed(11)
        latent = torch.randn(19, 132)
        packed = pack_ternary_weight(latent, group_size=4)
        module = PackedTernaryLinear(packed)
        inputs = torch.randn(3, 5, 132)
        expected = torch.nn.functional.linear(inputs, packed.dequantize())
        torch.testing.assert_close(module(inputs), expected, rtol=0.0, atol=0.0)

    def test_fp32_scales_are_compact_without_precision_loss(self) -> None:
        latent = torch.randn(768, 768)
        packed = pack_ternary_weight(latent, group_size=128, scale_dtype=torch.float32)
        self.assertGreater(packed.compression_ratio(), 14.0)

    def test_trainable_parameters_remain_fp32_and_adamw_steps(self) -> None:
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
        model = BitNetLlama(config)
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.float32})
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens = torch.arange(8).remainder(64).view(1, 8)
        model(tokens, targets=tokens)["loss"].backward()
        optimizer.step()

    def test_complete_model_payload_separates_dense_and_packed_state(self) -> None:
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
        model = BitNetLlama(config)
        payload = build_packed_model_payload(
            model,
            model_config=config.__dict__,
            source_checkpoint_sha256="a" * 64,
            run_id="unit-test",
            checkpoint_step=7,
        )
        summary = verify_packed_model_payload(payload)
        self.assertEqual(summary["matrix_count"], 14)
        self.assertIn("tok_embeddings.weight", payload["dense_state"])
        self.assertNotIn("layers.0.attn.q_proj.weight", payload["dense_state"])
        self.assertIn("layers.0.attn.q_proj", payload["packed_matrices"])

    def test_cost_aware_dense_policy_casts_large_matrix_but_not_norms(self) -> None:
        config = BitNetLlamaConfig(
            vocab_size=256,
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
        model = BitNetLlama(config)
        payload = build_packed_model_payload(
            model,
            model_config=config.__dict__,
            source_checkpoint_sha256="b" * 64,
            run_id="unit-test",
            checkpoint_step=8,
            dense_precision_policy=DENSE_POLICY_BF16_LARGE,
        )
        self.assertEqual(payload["dense_state"]["tok_embeddings.weight"].dtype, torch.bfloat16)
        self.assertEqual(payload["dense_state"]["norm.weight"].dtype, torch.float32)
        self.assertGreater(payload["storage_summary"]["dense_bytes_saved"], 0)


if __name__ == "__main__":
    unittest.main()
