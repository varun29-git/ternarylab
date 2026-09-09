import unittest
from dataclasses import replace

import torch

from terminus.preflight import run_preflight
from tests.helpers import tiny_config


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: these are mandatory hardware gates, not certified passes")
class GPUPreflightTests(unittest.TestCase):
    def test_fused_norm_swiglu_and_streaming_ternary_head(self):
        config = tiny_config(dim=128, n_heads=2, n_kv_heads=1, max_seq_len=32, attention_backend="fa3",
                             learned_norms=False, ternary_embeddings=True, loss_backend="streaming",
                             elementwise_backend="inductor", fused_projections=True)
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertEqual(result["status"], "passed")

    def test_flex_forward_backward_and_cached_offsets(self):
        config = tiny_config(dim=128, n_heads=2, n_kv_heads=1, max_seq_len=32, attention_backend="flex")
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertEqual(result["status"], "passed")

    def test_fa3_forward_backward_and_cached_offsets(self):
        config = tiny_config(dim=128, n_heads=2, n_kv_heads=1, max_seq_len=32, attention_backend="fa3")
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertEqual(result["status"], "passed")

    def test_deepgemm_full_fp8_backward_and_fused_cache(self):
        config = tiny_config(dim=256, n_heads=4, n_kv_heads=2, ffn_dim=512, max_seq_len=32,
                             attention_backend="fa3", linear_backend="deepgemm", quantize_activations=True,
                             weight_scale_layout="blockwise", fp8_backward="full", fp8_quantizer="triton",
                             fp8_cache_weights=True)
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertEqual(result["deepgemm"]["backward_mode"], "full")
        self.assertTrue(result["deepgemm"]["quantizers"])

    def test_deepgemm_native_forward_and_ste(self):
        config = tiny_config(dim=256, n_heads=4, n_kv_heads=2, ffn_dim=512, max_seq_len=32,
                             attention_backend="fa3", linear_backend="deepgemm", quantize_activations=True,
                             weight_scale_layout="blockwise")
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertTrue(result["deepgemm"])
