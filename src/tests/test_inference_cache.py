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
    from model import BitNetLlama, BitNetLlamaConfig, LayerKVCache


@unittest.skipIf(torch is None, "PyTorch is unavailable in this local environment")
class InferenceCacheTest(unittest.TestCase):
    def test_long_prefill_attends_before_compaction(self) -> None:
        cache = LayerKVCache(
            k=torch.empty(1, 1, 4, 2),
            v=torch.empty(1, 1, 4, 2),
            capacity=4,
        )
        values = torch.arange(10, dtype=torch.float32).view(1, 1, 5, 2)
        k_all, v_all, offset = cache.attention_inputs(values, values)
        self.assertEqual(offset, 0)
        self.assertEqual(k_all.shape[2], 5)
        cache.commit(k_all, v_all)
        self.assertEqual(cache.length, 4)
        torch.testing.assert_close(cache.active()[0], values[:, :, -4:, :])

    def test_cached_local_global_logits_match_full_context_across_boundary(self) -> None:
        torch.manual_seed(17017)
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
        model = BitNetLlama(config).eval()
        tokens = torch.randint(1, config.vocab_size, (1, 7))
        full = model(tokens, return_logits=True)["logits"][:, -1, :]
        _, cache = model.forward_cached(tokens[:, :5], return_logits=True)
        output, cache = model.forward_cached(tokens[:, 5:6], cache=cache, return_logits=True)
        output, cache = model.forward_cached(tokens[:, 6:], cache=cache, return_logits=True)
        torch.testing.assert_close(output["logits"][:, -1, :], full, rtol=1e-5, atol=1e-5)
        self.assertEqual(cache.layers[0].length, 4)
        self.assertEqual(cache.layers[1].length, 7)


if __name__ == "__main__":
    unittest.main()
