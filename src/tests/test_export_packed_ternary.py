from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT))
    from export_packed_ternary import verify_checkpoint_sidecar, write_artifact_atomic
    from model import BitNetLlama, BitNetLlamaConfig
    from packed_ternary import build_packed_model_payload, verify_packed_model_payload


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class PackedExportTest(unittest.TestCase):
    def test_verified_checkpoint_to_atomic_artifact(self) -> None:
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "run_id": "unit-test",
                    "next_step": 12,
                    "model_config": asdict(config),
                    "model": model.state_dict(),
                },
                checkpoint,
            )
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint.with_suffix(".pt.json").write_text(
                json.dumps({"size_bytes": checkpoint.stat().st_size, "sha256": digest})
            )
            self.assertEqual(verify_checkpoint_sidecar(checkpoint), digest)
            payload = build_packed_model_payload(
                model,
                model_config=asdict(config),
                source_checkpoint_sha256=digest,
                run_id="unit-test",
                checkpoint_step=12,
            )
            verify_packed_model_payload(payload)
            output = root / "packed.pt"
            metadata = write_artifact_atomic(output, payload)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".pt.json").exists())
            self.assertEqual(metadata["artifact_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            loaded = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(verify_packed_model_payload(loaded)["matrix_count"], 14)


if __name__ == "__main__":
    unittest.main()
