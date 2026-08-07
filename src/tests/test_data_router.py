from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np
    import torch  # noqa: F401
except ImportError:
    np = None

if np is not None:
    sys.path.insert(0, str(ROOT))
    from data import TokenBatcher, TokenDataConfig, TokenManifest, TokenShardSpec, TrainingTokenRouter


class CurriculumTelemetryWiringTest(unittest.TestCase):
    def test_training_logs_router_backed_curriculum_status(self) -> None:
        train = (ROOT / "train.py").read_text()
        self.assertIn("last_curriculum = val_batcher.curriculum_status(next_step)", train)
        self.assertNotIn("def curriculum_status(manifest:", train)


@unittest.skipIf(np is None, "NumPy/PyTorch are unavailable in this local environment")
class DataRouterTest(unittest.TestCase):
    def test_weighted_mix_is_exact_and_interleaved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.full(1601, 10, dtype=np.uint16).tofile(root / "cosmo.bin")
            np.full(401, 20, dtype=np.uint16).tofile(root / "fineweb.bin")
            np.full(101, 30, dtype=np.uint16).tofile(root / "val.bin")
            manifest = TokenManifest(
                config=TokenDataConfig(
                    seq_len=2,
                    batch_size=1,
                    total_train_tokens=2000,
                    validation_tokens=100,
                    source_mix={"cosmo": 0.8, "fineweb": 0.2},
                ),
                shards=[
                    TokenShardSpec("cosmo.bin", "cosmo", "train", 1601),
                    TokenShardSpec("fineweb.bin", "fineweb", "train", 401),
                    TokenShardSpec("val.bin", "validation", "val", 101),
                ],
            )
            router = TrainingTokenRouter(manifest, root=root)
            self.assertEqual(router.mix_schedule.count("cosmo"), 800)
            self.assertEqual(router.mix_schedule.count("fineweb"), 200)
            self.assertNotEqual(len(set(router.mix_schedule[:10])), 1)
            observed = []
            for example_index in range(20):
                inputs, _ = router.block_at_token_offset(example_index * 2)
                observed.append(int(inputs[0]))
            self.assertIn(10, observed)
            self.assertIn(20, observed)

    def test_phase_weighted_mix_switches_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.full(801, 10, dtype=np.uint16).tofile(root / "cosmo.bin")
            np.full(801, 20, dtype=np.uint16).tofile(root / "pre.bin")
            np.full(801, 30, dtype=np.uint16).tofile(root / "nemo.bin")
            np.full(101, 40, dtype=np.uint16).tofile(root / "val.bin")
            manifest = TokenManifest(
                config=TokenDataConfig(
                    seq_len=2,
                    batch_size=1,
                    total_train_tokens=800,
                    validation_tokens=100,
                    phase_source_mixes=[
                        {
                            "name": "phase_1",
                            "tokens": 400,
                            "source_mix": {"cosmo": 0.5, "pre": 0.5},
                        },
                        {
                            "name": "phase_2",
                            "tokens": 400,
                            "source_mix": {"pre": 0.25, "nemo": 0.75},
                        },
                    ],
                ),
                shards=[
                    TokenShardSpec("cosmo.bin", "cosmo", "train", 801),
                    TokenShardSpec("pre.bin", "pre", "train", 801),
                    TokenShardSpec("nemo.bin", "nemo", "train", 801),
                    TokenShardSpec("val.bin", "validation", "val", 101),
                ],
            )
            router = TrainingTokenRouter(manifest, root=root)
            first_phase = []
            for example_index in range(20):
                inputs, _ = router.block_at_token_offset(example_index * 2)
                first_phase.append(int(inputs[0]))
            second_phase = []
            for example_index in range(200, 220):
                inputs, _ = router.block_at_token_offset(example_index * 2)
                second_phase.append(int(inputs[0]))
            self.assertIn(10, first_phase)
            self.assertIn(20, first_phase)
            self.assertNotIn(30, first_phase)
            self.assertIn(20, second_phase)
            self.assertIn(30, second_phase)
            self.assertNotIn(10, second_phase)

    def test_phase_curriculum_status_reports_actual_batch_mix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.full(801, 10, dtype=np.uint16).tofile(root / "cosmo.bin")
            np.full(801, 20, dtype=np.uint16).tofile(root / "pre.bin")
            np.full(801, 30, dtype=np.uint16).tofile(root / "nemo.bin")
            np.full(101, 40, dtype=np.uint16).tofile(root / "val.bin")
            manifest = TokenManifest(
                config=TokenDataConfig(
                    seq_len=2,
                    batch_size=4,
                    total_train_tokens=800,
                    validation_tokens=100,
                    phase_source_mixes=[
                        {
                            "name": "phase_1",
                            "tokens": 400,
                            "source_mix": {"cosmo": 0.5, "pre": 0.5},
                        },
                        {
                            "name": "phase_2",
                            "tokens": 400,
                            "source_mix": {"pre": 0.25, "nemo": 0.75},
                        },
                    ],
                ),
                shards=[
                    TokenShardSpec("cosmo.bin", "cosmo", "train", 801),
                    TokenShardSpec("pre.bin", "pre", "train", 801),
                    TokenShardSpec("nemo.bin", "nemo", "train", 801),
                    TokenShardSpec("val.bin", "validation", "val", 101),
                ],
            )
            batcher = TokenBatcher(manifest, root=root)

            phase_1 = batcher.curriculum_status(step=1)
            self.assertEqual(phase_1["stage"], "phase_1")
            self.assertEqual(phase_1["source"], "mixed")
            self.assertEqual(phase_1["source_mix"], {"cosmo": 0.5, "pre": 0.5})
            self.assertEqual(sum(phase_1["last_completed_batch_source_counts"].values()), 4)

            phase_2 = batcher.curriculum_status(step=51)
            self.assertEqual(phase_2["stage"], "phase_2")
            self.assertEqual(phase_2["source"], "mixed")
            self.assertEqual(phase_2["source_mix"], {"pre": 0.25, "nemo": 0.75})
            self.assertEqual(sum(phase_2["last_completed_batch_source_counts"].values()), 4)


if __name__ == "__main__":
    unittest.main()
