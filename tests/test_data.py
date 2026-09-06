import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from terminus.data import (TokenBatcher, TokenDataConfig, TokenManifest, TokenShardSpec,
                           TokenShard, TokenSpan, TrainingTokenRouter, PrefetchingTokenLoader)
from terminus.integrity import file_sha256, validate_manifest
from tests.helpers import fixture, tiny_config


class DataTests(unittest.TestCase):
    def test_integer_budgets_and_phase_boundaries_are_explicit(self):
        for options in ({"batch_size": True}, {"total_train_tokens": 16.5}):
            with self.assertRaises(ValueError):
                TokenDataConfig(**options)
        with self.assertRaisesRegex(ValueError, "phase"):
            TokenDataConfig(seq_len=4, total_train_tokens=16,
                            phase_source_mixes=[{"tokens": 5, "source_mix": {"en": 1.0}},
                                                {"tokens": 11, "source_mix": {"en": 1.0}}])

    def test_cross_shard_next_token_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.arange(5, dtype=np.uint16).tofile(root / "one.bin")
            np.arange(5, 12, dtype=np.uint16).tofile(root / "two.bin")
            span = TokenSpan([TokenShardSpec("one.bin", "en", "train", 5), TokenShardSpec("two.bin", "en", "train", 7)], root=root)
            inputs, targets = span.block(1, 4)
            torch.testing.assert_close(inputs, torch.arange(4, 8))
            torch.testing.assert_close(targets, torch.arange(5, 9))

    def test_weighted_router_counts_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, offset in (("en", 0), ("hi", 1000)):
                np.arange(offset, offset + 100, dtype=np.uint16).tofile(root / (name + ".bin"))
            manifest = TokenManifest(TokenDataConfig(seq_len=4, batch_size=2, total_train_tokens=80,
                                      source_mix={"en": .6, "hi": .4}),
                                      [TokenShardSpec(name + ".bin", name, "train", 100) for name in ("en", "hi")])
            router = TrainingTokenRouter(manifest, root=root)
            seen = {"en": [], "hi": []}
            for index in range(20):
                span, offset = router._span_and_offset(index * 4)
                seen[span.shards[0].spec.source].append(offset)
            self.assertEqual(seen["en"], list(range(0, 48, 4)))
            self.assertEqual(seen["hi"], list(range(0, 32, 4)))
            restored = TrainingTokenRouter(manifest, root=root)
            for index in range(10, 20):
                for actual, expected in zip(restored.block_at_token_offset(index * 4), router.block_at_token_offset(index * 4)):
                    torch.testing.assert_close(actual, expected)

    def test_phase_mix_does_not_repeat_source_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("en", "hi"):
                np.arange(100, dtype=np.uint16).tofile(root / (name + ".bin"))
            config = TokenDataConfig(seq_len=4, batch_size=2, total_train_tokens=64,
                       phase_source_mixes=[{"tokens": 32, "source_mix": {"en": .5, "hi": .5}},
                                           {"tokens": 32, "source_mix": {"en": .75, "hi": .25}}])
            manifest = TokenManifest(config, [TokenShardSpec(name + ".bin", name, "train", 100) for name in ("en", "hi")])
            router = TrainingTokenRouter(manifest, root=root)
            positions = set()
            for index in range(16):
                span, offset = router._span_and_offset(index * 4)
                identity = (span.shards[0].spec.source, offset)
                self.assertNotIn(identity, positions)
                positions.add(identity)

    def test_distributed_tail_pads_only_empty_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = fixture(directory, examples=7, batch_size=2)
            batches = [TokenBatcher(manifest, root=directory, rank=rank, world_size=3) for rank in range(3)]
            first = [batcher.train_batch(0) for batcher in batches]
            self.assertEqual(sum(int(targets.ne(-100).sum()) for _, targets in first), 48)
            last = [batcher.train_batch(1) for batcher in batches]
            self.assertEqual(sum(int(targets.ne(-100).sum()) for _, targets in last), 8)
            self.assertTrue(last[1][1].eq(-100).all())
            for batcher in batches:
                with self.assertRaises(StopIteration):
                    batcher.train_batch(2)

    def test_prefetch_exhaustion_and_early_close(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = fixture(directory, examples=3)
            iterator = iter(PrefetchingTokenLoader(manifest, root=directory, prefetch_batches=1))
            self.assertEqual(len(list(iterator)), 2)
            self.assertFalse(iterator.thread.is_alive())
            iterator = iter(PrefetchingTokenLoader(manifest, root=directory, prefetch_batches=1))
            next(iterator)
            iterator.close()
            self.assertFalse(iterator.thread.is_alive())

    def test_worker_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = fixture(directory)
            Path(directory, "train.bin").unlink()
            iterator = iter(PrefetchingTokenLoader(manifest, root=directory))
            with self.assertRaises(FileNotFoundError):
                next(iterator)
            self.assertFalse(iterator.thread.is_alive())

    def test_validation_includes_last_full_sequence_and_partial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = fixture(directory, batch_size=3)
            batches = list(TokenBatcher(manifest, root=directory).val_batches())
            self.assertEqual([inputs.shape[0] for inputs, _ in batches], [3, 1])
            distributed_batches = [list(TokenBatcher(manifest, root=directory, rank=rank, world_size=2).val_batches()) for rank in range(2)]
            self.assertEqual(sum(targets.numel() for batches in distributed_batches for _, targets in batches), 32)

    def test_npy_dtype_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "tokens.npy")
            np.save(path, np.arange(8, dtype=np.float32))
            with self.assertRaises(ValueError):
                TokenShard(TokenShardSpec(path.name, "en", "train", 8), directory)

    def test_checksum_and_token_range_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = fixture(directory)
            report = validate_manifest(manifest, directory, config, development_mode=True)
            self.assertEqual(report["validated_shards"], 2)
            with Path(directory, "train.bin").open("r+b") as handle:
                handle.write(np.array([1000], dtype=np.uint16).tobytes())
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_manifest(manifest, directory, config, development_mode=True)
            shard = replace(manifest.shards[0], sha256=file_sha256(Path(directory, "train.bin")))
            changed = replace(manifest, shards=[shard, manifest.shards[1]])
            with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
                validate_manifest(changed, directory, config, development_mode=True)

    def test_uint16_cannot_store_96k_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = fixture(directory)
            with self.assertRaisesRegex(ValueError, "dtype"):
                validate_manifest(manifest, directory, replace(config, vocab_size=96000), development_mode=True)

    def test_duplicate_validation_file_or_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = fixture(directory)
            duplicate = replace(manifest.shards[0], split="val")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_manifest(replace(manifest, shards=[manifest.shards[0], duplicate]), directory, config, development_mode=True)
            Path(directory, "duplicate.bin").write_bytes(Path(directory, "train.bin").read_bytes())
            duplicate = replace(duplicate, path="duplicate.bin")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_manifest(replace(manifest, shards=[manifest.shards[0], duplicate]), directory, config, development_mode=True)

    def test_production_rejects_missing_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = fixture(directory)
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                validate_manifest(manifest, directory, config)
