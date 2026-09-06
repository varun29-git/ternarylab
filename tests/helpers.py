import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from terminus.data import TokenDataConfig, TokenManifest, TokenShardSpec
from terminus.integrity import file_sha256
from terminus.model import BitNetLlamaConfig
from terminus.optim import OptimizerConfig
from terminus.train import TrainConfig


torch.set_num_threads(1)


def tiny_config(**changes):
    return replace(BitNetLlamaConfig(vocab_size=32, dim=32, n_layers=2, n_heads=4,
                   n_kv_heads=2, ffn_dim=64, max_seq_len=16, local_attention_window=4,
                   global_attention_every=2, attention_backend="sdpa_reference", loss_chunk_size=7), **changes)


def fixture(root, *, examples=9, batch_size=2, accumulation=2, dropout=0.0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_config(max_seq_len=8, dropout=dropout)
    generator = np.random.default_rng(17)
    train_tokens = generator.integers(0, config.vocab_size, examples * 8 + 1, dtype=np.uint16)
    validation_tokens = generator.integers(0, config.vocab_size, 33, dtype=np.uint16)
    train_tokens.tofile(root / "train.bin")
    validation_tokens.tofile(root / "val.bin")
    shards = [TokenShardSpec(name + ".bin", "en", name, len(tokens), sha256=file_sha256(root / (name + ".bin")), language="en")
              for name, tokens in (("train", train_tokens), ("val", validation_tokens))]
    manifest = TokenManifest(TokenDataConfig(seq_len=8, batch_size=batch_size, total_train_tokens=examples * 8,
                             validation_tokens=32, source_mix={"en": 1.0}), shards)
    manifest.save(root / "manifest.json")
    training = TrainConfig(device="cpu", dtype="fp32", development_mode=True,
                           gradient_accumulation_steps=accumulation, log_interval=1,
                           validation_interval=2, checkpoint_interval=2, validation_batches_per_language=2)
    (root / "config.json").write_text(json.dumps({"model": asdict(config), "training": asdict(training), "optimizer": asdict(OptimizerConfig())}))
    return config, manifest


def checkpoint_payload(directory):
    directory = Path(directory) / "checkpoints"
    receipt = json.loads((directory / "latest.json").read_text())
    return torch.load(directory / receipt["file"], map_location="cpu", weights_only=True)
