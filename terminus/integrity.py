from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import DTYPE_BY_NAME, TokenManifest, TokenShard, TrainingTokenRouter


LANGUAGES = ("en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "or", "pa")


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {path.name: file_sha256(path) for path in sorted(root.glob("*.py"))}


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def validate_manifest(manifest: TokenManifest, root, model_config, *, development_mode=False):
    root = Path(root)
    if manifest.config.seq_len != model_config.max_seq_len:
        raise ValueError("manifest sequence length must equal model context")
    if manifest.config.total_examples == 0:
        raise ValueError("training budget must contain at least one full sequence")
    if manifest.config.train_split == manifest.config.val_split:
        raise ValueError("training and validation splits must be different")
    train = manifest.shards_for(split=manifest.config.train_split)
    validation = manifest.shards_for(split=manifest.config.val_split)
    if not train or not validation:
        raise ValueError("both training and held-out validation shards are required")
    if len(train) + len(validation) != len(manifest.shards):
        raise ValueError("manifest contains an unrecognized split")
    identities, digests, totals = set(), {}, {}
    sources = {}
    for spec in manifest.shards:
        shard = TokenShard(spec, root)
        stat = shard.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise ValueError("duplicate/overlapping shard file in manifest")
        identities.add(identity)
        if not spec.sha256 or file_sha256(shard.path) != spec.sha256.lower():
            raise ValueError(f"shard checksum mismatch or missing: {spec.path}")
        if spec.sha256.lower() in digests:
            raise ValueError("duplicate shard content, including possible train/validation contamination")
        digests[spec.sha256.lower()] = spec.split
        if np.iinfo(DTYPE_BY_NAME[spec.dtype]).max < model_config.vocab_size - 1:
            raise ValueError("token storage dtype cannot represent the complete vocabulary")
        for start in range(0, spec.num_tokens, 4 * 1024 * 1024):
            tokens = shard.tokens[start:start + 4 * 1024 * 1024]
            if int(tokens.min()) < 0 or int(tokens.max()) >= model_config.vocab_size:
                raise ValueError(f"out-of-vocabulary token in {spec.path}; never replace IDs with EOS")
        if spec.language not in LANGUAGES:
            raise ValueError(f"each shard needs an explicit supported language: {spec.path}")
        totals.setdefault(spec.split, {}).setdefault(spec.language, 0)
        totals[spec.split][spec.language] += spec.num_tokens
        if spec.split == manifest.config.train_split:
            previous_language = sources.setdefault(spec.source, spec.language)
            if previous_language != spec.language:
                raise ValueError("each training source must have one language for auditable mixture weights")
    contract = manifest.tokenizer
    if not contract and not development_mode:
        raise ValueError("production requires a frozen tokenizer identity")
    if contract:
        from .tokenizer import load_tokenizer, tokenizer_contract

        actual = tokenizer_contract(load_tokenizer(root / str(contract["path"])), root / str(contract["path"]))
        for name in ("sha256", "vocab_size", "eos_token_id", "bos_token_id", "pad_token_id", "normalization"):
            if contract.get(name) != actual[name]:
                raise ValueError(f"tokenizer contract mismatch: {name}")
        if actual["vocab_size"] != model_config.vocab_size or actual["pad_token_id"] != model_config.pad_token_id:
            raise ValueError("tokenizer and model vocabulary/padding identities differ")
    if not development_mode:
        for split in (manifest.config.train_split, manifest.config.val_split):
            missing = set(LANGUAGES) - set(totals[split])
            if missing:
                raise ValueError(f"missing language coverage in {split}: {sorted(missing)}")
            if any(count < model_config.max_seq_len + 1 for count in totals[split].values()):
                raise ValueError("every language needs a full validation/training sequence")
        mixes = ([manifest.config.source_mix] if manifest.config.source_mix else
                 [phase["source_mix"] for phase in manifest.config.phase_source_mixes])
        if not mixes:
            raise ValueError("production needs explicit language/source mixing, not a sequential language curriculum")
        for mix in mixes:
            if {sources.get(source) for source in mix} != set(LANGUAGES):
                raise ValueError("every curriculum phase must cover English and all ten Indic languages")
        if (manifest.provenance.get("normalization") != "NFC"
                or manifest.provenance.get("packing") != "continuous_eos"
                or manifest.provenance.get("document_dedup") != "sha256_nfc"
                or not manifest.provenance.get("licenses")):
            raise ValueError("production needs preprocessing, document deduplication and source-license provenance")
    router = TrainingTokenRouter(manifest, root=root)
    if not development_mode:
        phases = router.phase_specs or [{"schedule": router.mix_schedule, "start_example": 0,
                                         "end_example": manifest.config.total_examples}]
        for phase in phases:
            exposure = {language: 0 for language in LANGUAGES}
            examples = phase["end_example"] - phase["start_example"]
            for source in set(phase["schedule"]):
                exposure[sources[source]] += router._source_count_before(source, examples, phase["schedule"])
            if any(count == 0 for count in exposure.values()):
                raise ValueError("every phase must actually allocate full training examples to every language")
    return {"manifest_sha256": fingerprint(asdict(manifest)), "tokens_by_language": totals,
            "validated_shards": len(manifest.shards), "tokenizer": contract}
