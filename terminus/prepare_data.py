from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import TokenDataConfig, TokenManifest, TokenShardSpec, TrainingTokenRouter
from .integrity import file_sha256, write_json_atomic
from .tokenizer import encode_text, load_tokenizer, normalize_text, records, tokenizer_contract


def prepare_data(corpus, tokenizer_path, output_dir, *, source_mix, total_train_tokens,
                 seq_len=4096, batch_size=1, shard_tokens=10_000_000):
    if shard_tokens <= 0:
        raise ValueError("shard token budget must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("data preparation requires an empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_path)
    shutil.copyfile(tokenizer_path, output_dir / "tokenizer.model")
    with (output_dir / "tokenizer.model").open("rb") as handle:
        os.fsync(handle.fileno())
    dtype = np.uint16 if tokenizer.get_piece_size() <= 65536 else np.uint32
    token_dtype = np.dtype(dtype).name
    database = sqlite3.connect(output_dir / "document_hashes.sqlite")
    database.execute("CREATE TABLE documents (digest TEXT PRIMARY KEY, split TEXT NOT NULL)")
    streams, counts, indices = {}, defaultdict(int), defaultdict(int)
    totals, documents = defaultdict(int), defaultdict(int)
    shards, licenses = [], {}
    duplicate_documents = 0

    def finish_shard(key):
        handle = streams.pop(key, None)
        if handle is None:
            return
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        language, split = key
        path = output_dir / f"{language}.{split}.{indices[key]:05d}.bin"
        shards.append(TokenShardSpec(path.name, language, split, counts[key], dtype=token_dtype,
                                    language=language, sha256=file_sha256(path)))
        indices[key] += 1
        counts[key] = 0

    try:
        for record in records(corpus):
            language, split = record["language"], record.get("split")
            if split not in {"train", "val"}:
                raise ValueError("every document must have an explicit train/val split")
            source, license_name = record.get("source"), record.get("license")
            if not isinstance(source, str) or not source or not isinstance(license_name, str) or not license_name:
                raise ValueError("document source and source-license provenance are required")
            if source in licenses and licenses[source] != license_name:
                raise ValueError("inconsistent license metadata for a source")
            licenses[source] = license_name
            text = normalize_text(record["text"])
            if not text.strip():
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            previous = database.execute("SELECT split FROM documents WHERE digest = ?", (digest,)).fetchone()
            if previous:
                if previous[0] != split:
                    raise ValueError("exact normalized document appears in both train and validation")
                duplicate_documents += 1
                continue
            database.execute("INSERT INTO documents VALUES (?, ?)", (digest, split))
            key = (language, split)
            documents[key] += 1
            if sum(documents.values()) % 10000 == 0:
                database.commit()
            tokens = np.asarray(encode_text(tokenizer, text) + [tokenizer.eos_id()], dtype=dtype)
            start = 0
            while start < len(tokens):
                if key not in streams:
                    path = output_dir / f"{language}.{split}.{indices[key]:05d}.bin"
                    streams[key] = path.open("xb")
                take = min(len(tokens) - start, shard_tokens - counts[key])
                streams[key].write(tokens[start:start + take].tobytes())
                counts[key] += take
                totals[key] += take
                start += take
                if counts[key] == shard_tokens:
                    finish_shard(key)
        for key in list(streams):
            finish_shard(key)
        database.commit()
        config = TokenDataConfig(seq_len=seq_len, batch_size=batch_size, total_train_tokens=total_train_tokens,
                                 validation_tokens=sum(max(0, count - 1) for (language, split), count in totals.items() if split == "val"),
                                 token_dtype=token_dtype, source_mix=source_mix)
        contract = tokenizer_contract(tokenizer, output_dir / "tokenizer.model")
        contract["path"] = "tokenizer.model"
        provenance = {"normalization": "NFC", "packing": "continuous_eos", "document_dedup": "sha256_nfc",
                      "licenses": licenses, "corpus_sha256": file_sha256(corpus), "duplicate_documents_removed": duplicate_documents,
                      "documents": {f"{language}/{split}": count for (language, split), count in documents.items()}}
        manifest = TokenManifest(config, sorted(shards, key=lambda spec: spec.path), contract, provenance)
        TrainingTokenRouter(manifest, root=output_dir)
        write_json_atomic(output_dir / "manifest.json", asdict(manifest))
        return manifest
    finally:
        for handle in streams.values():
            handle.close()
        database.close()


def main():
    parser = argparse.ArgumentParser(description="Pack split-labelled Unicode JSONL with EOS, provenance and exact-document deduplication")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-mix", required=True, help="JSON file mapping language codes to token-mixture weights")
    parser.add_argument("--total-train-tokens", required=True, type=int)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    prepare_data(args.corpus, args.tokenizer, args.output_dir, source_mix=json.loads(Path(args.source_mix).read_text()),
                 total_train_tokens=args.total_train_tokens, seq_len=args.seq_len, batch_size=args.batch_size, shard_tokens=args.shard_tokens)


if __name__ == "__main__":
    main()
