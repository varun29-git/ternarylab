from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

from .integrity import LANGUAGES, file_sha256, write_json_atomic


def normalize_text(text):
    if not isinstance(text, str) or "\x00" in text or any(0xD800 <= ord(character) <= 0xDFFF for character in text):
        raise ValueError("text must be valid Unicode without NUL or unpaired surrogates")
    return unicodedata.normalize("NFC", text)


def load_tokenizer(path):
    import sentencepiece as sentencepiece
    from sentencepiece import sentencepiece_model_pb2

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(tokenizer.serialized_model_proto())
    if proto.normalizer_spec.name != "identity" or not proto.trainer_spec.byte_fallback:
        raise ValueError("tokenizer must use identity normalization plus byte fallback; NFC is applied by this wrapper")
    if proto.normalizer_spec.remove_extra_whitespaces or proto.normalizer_spec.add_dummy_prefix:
        raise ValueError("tokenizer must preserve whitespace without a dummy prefix")
    if tokenizer.eos_id() < 0:
        raise ValueError("tokenizer must define EOS")
    return tokenizer


def tokenizer_contract(tokenizer, path):
    return {"path": str(path), "sha256": file_sha256(path), "vocab_size": tokenizer.get_piece_size(),
            "eos_token_id": tokenizer.eos_id(), "bos_token_id": tokenizer.bos_id(),
            "pad_token_id": None if tokenizer.pad_id() < 0 else tokenizer.pad_id(), "normalization": "NFC"}


def encode_text(tokenizer, text):
    normalized = normalize_text(text)
    tokens = tokenizer.encode(normalized, out_type=int)
    if any(token < 0 or token >= tokenizer.get_piece_size() for token in tokens):
        raise ValueError("tokenizer returned an out-of-range ID")
    if tokenizer.unk_id() in tokens or tokenizer.decode(tokens) != normalized:
        raise ValueError("tokenizer must round-trip normalized Unicode without unknown IDs")
    return tokens


def records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record.get("language") not in LANGUAGES or not isinstance(record.get("text"), str):
                raise ValueError(f"invalid language/text at line {line_number}")
            yield record


def train_tokenizer(corpus, output_prefix, *, vocab_size=96000, sentences_per_language=10000, seed=1337, model_type="unigram"):
    import sentencepiece as sentencepiece

    if sentences_per_language <= 0 or vocab_size < 300:
        raise ValueError("invalid tokenizer sampling/vocabulary budget")
    prefix = Path(output_prefix)
    if Path(str(prefix) + ".model").exists() or Path(str(prefix) + ".vocab").exists():
        raise FileExistsError("refusing to overwrite a tokenizer")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    generator = random.Random(seed)
    sampled, seen = defaultdict(list), defaultdict(int)
    for record in records(corpus):
        if record.get("split", "train") != "train":
            continue
        language = record["language"]
        text = normalize_text(record["text"])
        if not text.strip() or len(text.encode("utf-8")) > 16384:
            continue
        seen[language] += 1
        if len(sampled[language]) < sentences_per_language:
            sampled[language].append(text)
        else:
            index = generator.randrange(seen[language])
            if index < sentences_per_language:
                sampled[language][index] = text
    if set(sampled) != set(LANGUAGES):
        raise ValueError("tokenizer training corpus must cover all eleven languages")
    sentences = [text for language in LANGUAGES for text in sampled[language]]
    generator.shuffle(sentences)
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences), model_prefix=str(prefix), vocab_size=vocab_size,
        model_type=model_type, normalization_rule_name="identity", character_coverage=1.0,
        byte_fallback=True, remove_extra_whitespaces=False, add_dummy_prefix=False,
        pad_id=0, eos_id=1, bos_id=2, unk_id=3, pad_piece="<pad>", eos_piece="<eos>",
        bos_piece="<bos>", unk_piece="<unk>", shuffle_input_sentence=False,
        num_threads=1, max_sentence_length=16384, hard_vocab_limit=True, minloglevel=1,
    )
    tokenizer = load_tokenizer(str(prefix) + ".model")
    receipt = tokenizer_contract(tokenizer, str(prefix) + ".model")
    receipt.update({"corpus_sha256": file_sha256(corpus), "seed": seed, "model_type": model_type,
                    "sampled_sentences": {language: len(sampled[language]) for language in LANGUAGES}})
    write_json_atomic(str(prefix) + ".json", receipt)
    return receipt


def audit_tokenizer(path, corpus):
    tokenizer = load_tokenizer(path)
    statistics = defaultdict(lambda: {"documents": 0, "tokens": 0, "utf8_bytes": 0, "codepoints": 0, "byte_fallback_tokens": 0})
    for record in records(corpus):
        text = normalize_text(record["text"])
        tokens = encode_text(tokenizer, text)
        stats = statistics[record["language"]]
        stats["documents"] += 1
        stats["tokens"] += len(tokens)
        stats["utf8_bytes"] += len(text.encode("utf-8"))
        stats["codepoints"] += len(text)
        stats["byte_fallback_tokens"] += sum(tokenizer.is_byte(token) for token in tokens)
    for stats in statistics.values():
        stats["bytes_per_token"] = stats["utf8_bytes"] / max(1, stats["tokens"])
        stats["codepoints_per_token"] = stats["codepoints"] / max(1, stats["tokens"])
        stats["byte_fallback_fraction"] = stats["byte_fallback_tokens"] / max(1, stats["tokens"])
    return {"tokenizer": tokenizer_contract(tokenizer, path), "languages": dict(statistics),
            "missing_languages": sorted(set(LANGUAGES) - set(statistics)), "corpus_sha256": file_sha256(corpus)}


def main():
    parser = argparse.ArgumentParser(description="Train or audit a local NFC/byte-fallback multilingual tokenizer")
    parser.add_argument("mode", choices=["train", "audit"])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--vocab-size", type=int, default=96000)
    parser.add_argument("--sentences-per-language", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram")
    args = parser.parse_args()
    if args.mode == "train":
        train_tokenizer(args.corpus, args.output, vocab_size=args.vocab_size,
                        sentences_per_language=args.sentences_per_language, seed=args.seed, model_type=args.model_type)
    else:
        if not args.tokenizer:
            parser.error("audit requires --tokenizer")
        write_json_atomic(args.output, audit_tokenizer(args.tokenizer, args.corpus))


if __name__ == "__main__":
    main()
