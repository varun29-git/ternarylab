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


PROBE_SAMPLES = {
    "en": "The weather is pleasant today. Keep every space and digit 123.",
    "hi": "आज मौसम बहुत अच्छा है। नमस्ते दुनिया।",
    "ta": "இன்று வானிலை மிகவும் நல்லது. வணக்கம் உலகம்.",
    "te": "ఈ రోజు వాతావరణం చాలా బాగుంది. నమస్కారం ప్రపంచం.",
    "bn": "আজ আবহাওয়া খুব ভালো। নমস্কার বিশ্ব।",
    "mr": "आज हवामान खूप छान आहे. नमस्कार जग.",
    "gu": "આજે હવામાન ખૂબ સરસ છે. નમસ્તે વિશ્વ.",
    "kn": "ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ. ನಮಸ್ಕಾರ ಜಗತ್ತು.",
    "ml": "ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്. നമസ്കാരം ലോകം.",
    "or": "ଆଜି ପାଗ ବହୁତ ଭଲ ଅଛି। ନମସ୍କାର ବିଶ୍ୱ।",
    "pa": "ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ। ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਦੁਨੀਆ।",
}

SARVAM_TOKENIZER_REPO = "sarvamai/sarvam-1"
SARVAM_VOCAB_SIZE = 68_096


def tokenizer_identity_file(path):
    path = Path(path)
    if path.is_dir():
        for name in ("tokenizer.json", "tokenizer.model"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise ValueError("tokenizer directory must contain tokenizer.json or tokenizer.model")
    return path


class HuggingFaceTokenizer:
    """Directory snapshot of a Hugging Face tokenizer, used for the Sarvam pilot."""

    kind = "huggingface"

    def __init__(self, path):
        from tokenizers import Tokenizer

        path = Path(path)
        json_path = path / "tokenizer.json" if path.is_dir() else path
        if not json_path.is_file():
            raise ValueError("Hugging Face tokenizer directories need tokenizer.json")
        self.path = path if path.is_dir() else json_path.parent
        self._tokenizer = Tokenizer.from_file(str(json_path))
        config = {}
        config_path = self.path / "tokenizer_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        self._eos = self._special_id(config, "eos_token", 2)
        self._bos = self._special_id(config, "bos_token", 1)
        self._unk = self._special_id(config, "unk_token", 0)
        pad = config.get("pad_token")
        self._pad = self._token_id(pad) if pad else -1
        if self._eos < 0:
            raise ValueError("tokenizer must define EOS")

    def _token_id(self, piece):
        if not isinstance(piece, str) or not piece:
            return None
        return self._tokenizer.token_to_id(piece)

    def _special_id(self, config, name, default):
        token_id = self._token_id(config.get(name))
        return default if token_id is None else token_id

    def get_piece_size(self):
        return int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    def eos_id(self):
        return self._eos

    def bos_id(self):
        return self._bos

    def pad_id(self):
        return self._pad

    def unk_id(self):
        return self._unk

    def encode(self, text, out_type=int):
        del out_type
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens):
        return self._tokenizer.decode(list(tokens), skip_special_tokens=False)

    def is_byte(self, token):
        piece = self._tokenizer.id_to_token(int(token))
        return isinstance(piece, str) and piece.startswith("<0x")


def load_sentencepiece_tokenizer(path):
    import sentencepiece as sentencepiece
    from sentencepiece import sentencepiece_model_pb2

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))
    tokenizer.kind = "sentencepiece"
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(tokenizer.serialized_model_proto())
    if proto.normalizer_spec.name != "identity" or not proto.trainer_spec.byte_fallback:
        raise ValueError("tokenizer must use identity normalization plus byte fallback; NFC is applied by this wrapper")
    if proto.normalizer_spec.remove_extra_whitespaces or proto.normalizer_spec.add_dummy_prefix:
        raise ValueError("tokenizer must preserve whitespace without a dummy prefix")
    if tokenizer.eos_id() < 0:
        raise ValueError("tokenizer must define EOS")
    return tokenizer


def load_tokenizer(path):
    path = Path(path)
    if path.is_dir() or path.name == "tokenizer.json":
        return HuggingFaceTokenizer(path)
    return load_sentencepiece_tokenizer(path)


def tokenizer_contract(tokenizer, path):
    identity = tokenizer_identity_file(path)
    return {"path": str(path), "sha256": file_sha256(identity), "vocab_size": tokenizer.get_piece_size(),
            "eos_token_id": tokenizer.eos_id(), "bos_token_id": tokenizer.bos_id(),
            "pad_token_id": None if tokenizer.pad_id() < 0 else tokenizer.pad_id(), "normalization": "NFC",
            "kind": getattr(tokenizer, "kind", "sentencepiece")}


def encode_text(tokenizer, text):
    normalized = normalize_text(text)
    tokens = list(tokenizer.encode(normalized, out_type=int))
    if any(token < 0 or token >= tokenizer.get_piece_size() for token in tokens):
        raise ValueError("tokenizer returned an out-of-range ID")
    unk = tokenizer.unk_id()
    if 0 <= unk and unk in tokens:
        raise ValueError("tokenizer must encode without unknown IDs")
    if getattr(tokenizer, "kind", "sentencepiece") == "sentencepiece" and tokenizer.decode(tokens) != normalized:
        raise ValueError("tokenizer must round-trip normalized Unicode without unknown IDs")
    return tokens


def fetch_huggingface_tokenizer(repo, output, *, revision="main"):
    from huggingface_hub import snapshot_download

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a tokenizer directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=output,
                      allow_patterns=["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"])
    tokenizer = load_tokenizer(output)
    receipt = tokenizer_contract(tokenizer, output)
    receipt.update({"repo": repo, "revision": revision})
    write_json_atomic(output / "receipt.json", receipt)
    return receipt


def write_probe_corpus(path, *, copies=4000):
    if copies <= 0:
        raise ValueError("probe corpus copies must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for copy in range(copies):
            for language in LANGUAGES:
                text = PROBE_SAMPLES[language]
                for split in ("train", "val"):
                    record = {"text": f"{text} {copy} {split}", "language": language, "split": split,
                              "source": f"probe-{language}", "license": "synthetic-throughput-probe"}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def probe_language_mix():
    return {language: (0.1 if language == "en" else 0.09) for language in LANGUAGES}


def records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record.get("language") not in LANGUAGES or not isinstance(record.get("text"), str):
                raise ValueError(f"invalid language/text at line {line_number}")
            yield record


def train_tokenizer(corpus, output_prefix, *, vocab_size=64000, sentences_per_language=10000, seed=1337, model_type="unigram"):
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
    import regex

    tokenizer = load_tokenizer(path)
    statistics = defaultdict(lambda: {"documents": 0, "tokens": 0, "utf8_bytes": 0, "codepoints": 0,
                                      "graphemes": 0, "whitespace_words": 0, "byte_fallback_tokens": 0})
    for record in records(corpus):
        text = normalize_text(record["text"])
        tokens = encode_text(tokenizer, text)
        stats = statistics[record["language"]]
        stats["documents"] += 1
        stats["tokens"] += len(tokens)
        stats["utf8_bytes"] += len(text.encode("utf-8"))
        stats["codepoints"] += len(text)
        stats["graphemes"] += len(regex.findall(r"\X", text))
        stats["whitespace_words"] += len(text.split())
        stats["byte_fallback_tokens"] += sum(tokenizer.is_byte(token) for token in tokens)
    for stats in statistics.values():
        stats["tokens_per_grapheme"] = stats["tokens"] / max(1, stats["graphemes"])
        stats["tokens_per_whitespace_word"] = stats["tokens"] / max(1, stats["whitespace_words"])
        stats["bytes_per_token"] = stats["utf8_bytes"] / max(1, stats["tokens"])
        stats["codepoints_per_token"] = stats["codepoints"] / max(1, stats["tokens"])
        stats["byte_fallback_fraction"] = stats["byte_fallback_tokens"] / max(1, stats["tokens"])
    return {"tokenizer": tokenizer_contract(tokenizer, path), "languages": dict(statistics),
            "missing_languages": sorted(set(LANGUAGES) - set(statistics)), "corpus_sha256": file_sha256(corpus)}


def main():
    parser = argparse.ArgumentParser(description="Train, fetch, or audit a multilingual tokenizer")
    parser.add_argument("mode", choices=["train", "audit", "fetch", "probe-corpus"])
    parser.add_argument("--corpus")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--repo", default=SARVAM_TOKENIZER_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--vocab-size", type=int, default=64000)
    parser.add_argument("--sentences-per-language", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram")
    parser.add_argument("--copies", type=int, default=4000)
    args = parser.parse_args()
    if args.mode == "fetch":
        write_json_atomic(Path(args.output) / "receipt.json", fetch_huggingface_tokenizer(args.repo, args.output, revision=args.revision))
        return
    if args.mode == "probe-corpus":
        write_probe_corpus(args.output, copies=args.copies)
        write_json_atomic(Path(args.output).with_name("language_mix.json"), probe_language_mix())
        return
    if not args.corpus:
        parser.error("train and audit require --corpus")
    if args.mode == "train":
        train_tokenizer(args.corpus, args.output, vocab_size=args.vocab_size,
                        sentences_per_language=args.sentences_per_language, seed=args.seed, model_type=args.model_type)
    else:
        if not args.tokenizer:
            parser.error("audit requires --tokenizer")
        write_json_atomic(args.output, audit_tokenizer(args.tokenizer, args.corpus))


if __name__ == "__main__":
    main()
