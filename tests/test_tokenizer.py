import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from terminus.integrity import LANGUAGES, validate_manifest
from terminus.model import BitNetLlamaConfig
from terminus.prepare_data import prepare_data
from terminus.tokenizer import audit_tokenizer, encode_text, load_tokenizer, normalize_text, tokenizer_contract, train_tokenizer
from tests.helpers import tiny_config


SAMPLES = {
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


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.corpus = cls.root / "corpus.jsonl"
        with cls.corpus.open("w", encoding="utf-8") as handle:
            for language, text in SAMPLES.items():
                for index in range(8):
                    handle.write(json.dumps({"language": language, "text": text + f" {index}", "split": "train",
                                              "source": "synthetic_fixture", "license": "test-only"}, ensure_ascii=False) + "\n")
        cls.prefix = cls.root / "tokenizer"
        train_tokenizer(cls.corpus, cls.prefix, vocab_size=512, sentences_per_language=6, model_type="bpe")
        cls.model_path = cls.root / "tokenizer.model"
        cls.tokenizer = load_tokenizer(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_languages_and_joiners_roundtrip(self):
        texts = list(SAMPLES.values()) + ["क्\u200dष क्\u200cष", "e\u0301", "  def f():\n\treturn  1\n", "<eos> <bos> <pad>", "🙂𑀅"]
        for text in texts:
            with self.subTest(text=text):
                tokens = encode_text(self.tokenizer, text)
                self.assertEqual(self.tokenizer.decode(tokens), normalize_text(text))
                self.assertNotIn(self.tokenizer.eos_id(), tokens)

    def test_normalization_preserves_whitespace_and_joiners(self):
        text = "  क्\u200dष\n\tகி  "
        self.assertEqual(normalize_text(text), text)
        with self.assertRaises(ValueError):
            normalize_text("bad\x00text")
        with self.assertRaises(ValueError):
            normalize_text("bad\ud800")

    def test_audit_uses_actual_byte_piece_identity(self):
        report = audit_tokenizer(self.model_path, self.corpus)
        self.assertEqual(report["missing_languages"], [])
        self.assertEqual(set(report["languages"]), set(LANGUAGES))
        extra = self.root / "unseen.jsonl"
        extra.write_text(json.dumps({"language": "en", "text": "🙂𑀅"}) + "\n")
        result = audit_tokenizer(self.model_path, extra)
        self.assertGreater(result["languages"]["en"]["byte_fallback_tokens"], 0)

    def test_prepare_records_eos_dedup_and_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "input.jsonl"
            rows = []
            for language, text in SAMPLES.items():
                for split in ("train", "val"):
                    rows.append({"language": language, "text": text + " " + split,
                                 "split": split, "source": "fixture", "license": "test-only"})
            rows.append(rows[0])
            corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            mix = {language: (.1 if language == "en" else .09) for language in LANGUAGES}
            manifest = prepare_data(corpus, self.model_path, root / "data", source_mix=mix,
                                     total_train_tokens=88, seq_len=8, shard_tokens=1000)
            self.assertEqual(manifest.provenance["duplicate_documents_removed"], 1)
            self.assertEqual(manifest.config.token_dtype, "uint16")
            self.assertEqual(manifest.tokenizer["vocab_size"], 512)
            for language in LANGUAGES:
                for split in ("train", "val"):
                    shards = [spec for spec in manifest.shards if spec.language == language and spec.split == split]
                    tokens = np.concatenate([np.fromfile(root / "data" / spec.path, dtype=np.uint16) for spec in shards])
                    self.assertEqual(int(tokens[-1]), self.tokenizer.eos_id())
            config = tiny_config(vocab_size=512, pad_token_id=0, max_seq_len=8)
            report = validate_manifest(manifest, root / "data", config)
            self.assertEqual(set(report["tokens_by_language"]["train"]), set(LANGUAGES))
            from dataclasses import replace

            insufficient = replace(manifest, config=replace(manifest.config, total_train_tokens=32))
            with self.assertRaisesRegex(ValueError, "actually allocate"):
                validate_manifest(insufficient, root / "data", config)

    def test_cross_split_document_contamination_fails_before_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "contaminated.jsonl"
            rows = [{"language": "en", "text": "The same document", "split": split, "source": "fixture", "license": "test-only"}
                    for split in ("train", "val")]
            corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "both train and validation"):
                prepare_data(corpus, self.model_path, root / "data", source_mix={"en": 1.}, total_train_tokens=8, seq_len=8)
            self.assertFalse((root / "data/manifest.json").exists())

    def test_dotted_output_prefix_cannot_overwrite_tokenizer(self):
        from terminus.tokenizer import train_tokenizer

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "tokenizer.v1"
            Path(str(prefix) + ".model").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                train_tokenizer("unused.jsonl", prefix)

    def test_tokenizer_identity_is_bound_to_file(self):
        contract = tokenizer_contract(self.tokenizer, self.model_path)
        self.assertEqual(contract["pad_token_id"], 0)
        self.assertEqual(contract["eos_token_id"], 1)
        self.assertEqual(len(contract["sha256"]), 64)
