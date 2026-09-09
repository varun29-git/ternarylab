"""Build a real-text throughput fixture, not a production tokenizer or corpus."""
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

from terminus.integrity import file_sha256, write_json_atomic
from terminus.prepare_data import prepare_data
from terminus.tokenizer import encode_text, load_tokenizer, normalize_text, train_tokenizer


LANGUAGE_SPLITS = {"hi": "hin", "bn": "ben", "ta": "tam", "te": "tel", "mr": "mar",
                   "gu": "guj", "kn": "kan", "ml": "mal", "or": "ori", "pa": "pan"}


def fetch_page(task):
    language, offset = task
    english = language == "en"
    params = {"dataset": "HuggingFaceFW/fineweb-edu" if english else "ai4bharat/sangraha",
              "config": "sample-10BT" if english else "verified",
              "split": "train" if english else LANGUAGE_SPLITS[language],
              "offset": offset, "length": 100}
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(params)
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = json.load(response)
            break
        except Exception as error:
            if attempt == 9:
                raise
            delay = 30 if isinstance(error, urllib.error.HTTPError) and error.code == 429 else min(2 ** attempt, 30)
            time.sleep(delay)
    records = []
    for item in payload["rows"]:
        row = item["row"]
        text = normalize_text(row["text"])
        text = text.encode("utf-8")[:16000].decode("utf-8", errors="ignore")
        if not text.strip():
            continue
        records.append({"text": text, "language": language,
                        "split": "val" if item["row_idx"] < 20 else "train",
                        "source": params["dataset"] + "/" + params["config"],
                        "license": "odc-by" if english else "cc-by-4.0",
                        "source_id": row.get("id", row.get("doc_id")), "viewer_row": item["row_idx"]})
    return records


def main():
    root = Path("data/azure_probe")
    root.mkdir(parents=True, exist_ok=True)
    corpus = root / "corpus.jsonl"
    if not corpus.exists():
        tasks = [("en", offset) for offset in range(0, 12000, 100)]
        tasks += [(language, offset) for language in LANGUAGE_SPLITS for offset in range(0, 1000, 100)]
        seen, counts, completed_pages = set(), {}, set()
        partial = corpus.with_suffix(".partial")
        if partial.exists():
            with partial.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    seen.add(hashlib.sha256(row["text"].encode()).hexdigest())
                    key = row["language"] + "/" + row["split"]
                    counts[key] = counts.get(key, 0) + 1
                    completed_pages.add((row["language"], row["viewer_row"] // 100 * 100))
            tasks = [task for task in tasks if task not in completed_pages]
        print(f"Fetching {len(tasks)} remaining pages", flush=True)
        with partial.open("a", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                for page_number, rows in enumerate(executor.map(fetch_page, tasks), 1):
                    for row in rows:
                        digest = hashlib.sha256(row["text"].encode()).hexdigest()
                        if digest in seen:
                            continue
                        seen.add(digest)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        key = row["language"] + "/" + row["split"]
                        counts[key] = counts.get(key, 0) + 1
                    if page_number % 20 == 0:
                        print(f"Downloaded {page_number}/{len(tasks)} pages", flush=True)
        corpus.with_suffix(".partial").rename(corpus)
        write_json_atomic(root / "source_receipt.json", {
            "sources": ["HuggingFaceFW/fineweb-edu/sample-10BT", "ai4bharat/sangraha/verified"],
            "documents": counts, "corpus_sha256": file_sha256(corpus),
            "scope": "Public dataset-viewer rows, each clipped to a valid UTF-8 prefix of at most 16,000 bytes. Exact normalized duplicates removed. A throughput fixture, not randomized quality evidence or a production data selection."})
    tokenizer = root / "tokenizer.model"
    if not tokenizer.exists():
        train_tokenizer(corpus, root / "tokenizer", vocab_size=62402,
                        sentences_per_language=1000, seed=1337, model_type="bpe")
    if not (root / "prepared/manifest.json").exists():
        validated = root / "corpus.validated.jsonl"
        rejected = 0
        if not validated.exists():
            tokenizer_instance = load_tokenizer(tokenizer)
            with corpus.open(encoding="utf-8") as source, validated.open("x", encoding="utf-8") as destination:
                for line in source:
                    row = json.loads(line)
                    try:
                        encode_text(tokenizer_instance, row["text"])
                    except ValueError:
                        rejected += 1
                        continue
                    destination.write(line)
            write_json_atomic(root / "tokenization_receipt.json", {"input_corpus_sha256": file_sha256(corpus),
                              "validated_corpus_sha256": file_sha256(validated), "roundtrip_rejected_documents": rejected})
        mix = json.loads(Path("configs/1.5b_exact_language_mix.json").read_text())
        prepare_data(validated, tokenizer, root / "prepared", source_mix=mix,
                     total_train_tokens=4194304, seq_len=4096, batch_size=4,
                     shard_tokens=4_194_304)
    print("Real-token probe prepared", flush=True)


if __name__ == "__main__":
    main()
