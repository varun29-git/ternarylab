"""Compare compression on identical, independently held-out documents."""
import argparse

from .integrity import LANGUAGES, file_sha256, write_json_atomic
from .tokenizer import audit_tokenizer


def compare(candidate, baseline, corpus):
    reports = [audit_tokenizer(path, corpus) for path in (candidate, baseline)]
    # Audit reports retain the tokenizer hash and input-corpus hash.
    candidate_languages, baseline_languages = (report["languages"] for report in reports)
    if set(candidate_languages) != set(LANGUAGES):
        raise ValueError("compression comparison must cover all eleven target languages")
    languages = {}
    for language in LANGUAGES:
        actual, reference = candidate_languages[language], baseline_languages[language]
        if not actual["whitespace_words"] or not actual["tokens"] or not reference["tokens"]:
            raise ValueError("each language needs nonempty held-out text")
        languages[language] = {"candidate": actual, "baseline": reference,
                               "token_reduction_fraction": 1 - actual["tokens"] / reference["tokens"]}
    return {"candidate": reports[0]["tokenizer"], "baseline": reports[1]["tokenizer"],
            "corpus_sha256": file_sha256(corpus), "languages": languages,
            "scope": "Measured only on this corpus. Tokens/whitespace-word is a fertility proxy; graphemes use Unicode extended grapheme clusters. No extrapolation to unmeasured web data."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, compare(args.candidate, args.baseline, args.corpus))


if __name__ == "__main__":
    main()
