from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight Terminus Flagship 30B data and manifest")
    parser.add_argument("--manifest", default="data_manifest.template.json")
    parser.add_argument("--data-root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    root = Path(args.data_root)
    payload = json.loads(manifest_path.read_text())
    problems = []
    for shard in payload["shards"]:
        path = root / shard["path"]
        expected = int(shard["num_tokens"])
        dtype = np.dtype(np.uint16 if shard.get("dtype", "uint16") == "uint16" else shard["dtype"])
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        actual = path.stat().st_size // dtype.itemsize
        status = "OK" if actual == expected else "BAD"
        print(f"{status} {path}: actual={actual:,} expected={expected:,}")
        if actual != expected:
            problems.append(f"{path} has {actual:,} tokens, expected {expected:,}")
    if problems:
        raise SystemExit("\n".join(problems))
    print("preflight data ok")


if __name__ == "__main__":
    main()
