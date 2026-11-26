from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sample import generate, load_model_and_tokenizer, resolve_device, resolve_paths, set_seed  # noqa: E402


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "A good explanation of why the sky looks blue is",
    "Question: If I leave ice in the sun, what happens?\nAnswer:",
    "The best way to learn programming is",
    "In a surprising discovery, scientists found that",
    "A short recipe for tea:\n1.",
    "The quick brown fox",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a prompt suite against a Terminus Flagship 30B checkpoint")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--prompts-file", default=None, help="Text file with prompts separated by blank lines")
    parser.add_argument("--output-file", default=None, help="Markdown output file")
    parser.add_argument("--json-output-file", default=None, help="JSON output file")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--attention-backend",
        choices=["fa3", "flex", "sdpa_reference"],
        default=None,
    )
    return parser.parse_args()


def load_prompts(path: Path | None) -> List[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"prompt file is empty: {path}")
    prompts = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not prompts:
        raise ValueError(f"prompt file produced no prompts: {path}")
    return prompts


def default_output_paths(run_dir: Path | None, checkpoint: Path) -> tuple[Path, Path]:
    root = run_dir if run_dir is not None else checkpoint.parent
    analysis_dir = root / "analysis"
    stem = checkpoint.stem
    return analysis_dir / f"{stem}_sample_outputs.md", analysis_dir / f"{stem}_sample_outputs.json"


def write_markdown(path: Path, *, prompts: List[str], outputs: List[str], checkpoint_path: Path, run_dir: Path | None, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sample Outputs",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Run dir: `{run_dir}`" if run_dir is not None else "- Run dir: `(not provided)`",
        f"- Device: `{args.device}`",
        f"- Dtype: `{args.dtype}`",
        f"- Max new tokens: `{args.max_new_tokens}`",
        f"- Temperature: `{args.temperature}`",
        f"- Top-k: `{args.top_k}`",
        f"- Top-p: `{args.top_p}`",
        f"- Seed: `{args.seed}`",
        "",
    ]
    for index, (prompt, output) in enumerate(zip(prompts, outputs), start=1):
        lines.extend(
            [
                f"## Prompt {index}",
                "",
                "**Prompt**",
                "",
                "```text",
                prompt,
                "```",
                "",
                "**Output**",
                "",
                "```text",
                output,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def write_json_output(path: Path, *, prompts: List[str], outputs: List[str], checkpoint_path: Path, run_dir: Path | None, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples": [
            {"index": index, "prompt": prompt, "output": output}
            for index, (prompt, output) in enumerate(zip(prompts, outputs), start=1)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint_path, run_dir = resolve_paths(args)
    prompts = load_prompts(Path(args.prompts_file).expanduser().resolve() if args.prompts_file else None)
    output_file, json_output_file = default_output_paths(run_dir, checkpoint_path)
    if args.output_file:
        output_file = Path(args.output_file).expanduser().resolve()
    if args.json_output_file:
        json_output_file = Path(args.json_output_file).expanduser().resolve()

    device = resolve_device(args.device)
    model, tokenizer, _, _ = load_model_and_tokenizer(
        checkpoint_path,
        run_dir=run_dir,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        attention_backend=args.attention_backend,
        device=device,
        dtype=args.dtype,
    )

    outputs: List[str] = []
    for index, prompt in enumerate(prompts, start=1):
        text = generate(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        outputs.append(text)
        print(f"[sample {index}/{len(prompts)}] {prompt[:60]!r}", flush=True)

    write_markdown(output_file, prompts=prompts, outputs=outputs, checkpoint_path=checkpoint_path, run_dir=run_dir, args=args)
    write_json_output(json_output_file, prompts=prompts, outputs=outputs, checkpoint_path=checkpoint_path, run_dir=run_dir, args=args)
    print(f"[written] {output_file}", flush=True)
    print(f"[written] {json_output_file}", flush=True)


if __name__ == "__main__":
    main()
