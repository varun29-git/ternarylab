"""Generate a bounded, exactly 1.5B/32-GPU ablation matrix."""
import argparse
import copy
import json
import math
from pathlib import Path

from .integrity import write_json_atomic
from .model import BitNetLlamaConfig
from .preflight import count_model


def candidates(base):
    cases = {}
    for precision in ("fp8", "bf16"):
        for microbatch in (1, 2, 4, 8):
            item = copy.deepcopy(base)
            item["training"].update(expected_batch_size=microbatch, gradient_accumulation_steps=32 // microbatch)
            if precision == "bf16":
                item["model"].update(linear_backend="torch", quantize_activations=False,
                                     fp8_backward="bf16", fp8_quantizer="torch", fp8_cache_weights=False)
            cases[f"{precision}_micro{microbatch}"] = item
    # Same tokenizer, width, parameter count, six global layers and global batch.
    for layers, ffn in ((24, 8448), (48, 3456)):
        item = copy.deepcopy(base)
        item["model"].update(n_layers=layers, ffn_dim=ffn, global_attention_every=layers // 6,
                             residual_init_scale=1 / math.sqrt(2 * layers))
        cases[f"fp8_layers{layers}"] = item
    for name, section, values in (
        ("fp8_checkpoint", "model", {"gradient_checkpointing": True}),
        ("fp8_chunked_loss", "model", {"loss_backend": "chunked"}),
        ("fp8_eager_elementwise", "model", {"elementwise_backend": "torch"}),
        ("fp8_separate_projections", "model", {"fused_projections": False}),
        ("fp8_bucket32", "training", {"ddp_bucket_cap_mb": 32}),
        ("fp8_bucket256", "training", {"ddp_bucket_cap_mb": 256}),
        ("fp8_dgrad", "model", {"fp8_backward": "dgrad"}),
    ):
        item = copy.deepcopy(base)
        item[section].update(values)
        cases[name] = item
    for item in cases.values():
        counts = count_model(BitNetLlamaConfig(**item["model"]))
        if counts["total"] != 1_500_000_000 or counts["unquantized"] != 0:
            raise ValueError("ablation changed the fixed parameter count or ternary coverage")
        training = item["training"]
        if training["expected_batch_size"] * training["gradient_accumulation_steps"] * 32 * 4096 != 4_194_304:
            raise ValueError("ablation changed the global token batch")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/1.5b_exact_h100_fp8.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cases = candidates(json.loads(Path(args.base).read_text()))
    for name, config in cases.items():
        write_json_atomic(output / f"{name}.json", config)
    write_json_atomic(output / "matrix.json", {
        "cases": list(cases), "warmup_updates": 8, "measured_updates": 64,
        "repetitions": 3, "order_seed": 1337, "global_tokens_per_update": 4_194_304,
        "scope": "Generate only; no H100 measurements made. Randomize order with seed 1337 for each repetition. Native preflight required per distinct model policy; real-data manifests must use the case's per-rank batch size. OOM is a failed candidate, not permission to change batch silently."})
    print(f"Wrote {len(cases)} exact candidates to {output}")


if __name__ == "__main__":
    main()
