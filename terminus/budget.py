"""Exact fixed-cluster arithmetic, never a substitute for measured goodput."""
from __future__ import annotations

import argparse
import json
import math

from .integrity import write_json_atomic
from .preflight import count_model
from .train import load_config


GPUS = 32
COST_RUPEES = 600_000
RUPEES_PER_GPU_HOUR = 96
PRODUCTION_SECONDS = 180 * 3600
TARGET_TOKENS = 1_400_004_149_248
# Dense SXM rates; 2:4 structured-sparsity marketing rates are inapplicable.
BF16_FLOPS = 989_000_000_000_000
FP8_FLOPS = 1_979_000_000_000_000


def flop_accounting(config):
    counts = count_model(config)
    embedding = config.vocab_size * config.dim
    matrix_parameters = counts["ternary"] - (embedding if config.ternary_embeddings else 0)
    length, window = config.max_seq_len, config.local_attention_window
    global_layers = config.n_layers // config.global_attention_every
    global_pairs = length * (length + 1) // 2
    local_pairs = length * window - window * (window - 1) // 2
    pairs = global_layers * global_pairs + (config.n_layers - global_layers) * local_pairs
    attention = 12 * config.dim * pairs / length
    projections = 6 * matrix_parameters
    head = 6 * embedding
    # Additional embedding rotations, if configured, are dense matrices.
    rotations = 12 * config.dim ** 2 if config.use_embedding_projection else 0
    useful = projections + head + attention + rotations
    fp8 = projections if config.linear_backend == "deepgemm" and config.fp8_backward == "full" else 0
    if config.linear_backend == "deepgemm" and config.fp8_backward != "full":
        fp8 = projections * (2 if config.fp8_backward == "dgrad" else 1) / 3
    recompute = (projections + attention) / 3 if config.gradient_checkpointing else 0
    head_recompute = head / 3 if config.loss_backend == "chunked" else 0
    return {"useful_flops_per_token": useful, "fp8_useful_flops_per_token": fp8,
            "bf16_useful_flops_per_token": useful - fp8,
            "checkpoint_recompute_flops_per_token": recompute + head_recompute,
            "scope": "Dense projection/head GEMMs and exact causal attention pairs; excludes quantization, norms, softmax, optimizer, padding and communication. Lookup is not charged a dense GEMM. MFU excludes recomputation."}


def budget_report(model_config, training, measured_tokens_per_second=None):
    counts = count_model(model_config)
    if counts["total"] != 1_500_000_000 or counts["ternary"] != counts["total"]:
        raise ValueError("fixed design requires exactly 1,500,000,000 ternary learned weights")
    if training.expected_world_size != GPUS or training.expected_train_tokens != TARGET_TOKENS:
        raise ValueError("configuration does not enforce the fixed cluster/token budget")
    if not training.expected_batch_size:
        raise ValueError("configuration must enforce an exact per-rank microbatch")
    update_tokens = GPUS * training.expected_batch_size * model_config.max_seq_len * training.gradient_accumulation_steps
    if TARGET_TOKENS % update_tokens or TARGET_TOKENS <= 1_400_000_000_000:
        raise ValueError("token budget must exceed 1.40T with complete updates")
    flops = flop_accounting(model_config)
    required = TARGET_TOKENS / PRODUCTION_SECONDS
    ideal_gpu_seconds_per_token = (flops["fp8_useful_flops_per_token"] / FP8_FLOPS
                                  + flops["bf16_useful_flops_per_token"] / BF16_FLOPS)
    result = {"status": "analytical_only", "counts": counts, "gpus": GPUS, "nodes": 4, "gpus_per_node": 8,
              "total_gpu_hours": COST_RUPEES / RUPEES_PER_GPU_HOUR,
              "total_cluster_hours": COST_RUPEES / (RUPEES_PER_GPU_HOUR * GPUS),
              "allocation_cluster_hours": {"production_including_eval_and_checkpoints": 180,
                                           "setup_and_kernel_compilation": 4, "experiments": 8, "recovery_reserve": 3.3125},
              "target_tokens": TARGET_TOKENS, "tokens_per_update": update_tokens,
              "optimizer_updates": TARGET_TOKENS // update_tokens,
              "microsteps_per_rank": TARGET_TOKENS // update_tokens * training.gradient_accumulation_steps,
              "required_cluster_goodput_tokens_per_second": required,
              "required_gpu_goodput_tokens_per_second": required / GPUS,
              "maximum_amortized_update_seconds": update_tokens / required,
              "token_storage_bytes_uint16": TARGET_TOKENS * 2,
              "flops": flops,
              "required_conventional_bf16_mfu": required * flops["useful_flops_per_token"] / (GPUS * BF16_FLOPS),
              "required_mixed_precision_ideal_time_utilization": required * ideal_gpu_seconds_per_token / GPUS,
              "dense_bf16_peak_flops_per_gpu": BF16_FLOPS, "dense_fp8_peak_flops_per_gpu": FP8_FLOPS,
              "caveat": "Required rates include production validation/checkpoint time. The mixed-precision metric is a serial-roofline proxy, not measured MFU or a convergence claim. Experiment and recovery allocations consume the same hard budget."}
    if measured_tokens_per_second is not None:
        if not math.isfinite(measured_tokens_per_second) or measured_tokens_per_second <= 0:
            raise ValueError("measured goodput must be finite and positive")
        hours = TARGET_TOKENS / measured_tokens_per_second / 3600
        result["projection_from_supplied_goodput"] = {
            "production_cluster_hours": hours, "within_production_allocation": hours <= 180,
            "production_cost_rupees": hours * GPUS * RUPEES_PER_GPU_HOUR,
            "total_cost_with_full_reserves_rupees": (hours + 15.3125) * GPUS * RUPEES_PER_GPU_HOUR,
            "conventional_bf16_mfu": measured_tokens_per_second * flops["useful_flops_per_token"] / (GPUS * BF16_FLOPS),
            "mixed_precision_ideal_time_utilization": measured_tokens_per_second * ideal_gpu_seconds_per_token / GPUS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/1.5b_exact_h100_fp8.json")
    parser.add_argument("--measured-goodput", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    config, training, _ = load_config(args.config)
    result = budget_report(config, training, args.measured_goodput)
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
