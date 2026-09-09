"""Measure whole-model DDP updates, optionally with the real mmap/H2D path.

Includes optimizer updates and gradient communication. Excludes compilation,
validation and checkpoint writes; use production events for amortized goodput.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .budget import flop_accounting
from .data import PrefetchingTokenLoader, TokenManifest
from .distributed_utils import validate_node_manifest
from .integrity import fingerprint, source_hashes, write_json_atomic
from .model import BitNetLlama
from .optim import build_optimizer
from .preflight import validate_preflight_report
from .train import load_config, runtime_contract


def benchmark(config_path, output, *, manifest_path=None, data_root=".", synthetic=False,
              batch_size=4, warmup=8, steps=64, preflight_report=None, profile_dir=None, allow_unverified=False):
    if (manifest_path is None) == (not synthetic):
        raise ValueError("choose exactly one of a real manifest or explicit synthetic data")
    if min(batch_size, steps) <= 0 or warmup < 0:
        raise ValueError("invalid benchmark dimensions")
    config, training, optimizer_config = load_config(config_path)
    world, rank, local = (int(os.environ.get(key, default)) for key, default in
                          (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    device = torch.device("cuda", local) if training.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if not training.development_mode and not allow_unverified:
        validate_preflight_report(preflight_report, config, device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    iterator = None
    try:
        torch.manual_seed(training.seed)
        model = BitNetLlama(config).to(device).train()
        optimizer = build_optimizer(model, optimizer_config)
        forward = torch.compile(model) if training.compile else model
        if world > 1:
            forward = DDP(forward, device_ids=[local] if device.type == "cuda" else None,
                          gradient_as_bucket_view=True, broadcast_buffers=False,
                          bucket_cap_mb=training.ddp_bucket_cap_mb)
        torch.manual_seed(training.seed + rank)
        total_microsteps = (warmup + steps + (1 if profile_dir else 0)) * training.gradient_accumulation_steps
        manifest_report = None
        if manifest_path:
            manifest = TokenManifest.load(manifest_path)
            if manifest.config.batch_size != batch_size:
                raise ValueError("manifest batch size must equal benchmark batch size")
            if manifest.config.total_examples < total_microsteps * batch_size * world:
                raise ValueError("benchmark requires complete batches without data repetition")
            manifest_report = validate_node_manifest(manifest, data_root, config, development_mode=training.development_mode)
            iterator = iter(PrefetchingTokenLoader(manifest, root=data_root, rank=rank, world_size=world,
                            max_steps=total_microsteps, prefetch_batches=training.prefetch_batches, device=device))
        else:
            # Fixed resident tensors intentionally exclude random generation and
            # IO; this mode cannot establish real-data goodput or quality.
            inputs = torch.randint(config.vocab_size, (batch_size, config.max_seq_len), device=device)
            targets = torch.randint(config.vocab_size, inputs.shape, device=device)
        amp_dtype = torch.bfloat16 if training.dtype == "bf16" else torch.float32
        local_tokens = batch_size * config.max_seq_len * training.gradient_accumulation_steps

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def update():
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for micro in range(training.gradient_accumulation_steps):
                x, y = next(iterator) if iterator is not None else (inputs, targets)
                no_sync = forward.no_sync() if world > 1 and micro + 1 < training.gradient_accumulation_steps else contextlib.nullcontext()
                with no_sync:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                        result = forward(x, y, return_logits=False)
                        loss = result["loss_sum"] / local_tokens
                    loss.backward()
                loss_sum.add_(result["loss_sum"].detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip or float("inf"),
                                                  foreach=True, error_if_nonfinite=True)
            optimizer.step()
            model.clear_fp8_weight_cache()
            return loss_sum / local_tokens, norm

        for _ in range(warmup):
            update()
        synchronize()
        if world > 1:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations, losses, norms = [], [], []
        for _ in range(steps):
            started = time.perf_counter()
            loss, norm = update()
            synchronize()
            durations.append(time.perf_counter() - started)
            losses.append(float(loss))
            norms.append(float(norm))
        if not all(math.isfinite(value) for value in losses + norms):
            raise FloatingPointError("nonfinite benchmark loss or gradient norm")
        local_result = {"rank": rank, "seconds": durations, "losses": losses, "gradient_norms": norms,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        results = [None] * world
        if world > 1:
            dist.all_gather_object(results, local_result)
        else:
            results[0] = local_result
        times = [max(record["seconds"][step] for record in results) for step in range(steps)]
        tokens_per_update = local_tokens * world
        goodput = tokens_per_update * steps / sum(times)
        report = {"schema": 1, "source": source_hashes(), "runtime": runtime_contract(device),
                  "configuration": {"model": asdict(config), "training": asdict(training), "optimizer": asdict(optimizer_config)},
                  "config_sha256": fingerprint({"model": asdict(config), "training": asdict(training), "optimizer": asdict(optimizer_config)}),
                  "counts": model.count_parameters(), "world_size": world, "batch_size_per_rank": batch_size,
                  "tokens_per_update": tokens_per_update, "warmup_updates": warmup, "measured_updates": steps,
                  "tokens_per_second": goodput, "median_update_seconds": statistics.median(times),
                  "p95_update_seconds": sorted(times)[math.ceil(.95 * steps) - 1],
                  "rank_results": results, "manifest": manifest_report, "synthetic": synthetic,
                  "preflight_verified": not allow_unverified,
                  "flops": flop_accounting(config), "scope": __doc__}
        if profile_dir:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True,
                                       on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir, worker_name=f"rank-{rank}")) as profiler:
                update()
                synchronize()
                profiler.step()
        if rank == 0:
            write_json_atomic(output, report)
            print(json.dumps({key: report[key] for key in ("world_size", "tokens_per_second", "median_update_seconds", "p95_update_seconds", "synthetic")}, indent=2))
        return report
    finally:
        if iterator is not None:
            iterator.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--preflight-report")
    parser.add_argument("--profile-dir")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="benchmark only: record throughput when a production numerical receipt is unavailable")
    args = parser.parse_args()
    torch.set_num_threads(1)
    benchmark(args.config, args.output, manifest_path=args.manifest, data_root=args.data_root, synthetic=args.synthetic,
              batch_size=args.batch_size, warmup=args.warmup, steps=args.steps,
              preflight_report=args.preflight_report, profile_dir=args.profile_dir,
              allow_unverified=args.allow_unverified)


if __name__ == "__main__":
    main()
