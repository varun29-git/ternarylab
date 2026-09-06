from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import load_checkpoint, restore_rng_state, rng_state, save_checkpoint
from .data import TokenManifest, TokenSpan, PrefetchingTokenLoader
from .integrity import file_sha256, fingerprint, source_hashes, validate_manifest, write_json_atomic
from .model import BitNetLlama, BitNetLlamaConfig
from .optim import OptimizerConfig, build_optimizer


@dataclass(frozen=True)
class TrainConfig:
    device: str = "cuda"
    dtype: str = "bf16"
    development_mode: bool = False
    seed: int = 1337
    gradient_accumulation_steps: int = 1
    warmup_tokens: int = 0
    final_lr_fraction: float = 0.1
    grad_clip: float = 1.0
    quantization_warmup_tokens: int = 0
    quantization_start_strength: float = 1.0
    prefetch_batches: int = 4
    checkpoint_interval: int = 1000
    validation_interval: int = 1000
    validation_batches_per_language: int = 8
    log_interval: int = 10
    compile: bool = False
    time_limit_seconds: float | None = None

    def __post_init__(self):
        for name in ("development_mode", "compile"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.device not in {"cpu", "cuda"} or self.dtype not in {"fp32", "bf16"}:
            raise ValueError("trainer supports CPU FP32 and CUDA BF16/FP32, not FP16 latent updates")
        if self.device == "cpu" and self.dtype != "fp32":
            raise ValueError("CPU training is an FP32 correctness reference")
        for name in ("gradient_accumulation_steps", "prefetch_batches", "checkpoint_interval", "validation_interval", "validation_batches_per_language", "log_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_tokens", "quantization_warmup_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.final_lr_fraction <= 1 or not 0 <= self.quantization_start_strength <= 1:
            raise ValueError("invalid learning-rate fraction or quantization strength")
        if not math.isfinite(self.grad_clip) or self.grad_clip < 0:
            raise ValueError("grad_clip must be finite and nonnegative; zero disables clipping")
        if self.time_limit_seconds is not None and (not math.isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0):
            raise ValueError("time limit must be finite and positive")


def load_config(path):
    payload = json.loads(Path(path).read_text())
    if set(payload) != {"model", "training", "optimizer"}:
        raise ValueError("configuration must contain exactly model, training and optimizer sections")
    model, training = BitNetLlamaConfig(**payload["model"]), TrainConfig(**payload["training"])
    if model.linear_backend == "deepgemm" and training.compile:
        raise ValueError("native DeepGEMM uses its own JIT; full-model torch.compile is not validated")
    return model, training, OptimizerConfig(**payload["optimizer"])


def schedule_scale(tokens_after_update, total_tokens, warmup_tokens, final_lr_fraction):
    if not 0 <= warmup_tokens < total_tokens:
        raise ValueError("warmup must be shorter than the training budget")
    if warmup_tokens and tokens_after_update <= warmup_tokens:
        return tokens_after_update / warmup_tokens
    progress = min(1.0, max(0.0, (tokens_after_update - warmup_tokens) / (total_tokens - warmup_tokens)))
    return final_lr_fraction + (1 - final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))


@lru_cache(maxsize=1)
def native_source_contract():
    import importlib.util

    result = {}
    suffixes = {".py", ".so", ".pyd", ".dll", ".h", ".hpp", ".cu", ".cuh", ".cpp", ".cubin", ".ptx"}
    for name in ("flash_attn_interface", "flash_attn_3_cuda", "deep_gemm", "triton", "muon"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            result[name] = None
            continue
        if spec.submodule_search_locations:
            result[name] = {str(path.relative_to(root)): file_sha256(path)
                            for location in spec.submodule_search_locations
                            for root in (Path(location),) for path in sorted(root.rglob("*"))
                            if path.is_file() and path.suffix in suffixes}
        else:
            result[name] = {Path(spec.origin).name: file_sha256(spec.origin)}
    return result


def runtime_contract(device):
    import importlib.metadata
    import platform

    packages = {}
    for name in ("numpy", "sentencepiece", "triton", "flash-attn", "muon-optimizer", "deep-gemm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "packages": packages,
            "gpu_memory_bytes": properties.total_memory if properties else None,
            "gpu_multiprocessors": properties.multi_processor_count if properties else None,
            "python": platform.python_version(), "native_sources": native_source_contract(),
            "device": device.type, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None}


def make_resume_contract(model_config, training, optimizer_config, manifest_report, world_size, device):
    training_values = asdict(training)
    for name in ("checkpoint_interval", "validation_interval", "validation_batches_per_language", "log_interval", "prefetch_batches", "time_limit_seconds"):
        training_values.pop(name)
    return {"model": asdict(model_config), "training": training_values,
            "optimizer": json.loads(json.dumps(asdict(optimizer_config))), "data": manifest_report,
            "world_size": world_size, "source": source_hashes(), "runtime": runtime_contract(device)}


def append_event(path, payload):
    payload = {"time_unix": time.time(), **payload}
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
    print(encoded, flush=True)


@torch.no_grad()
def evaluate(model, manifest, root, device, amp_dtype, max_batches, rank, world_size):
    was_training = model.training
    state = rng_state()
    model.eval()
    result = {}
    try:
        languages = sorted({spec.language for spec in manifest.shards_for(split=manifest.config.val_split)})
        for language in languages:
            span = TokenSpan([spec for spec in manifest.shards_for(split=manifest.config.val_split) if spec.language == language], root=root)
            blocks = (span.num_tokens - 1) // manifest.config.seq_len
            selected = min(blocks, max_batches * manifest.config.batch_size * world_size)
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            local_indices = range(rank, selected, world_size)
            for start in range(0, len(local_indices), manifest.config.batch_size):
                rows = [span.block((index * (blocks - 1)) // max(1, selected - 1), manifest.config.seq_len)
                        for index in local_indices[start:start + manifest.config.batch_size]]
                inputs, targets = (torch.stack([row[column] for row in rows]).to(device) for column in (0, 1))
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    output = model(inputs, targets, return_logits=False)
                totals[0] += output["loss_sum"].double()
                totals[1] += output["loss_tokens"].double()
            if world_size > 1:
                distributed.all_reduce(totals)
            loss_sum, tokens = totals.tolist()
            if tokens <= 0 or not math.isfinite(loss_sum):
                raise FloatingPointError(f"invalid validation loss or empty language split: {language}")
            mean = loss_sum / tokens
            result[language] = {"loss": mean, "tokens": int(tokens), "perplexity": math.exp(mean) if mean < 700 else None}
    finally:
        model.train(was_training)
        restore_rng_state(state)
    return result


def collective_checkpoint(directory, model, optimizer, state, contract, rank, world_size):
    local_rng = rng_state()
    states = [None] * world_size if rank == 0 else None
    if world_size > 1:
        distributed.gather_object(local_rng, states, dst=0)
    else:
        states = [local_rng]
    error = [None]
    if rank == 0:
        try:
            save_checkpoint(directory, model, optimizer, state, contract, states)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"checkpoint transaction failed: {error[0]}")


def run_training(config_path, manifest_path, data_root, output_dir, *, resume=None, stop_after_updates=None, preflight_report=None):
    model_config, training, optimizer_config = load_config(config_path)
    if stop_after_updates is not None and stop_after_updates <= 0:
        raise ValueError("stop_after_updates must be positive")
    world_size, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world size")
    if training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(local_rank)
        if training.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported on the selected GPU")
    device = torch.device("cuda", local_rank) if training.device == "cuda" else torch.device("cpu")
    if not training.development_mode and (device.type != "cuda" or training.dtype != "bf16" or model_config.attention_backend == "sdpa_reference"):
        raise ValueError("production requires BF16 autocast and an explicit CUDA attention backend")
    if not training.development_mode:
        from .preflight import validate_preflight_report

        validate_preflight_report(preflight_report, model_config, device)
    if world_size > 1:
        distributed.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=5))
    lock_handle = None
    iterator = None
    old_handlers = {}
    try:
        manifest = TokenManifest.load(manifest_path)
        manifest_report = validate_manifest(manifest, data_root, model_config, development_mode=training.development_mode)
        total_tokens = manifest.config.usable_train_tokens
        if training.warmup_tokens >= total_tokens:
            raise ValueError("warmup token budget must be shorter than the total budget")
        if training.quantization_warmup_tokens > total_tokens:
            raise ValueError("quantization warmup exceeds the total token budget")
        if not training.development_mode and model_config.quantization_strength != 1:
            raise ValueError("production must finish with fully ternary projections")
        if model_config.linear_backend == "deepgemm" and (training.quantization_warmup_tokens or training.quantization_start_strength != 1):
            raise ValueError("native DeepGEMM is not validated for a quantization curriculum")
        contract = make_resume_contract(model_config, training, optimizer_config, manifest_report, world_size, device)
        if world_size > 1:
            contracts = [None] * world_size
            distributed.all_gather_object(contracts, fingerprint(contract))
            if len(set(contracts)) != 1:
                raise ValueError("ranks disagree on data, source, configuration, or runtime identity")
        output_dir = Path(output_dir)
        setup_error = [None]
        if rank == 0:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                lock_handle = (output_dir / ".run.lock").open("a")
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_file = output_dir / "run.json"
                if resume:
                    if run_file.exists() and json.loads(run_file.read_text())["contract_sha256"] != fingerprint(contract):
                        raise ValueError("output directory belongs to an incompatible run")
                elif run_file.exists() or (output_dir / "events.jsonl").exists() or (output_dir / "checkpoints").exists():
                    raise FileExistsError("choose a fresh output directory or resume explicitly")
                write_json_atomic(run_file, {"contract_sha256": fingerprint(contract), "contract": contract})
            except Exception as exc:
                setup_error[0] = f"{type(exc).__name__}: {exc}"
        if world_size > 1:
            distributed.broadcast_object_list(setup_error, src=0)
        if setup_error[0]:
            raise RuntimeError(setup_error[0])
        random.seed(training.seed)
        np.random.seed(training.seed)
        torch.manual_seed(training.seed)
        model = BitNetLlama(model_config).to(device)
        optimizer = build_optimizer(model, optimizer_config)
        state = {"next_micro_step": 0, "tokens_seen": 0, "updates": 0}
        payload = load_checkpoint(resume, contract) if resume else None
        if payload is not None:
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            state = payload["state"]
        amp_dtype = torch.bfloat16 if training.dtype == "bf16" else torch.float32
        forward_model = torch.compile(model) if training.compile else model
        if world_size > 1:
            forward_model = DistributedDataParallel(forward_model, device_ids=[local_rank] if device.type == "cuda" else None,
                                                    broadcast_buffers=False, gradient_as_bucket_view=True)
        if payload is not None:
            if len(payload["rank_rng"]) != world_size:
                raise ValueError("checkpoint rank RNG states do not match world size")
            restore_rng_state(payload["rank_rng"][rank])
            del payload
        else:
            torch.manual_seed(training.seed + rank)
        micro_tokens = manifest.config.tokens_per_step * world_size
        total_micro_steps = math.ceil(manifest.config.total_examples / (manifest.config.batch_size * world_size))
        if state["tokens_seen"] != min(state["next_micro_step"] * micro_tokens, total_tokens):
            raise ValueError("checkpoint token and data-cursor counters disagree")
        if state["next_micro_step"] != min(state["updates"] * training.gradient_accumulation_steps, total_micro_steps):
            raise ValueError("checkpoint update and accumulation counters disagree")
        if state["tokens_seen"] >= total_tokens:
            raise ValueError("training budget is already complete")
        iterator = iter(PrefetchingTokenLoader(manifest, root=data_root, rank=rank, world_size=world_size,
                        start_step=state["next_micro_step"], max_steps=total_micro_steps - state["next_micro_step"],
                        prefetch_batches=training.prefetch_batches, device=device, return_step=True))
        should_stop = [False]

        def request_stop(signum, frame):
            should_stop[0] = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        started = last_log = time.monotonic()
        tokens_at_start = last_log_tokens = state["tokens_seen"]
        if rank == 0:
            append_event(output_dir / "events.jsonl", {"event": "startup", "counts": model.count_parameters(),
                         "world_size": world_size, "state": state, "unused_tail_tokens": manifest.config.total_train_tokens - total_tokens})
        model.train()
        while state["next_micro_step"] < total_micro_steps:
            micro_steps = min(training.gradient_accumulation_steps, total_micro_steps - state["next_micro_step"])
            update_tokens = min(micro_steps * micro_tokens, total_tokens - state["tokens_seen"])
            scale = schedule_scale(state["tokens_seen"] + update_tokens, total_tokens, training.warmup_tokens, training.final_lr_fraction)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            strength = model_config.quantization_strength
            if training.quantization_warmup_tokens:
                progress = min(1.0, (state["tokens_seen"] + update_tokens) / training.quantization_warmup_tokens)
                strength = training.quantization_start_strength + (strength - training.quantization_start_strength) * progress
            model.set_quantization_strength_(strength)
            optimizer.zero_grad(set_to_none=True)
            local_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            for micro_index in range(micro_steps):
                batch_index, inputs, targets = next(iterator)
                if batch_index != state["next_micro_step"] + micro_index:
                    raise RuntimeError("prefetch cursor drift")
                sync_context = forward_model.no_sync() if world_size > 1 and micro_index + 1 < micro_steps else contextlib.nullcontext()
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                        output = forward_model(inputs, targets, return_logits=False)
                        loss = output["loss_sum"] * (world_size / update_tokens)
                    local_loss_sum += output["loss_sum"].detach().double()
                    loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip or float("inf"), error_if_nonfinite=True, foreach=True)
            elapsed = time.monotonic() - started
            stop = should_stop[0] or (training.time_limit_seconds is not None and elapsed >= training.time_limit_seconds)
            summary = torch.stack((local_loss_sum, local_loss_sum.new_tensor(float(stop))))
            if world_size > 1:
                distributed.all_reduce(summary)
            loss_sum, stop_requests = summary.tolist()
            if not math.isfinite(loss_sum):
                raise FloatingPointError("nonfinite global training loss; optimizer update was not applied")
            optimizer.step()
            model.clear_fp8_weight_cache()
            state = {"next_micro_step": state["next_micro_step"] + micro_steps,
                     "tokens_seen": state["tokens_seen"] + update_tokens, "updates": state["updates"] + 1}
            stopped = stop_requests > 0 or (stop_after_updates is not None and state["updates"] >= stop_after_updates)
            complete = state["tokens_seen"] == total_tokens
            if rank == 0 and (state["updates"] % training.log_interval == 0 or stopped or complete):
                now = time.monotonic()
                append_event(output_dir / "events.jsonl", {"event": "train", **state, "loss": loss_sum / update_tokens,
                             "loss_tokens": update_tokens, "grad_norm_before_clip": float(grad_norm), "lr_scale": scale,
                             "quantization_strength": strength, "tokens_per_second": (state["tokens_seen"] - last_log_tokens) / (now - last_log),
                             "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
                last_log, last_log_tokens = now, state["tokens_seen"]
            if state["updates"] % training.validation_interval == 0 or complete:
                results = evaluate(model, manifest, data_root, device, amp_dtype, training.validation_batches_per_language, rank, world_size)
                if rank == 0:
                    append_event(output_dir / "events.jsonl", {"event": "validation", **state, "languages": results})
            if state["updates"] % training.checkpoint_interval == 0 or stopped or complete:
                collective_checkpoint(output_dir / "checkpoints", model, optimizer, state, contract, rank, world_size)
            if stopped or complete:
                if rank == 0:
                    append_event(output_dir / "events.jsonl", {"event": "complete" if complete else "stopped", **state,
                                 "tokens_this_invocation": state["tokens_seen"] - tokens_at_start})
                break
        return state
    finally:
        if iterator is not None:
            iterator.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if lock_handle is not None:
            lock_handle.close()
        if world_size > 1 and distributed.is_initialized():
            distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Token-correct single-device/DDP BitNet pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--preflight-report")
    args = parser.parse_args()
    run_training(args.config, args.manifest, args.data_root, args.output_dir,
                 resume=args.resume, stop_after_updates=args.stop_after_updates, preflight_report=args.preflight_report)


if __name__ == "__main__":
    main()
