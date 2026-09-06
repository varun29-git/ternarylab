from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict

import torch

from .checkpoint import validate_finite_state
from .integrity import fingerprint, source_hashes, write_json_atomic
from .model import BitNetLlama, chunked_cross_entropy_sum, linear_cross_entropy_sum
from .optim import build_optimizer
from .preflight import validate_preflight_report
from .train import load_config, runtime_contract


def loss_memory_benchmark(*, tokens=1024, vocabulary=4096, width=128, chunk_size=64):
    if min(tokens, vocabulary, width, chunk_size) <= 0:
        raise ValueError("benchmark dimensions must be positive")
    torch.manual_seed(1337)
    hidden = torch.randn(tokens, width, requires_grad=True)
    weight = torch.randn(vocabulary, width, requires_grad=True)
    targets = torch.randint(vocabulary, (tokens,))
    excluded = {value.untyped_storage().data_ptr() for value in (hidden, weight, targets)}
    cases, gradients = {}, {}
    for name in ("full", "chunked"):
        saved = {}

        def pack(value):
            storage = value.untyped_storage()
            if storage.data_ptr() not in excluded:
                saved[storage.data_ptr()] = storage.nbytes()
            return value

        started = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            loss = (linear_cross_entropy_sum(hidden, weight, targets) if name == "full" else
                    chunked_cross_entropy_sum(hidden, weight, targets, chunk_size))
        retained_bytes = sum(saved.values())
        loss.backward()
        elapsed = time.perf_counter() - started
        gradients[name] = (hidden.grad.clone(), weight.grad.clone())
        cases[name] = {"loss": float(loss.detach()), "retained_extra_forward_storage_bytes": retained_bytes,
                       "forward_backward_seconds": elapsed}
        hidden.grad = weight.grad = None
    gradient_errors = [float((actual - expected).abs().max()) for actual, expected in zip(gradients["chunked"], gradients["full"])]
    for actual, expected in zip(gradients["chunked"], gradients["full"]):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    return {"device": "cpu", "torch": str(torch.__version__), "tokens": tokens, "vocabulary": vocabulary,
            "width": width, "chunk_size": chunk_size, "cases": cases, "max_gradient_absolute_errors": gradient_errors,
            "scope": "Unique extra tensor storage retained by autograd after the loss forward, excluding shared inputs/weights/targets; NOT peak process/GPU memory. Chunked backward recomputes logits and trades compute for retention."}


def training_benchmark(config_path, *, batch_size=1, warmup_steps=2, measured_steps=5, preflight_report=None):
    if batch_size <= 0 or warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("invalid benchmark batch or step count")
    config, training, optimizer_config = load_config(config_path)
    device = torch.device(training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the configured CUDA runtime is unavailable")
    if not training.development_mode:
        validate_preflight_report(preflight_report, config, device)
    torch.manual_seed(training.seed)
    model = BitNetLlama(config).to(device).train()
    optimizer = build_optimizer(model, optimizer_config)
    forward_model = torch.compile(model) if training.compile else model
    inputs = torch.randint(config.vocab_size, (batch_size, config.max_seq_len + 1), device=device)
    amp_dtype = torch.bfloat16 if training.dtype == "bf16" else torch.float32
    durations, losses, gradient_norms = [], [], []
    tokens_per_update = batch_size * config.max_seq_len * training.gradient_accumulation_steps

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup_steps + measured_steps):
        synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_value = torch.zeros((), device=device)
        for _ in range(training.gradient_accumulation_steps):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                output = forward_model(inputs[:, :-1], inputs[:, 1:], return_logits=False)
                loss = output["loss"] / training.gradient_accumulation_steps
            loss.backward()
            loss_value += loss.detach()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip or float("inf"),
                                                       error_if_nonfinite=True, foreach=True)
        optimizer.step()
        model.clear_fp8_weight_cache()
        synchronize()
        duration = time.perf_counter() - started
        if not math.isfinite(float(loss_value)):
            raise FloatingPointError("nonfinite synthetic benchmark loss")
        if step >= warmup_steps:
            durations.append(duration)
            losses.append(float(loss_value))
            gradient_norms.append(float(gradient_norm))
    validate_finite_state((model.state_dict(), optimizer.state_dict()))
    return {"model_sha256": fingerprint(asdict(config)), "source": source_hashes(), "runtime": runtime_contract(device),
            "counts": model.count_parameters(), "batch_size": batch_size, "tokens_per_update": tokens_per_update,
            "warmup_steps": warmup_steps, "measured_steps": measured_steps, "step_seconds": durations,
            "median_step_seconds": statistics.median(durations), "tokens_per_second": tokens_per_update * measured_steps / sum(durations),
            "losses": losses, "gradient_norms": gradient_norms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "scope": "Single-device synthetic full-model forward/backward/optimizer, including gradient accumulation; excludes data IO, distributed communication, evaluation and checkpoint costs. Not a quality or long-soak result."}


def main():
    parser = argparse.ArgumentParser(description="Measure loss retention or rehearse configured full-model optimizer updates")
    parser.add_argument("mode", choices=["loss-memory", "training"])
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--preflight-report")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.mode == "loss-memory":
        result = loss_memory_benchmark()
    else:
        if not args.config:
            parser.error("training requires --config")
        result = training_benchmark(args.config, batch_size=args.batch_size, warmup_steps=args.warmup_steps,
                                    measured_steps=args.steps, preflight_report=args.preflight_report)
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
