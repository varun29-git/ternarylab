from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terminus.integrity import file_sha256, write_json_atomic
from terminus.model import BitNetLlama, BitNetLlamaConfig


def load_archived_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def behavior(model_class, config_class):
    torch.manual_seed(17)
    config = config_class(vocab_size=32, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                          ffn_dim=64, max_seq_len=16, local_attention_window=4,
                          global_attention_every=2, attention_backend="sdpa_reference",
                          weight_init="normal", pad_token_id=None)
    model = model_class(config).eval()
    inputs = torch.randint(1, 32, (1, 12))
    with torch.no_grad():
        full = model(inputs)["logits"]
        pieces, cache = [], None
        for index in range(inputs.shape[1]):
            output, cache = model.forward_cached(inputs[:, index:index + 1], cache=cache)
            pieces.append(output["logits"])
        cached_error = float((full - torch.cat(pieces, dim=1)).abs().max())
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[:, 2] = False
        changed = inputs.clone()
        changed[:, 2] = (changed[:, 2] % 31) + 1
        masked_error = float((model(inputs, attention_mask=mask)["logits"][:, 3:] -
                              model(changed, attention_mask=mask)["logits"][:, 3:]).abs().max())
    return {"cached_vs_full_max_absolute_error": cached_error, "masked_key_leak_max_absolute_error": masked_error}


def reproduce():
    torch.set_num_threads(1)
    archive = ROOT / "archive/150m-30btokens-run-original"
    legacy = load_archived_module("terminus_archived_model", archive / "source/model.py")
    legacy_train = load_archived_module("terminus_archived_train", archive / "source/train.py")
    try:
        legacy_train.validate_train_config(legacy_train.TrainConfig())
        config_error = None
    except Exception as error:
        config_error = f"{type(error).__name__}: {error}"
    _, scales = legacy.quantize_weight_fp8_e4m3_2d(torch.full((3, 5), .1), block_size=4)
    torch.manual_seed(17)
    inputs = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(6, 8, requires_grad=True)
    output = legacy.DeepGEMMBitLinearFunction.apply(inputs, weight, 4, 1e-5)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    codes, activation_scales = legacy.quantize_activation_fp8_e4m3(inputs.detach(), block_size=4, eps=1e-5)
    effective_inputs = legacy.dequantize_activation_fp8_e4m3(codes, activation_scales, block_size=4)
    gradient_error = float((weight.grad - gradient.T @ effective_inputs).abs().max())
    manifest = json.loads((archive / "_PROTECTION_LOCK").read_text())["file_manifest"]
    drift = {name: {"recorded": digest, "actual": file_sha256(archive / name) if (archive / name).is_file() else None}
             for name, digest in manifest.items() if not (archive / name).is_file() or file_sha256(archive / name) != digest}
    return {"date": "2026-09-06", "torch": str(torch.__version__), "device": "cpu",
            "archived_behavior": behavior(legacy.BitNetLlama, legacy.BitNetLlamaConfig),
            "active_behavior": behavior(BitNetLlama, BitNetLlamaConfig),
            "archived_default_training_config_error": config_error,
            "archived_partial_block_scales_for_uniform_point_one": scales.tolist(),
            "archived_fp8_ste_weight_gradient_max_absolute_error": gradient_error,
            "archive_lock_drift": drift,
            "scope": "Deterministic local numerical reproductions, not an archived checkpoint evaluation or a CUDA benchmark. The archive is read, never modified."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    report = reproduce()
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
