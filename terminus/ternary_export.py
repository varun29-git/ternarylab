"""Packed ternary inference reference; no low-bit GPU speedup is claimed.

Five base-3 codes occupy one byte: 1.6 physical bits/weight, plus block
scales, close to the log2(3)=1.58496 information bound. No FP32 latent weights
are stored in the export. Decoding for dense GEMMs is a correctness runtime.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import load_checkpoint
from .integrity import file_sha256, write_json_atomic
from .model import BitNetLlama, BitNetLlamaConfig
from .quantization import expand_scales


def pack_trits(codes):
    if codes.dtype != torch.int8 or not bool(((codes >= -1) & (codes <= 1)).all()):
        raise ValueError("ternary codes must be int8 values in {-1, 0, 1}")
    flat = codes.reshape(-1).to(torch.int16) + 1
    flat = F.pad(flat, (0, (-flat.numel()) % 5), value=1)
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=codes.device)
    return (flat.reshape(-1, 5) * powers).sum(1).to(torch.uint8)


def unpack_trits(packed, shape):
    count = math.prod(shape)
    if packed.dtype != torch.uint8 or packed.ndim != 1 or packed.numel() != (count + 4) // 5:
        raise ValueError("invalid packed ternary shape or dtype")
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=packed.device)
    return ((packed.to(torch.int16)[:, None] // powers) % 3 - 1).to(torch.int8).flatten()[:count].reshape(shape)


@torch.no_grad()
def export_model(model, output):
    counts = model.count_parameters()
    if counts["total"] != counts["ternary"] or model.config.learned_norms or model.config.use_embedding_projection:
        raise ValueError("fully ternary export requires all learned weights to be ternary matrices")
    matrices = {}
    for name, module in model.bitlinear_named_modules():
        if float(module.quantization_strength) != 1:
            raise ValueError("export requires fully enabled QAT")
        if not bool(torch.isfinite(module.weight).all()):
            raise ValueError("nonfinite latent weight")
        matrices[name] = {"shape": list(module.weight.shape), "packed": pack_trits(module.ternary_weight()).cpu(),
                          "scales": module.alpha_values().cpu(), "block_size": module.alpha_group_size,
                          "layout": module.weight_scale_layout}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "terminus-base3-v1", "config": asdict(model.config), "counts": counts, "matrices": matrices}
    with output.open("xb") as handle:
        torch.save(payload, handle)
    receipt = {"sha256": file_sha256(output), "format": payload["format"], "counts": counts,
               "packed_code_bytes": sum(value["packed"].numel() for value in matrices.values()),
               "scale_bytes": sum(value["scales"].numel() * 4 for value in matrices.values()),
               "file_bytes": output.stat().st_size, "latent_weights_included": False}
    write_json_atomic(str(output) + ".json", receipt)
    return receipt


class PackedTernaryMatrix(nn.Module):
    def __init__(self, record):
        super().__init__()
        self.shape, self.block_size, self.layout = record["shape"], record["block_size"], record["layout"]
        self.register_buffer("packed", record["packed"])
        self.register_buffer("scales", record["scales"])

    def quantized_weight(self):
        codes = unpack_trits(self.packed, self.shape)
        return codes.float() * expand_scales(self.scales, *self.shape, self.block_size, self.layout)

    def forward(self, values):
        return F.linear(values, self.quantized_weight().to(values.dtype))


def load_export(path, *, attention_backend="sdpa_reference"):
    path = Path(path)
    receipt = json.loads(Path(str(path) + ".json").read_text())
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError("ternary export checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != "terminus-base3-v1":
        raise ValueError("unsupported ternary export format")
    config = replace(BitNetLlamaConfig(**payload["config"]), linear_backend="torch", fp8_backward="bf16",
                     fp8_quantizer="torch", fp8_cache_weights=False, quantize_activations=False,
                     attention_backend=attention_backend, elementwise_backend="torch")
    with torch.device("meta"):
        model = BitNetLlama(config)
    expected = {name: module for name, module in model.bitlinear_named_modules()}
    if set(expected) != set(payload["matrices"]) or model.count_parameters() != payload["counts"]:
        raise ValueError("export matrix coverage/count mismatch")
    for name, record in payload["matrices"].items():
        if list(expected[name].weight.shape) != record["shape"]:
            raise ValueError("export matrix shape mismatch")
        if record["block_size"] != config.alpha_group_size or record["layout"] != config.weight_scale_layout:
            raise ValueError("export scale policy mismatch")
        unpack_trits(record["packed"], record["shape"])
        if bool((record["packed"] > 242).any()) or not bool(torch.isfinite(record["scales"]).all()) or not bool((record["scales"] > 0).all()):
            raise ValueError("invalid ternary code or scale")
        if record["scales"].shape != expected[name].alpha_values().shape:
            raise ValueError("export scale shape mismatch")
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, PackedTernaryMatrix(record))
    if any(value.is_meta for value in list(model.parameters()) + list(model.buffers())):
        raise ValueError("export left uninitialized model state")
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    contract = json.loads((run / "run.json").read_text())["contract"]
    payload = load_checkpoint(run / "checkpoints", contract)
    with torch.device("meta"):
        model = BitNetLlama(BitNetLlamaConfig(**contract["model"]))
    model.load_state_dict(payload["model"], assign=True)
    del payload
    print(json.dumps(export_model(model, args.output), indent=2))


if __name__ == "__main__":
    main()
