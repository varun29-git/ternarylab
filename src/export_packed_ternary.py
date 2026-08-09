#!/usr/bin/env python3
"""Export a verified checkpoint into the Terminus packed ternary v2 format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

from model import BitNetLlama, BitNetLlamaConfig
from packed_ternary import (
    DENSE_POLICY_BF16_LARGE,
    DENSE_PRECISION_POLICIES,
    build_packed_model_payload,
    verify_packed_model_payload,
)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def verify_checkpoint_sidecar(checkpoint: Path) -> str:
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not sidecar_path.exists():
        raise RuntimeError(f"checkpoint integrity sidecar is missing: {sidecar_path}")
    sidecar = load_json(sidecar_path)
    if int(sidecar.get("size_bytes", -1)) != checkpoint.stat().st_size:
        raise RuntimeError("checkpoint size does not match sidecar")
    digest = file_sha256(checkpoint)
    if sidecar.get("sha256") != digest:
        raise RuntimeError("checkpoint SHA-256 does not match sidecar")
    return digest


def resolve_model_config(
    checkpoint_payload: Mapping[str, object],
    run_dir: Optional[Path],
) -> Dict[str, object]:
    embedded = checkpoint_payload.get("model_config")
    disk_config = None
    if run_dir is not None:
        path = run_dir / "configs" / "model_config.json"
        if path.exists():
            disk_config = load_json(path)
    if isinstance(embedded, Mapping):
        embedded_dict = dict(embedded)
        if disk_config is not None and disk_config != embedded_dict:
            raise RuntimeError("checkpoint and run-directory model configs disagree")
        return embedded_dict
    if disk_config is not None:
        return disk_config
    raise RuntimeError("model config is absent; refusing to reconstruct from defaults")


def normalize_state_dict(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[10:] if key.startswith("_orig_mod.") else key): value
        for key, value in state.items()
    }


def write_artifact_atomic(output: Path, payload: Mapping[str, object]) -> Dict[str, object]:
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError(f"refusing to overwrite existing packed artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "artifact_type": payload["artifact_type"],
        "format_version": payload["format_version"],
        "run_id": payload["run_id"],
        "checkpoint_step": payload["checkpoint_step"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "artifact_sha256": file_sha256(output),
        "size_bytes": output.stat().st_size,
        "storage_summary": payload["storage_summary"],
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    try:
        sidecar_temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        with sidecar_temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(sidecar_temporary, sidecar)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        sidecar_temporary.unlink(missing_ok=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--dense-precision-policy",
        choices=tuple(sorted(DENSE_PRECISION_POLICIES)),
        default=DENSE_POLICY_BF16_LARGE,
        help=(
            "Inference-only dense tensor policy. The default casts large matrices "
            "such as the tied embedding/head to BF16 while retaining FP32 controls."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    output = args.output.expanduser().resolve()
    checkpoint_sha256 = verify_checkpoint_sidecar(checkpoint)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config_dict = resolve_model_config(checkpoint_payload, run_dir)
    model_config = BitNetLlamaConfig(**model_config_dict)
    model = BitNetLlama(model_config)
    model.load_state_dict(normalize_state_dict(checkpoint_payload["model"]), strict=True)
    model.eval()
    scale_dtype = torch.float32 if args.scale_dtype == "fp32" else torch.bfloat16
    packed_payload = build_packed_model_payload(
        model,
        model_config=asdict(model_config),
        source_checkpoint_sha256=checkpoint_sha256,
        run_id=str(checkpoint_payload.get("run_id", "")),
        checkpoint_step=int(checkpoint_payload.get("next_step", checkpoint_payload.get("step", 0))),
        scale_dtype=scale_dtype,
        dense_precision_policy=args.dense_precision_policy,
    )
    verification = verify_packed_model_payload(packed_payload)
    metadata = write_artifact_atomic(output, packed_payload)
    print(json.dumps({"verification": verification, "artifact": metadata}, indent=2))


if __name__ == "__main__":
    main()
