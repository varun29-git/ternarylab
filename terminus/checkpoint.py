from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

from .integrity import file_sha256, fingerprint, fsync_directory, write_json_atomic


SCHEMA_VERSION = 1


def rng_state():
    numpy_state = np.random.get_state()
    state = {"python": random.getstate(), "torch": torch.get_rng_state(),
             "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint CUDA RNG cannot be restored on CPU")
        torch.cuda.set_rng_state(state["cuda"].cpu())


@torch.no_grad()
def validate_finite_state(payload):
    pending = [payload]
    checks = {}
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            checks.setdefault(value.device, []).append(torch.isfinite(value).all())
    if any(not torch.stack(values).all().item() for values in checks.values()):
        raise FloatingPointError("refusing to commit nonfinite model or optimizer state")


def save_checkpoint(directory, model, optimizer, state, contract, rank_rng):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{state['updates']:08d}.pt"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    temporary = directory / f".{uuid.uuid4().hex}.tmp"
    payload = {"schema_version": SCHEMA_VERSION, "contract": contract, "state": state,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rank_rng": rank_rng}
    validate_finite_state((payload["model"], payload["optimizer"]))
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {"schema_version": SCHEMA_VERSION, "file": path.name, "sha256": file_sha256(path),
               "size_bytes": path.stat().st_size, "contract_sha256": fingerprint(contract), "state": state}
    write_json_atomic(path.with_suffix(".json"), receipt)
    write_json_atomic(directory / "latest.json", receipt)
    return path


def load_checkpoint(path, expected_contract):
    path = Path(path)
    if path.is_dir():
        receipt = json.loads((path / "latest.json").read_text())
        name = receipt["file"]
        if Path(name).name != name:
            raise ValueError("invalid checkpoint path in receipt")
        path = path / name
    else:
        receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["size_bytes"] != path.stat().st_size or receipt["sha256"] != file_sha256(path):
        raise ValueError("checkpoint integrity verification failed")
    if receipt["contract_sha256"] != fingerprint(expected_contract):
        raise ValueError("resume contract changed; use a deliberate fresh run instead")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("contract") != expected_contract:
        raise ValueError("checkpoint schema or resume contract mismatch")
    if payload["state"] != receipt["state"]:
        raise ValueError("checkpoint counters disagree with the committed receipt")
    return payload
