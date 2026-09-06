from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import metadata

import torch


MUON_COMMIT = "f98f1cacc0263b04290753e32be8d498c1efc806"


@dataclass(frozen=True)
class OptimizerConfig:
    kind: str = "adamw"
    adamw_lr: float = 3e-4
    norm_lr: float = 1e-4
    muon_lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.95

    def __post_init__(self):
        if self.kind not in {"adamw", "muon"}:
            raise ValueError("optimizer must be adamw or muon")
        for name in ("adamw_lr", "norm_lr", "muon_lr", "eps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas) or not 0 <= self.momentum < 1:
            raise ValueError("optimizer moment coefficients must lie in [0, 1)")


def muon_update_scale_for_param(parameter):
    if parameter.ndim != 2:
        raise ValueError("Muon requires a matrix")
    return 0.2 * math.sqrt(parameter.shape[1])


def build_optimizer(model, config=None):
    config = config or OptimizerConfig()
    matrices = {id(module.weight) for _, module in model.bitlinear_named_modules()}
    for name in ("embed_rot_in", "embed_rot_out"):
        module = getattr(model, name)
        if hasattr(module, "weight"):
            matrices.add(id(module.weight))
    groups = []
    adam_matrices, norms = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if config.kind == "muon" and id(parameter) in matrices:
            scale = muon_update_scale_for_param(parameter)
            groups.append({"params": [parameter], "lr": config.muon_lr * scale,
                           "weight_decay": config.weight_decay / scale,
                           "momentum": config.momentum, "use_muon": True})
        elif parameter.ndim < 2:
            norms.append(parameter)
        else:
            adam_matrices.append(parameter)
    for parameters, learning_rate, decay in ((adam_matrices, config.adamw_lr, config.weight_decay), (norms, config.norm_lr, 0.0)):
        if parameters:
            group = {"params": parameters, "lr": learning_rate, "weight_decay": decay, "betas": tuple(config.betas), "eps": config.eps}
            if config.kind == "muon":
                group["use_muon"] = False
            groups.append(group)
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(grouped) != len(set(grouped)) or set(grouped) != trainable:
        raise ValueError("optimizer must cover each trainable parameter exactly once")
    if config.kind == "muon":
        distribution = metadata.distribution("muon-optimizer")
        provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
        if provenance.get("vcs_info", {}).get("commit_id") != MUON_COMMIT:
            raise RuntimeError("Muon must match the pinned commit; its shape scaling is part of the optimizer contract")
        from muon import SingleDeviceMuonWithAuxAdam

        optimizer = SingleDeviceMuonWithAuxAdam(groups)
    else:
        use_cuda = all(parameter.is_cuda for parameter in model.parameters())
        optimizer = torch.optim.AdamW(groups, fused=use_cuda, foreach=not use_cuda)
    for index, group in enumerate(optimizer.param_groups):
        group["name"] = f"{config.kind}_{index}"
        group["base_lr"] = group["lr"]
    return optimizer
