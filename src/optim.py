from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Type

import torch


DEFAULT_MUON_IMPORT = "muon:SingleDeviceMuonWithAuxAdam"
MUON_IMPORT_ENV = "BITNET_MUON_IMPORT"
_TRANSFORMER_WEIGHT_SUFFIXES = (
    ".attn.q_proj.weight",
    ".attn.k_proj.weight",
    ".attn.v_proj.weight",
    ".attn.o_proj.weight",
    ".ffn.gate_proj.weight",
    ".ffn.up_proj.weight",
    ".ffn.down_proj.weight",
)
_MUON_EMBEDDING_PROJECTION_SUFFIXES = (
    "embed_rot_in.weight",
    "embed_rot_out.weight",
)
_MUON_ROLE_ORDER = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "o_proj": 3,
    "gate_proj": 4,
    "up_proj": 5,
    "down_proj": 6,
    "embed_rot_in": 7,
    "embed_rot_out": 8,
}


class MuonImportError(RuntimeError):
    pass


def _layer_index_from_name(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
        return int(parts[1])
    return None


def _transformer_group_name_for_param(name: str) -> str:
    role = _muon_role_name_from_param(name)
    layer_index = _layer_index_from_name(name)
    if layer_index is None:
        return f"muon_transformer_{role}"
    return f"muon_transformer_l{layer_index}_{role}"


def _muon_role_name_from_param(name: str) -> str:
    if name.endswith(".weight"):
        name = name[:-len(".weight")]
    return name.rsplit(".", 1)[-1]


def _transformer_group_sort_key(name: str) -> tuple[int, int]:
    if name.startswith("muon_transformer_l"):
        rest = name[len("muon_transformer_l") :]
        layer_str, _, role = rest.partition("_")
        return (0, int(layer_str), _MUON_ROLE_ORDER.get(role, 999))
    if name.startswith("muon_transformer_"):
        role = name[len("muon_transformer_") :]
        return (1, -1, _MUON_ROLE_ORDER.get(role, 999))
    return (2, -1, 999)


@dataclass(frozen=True)
class OptimizerConfig:
    kind: str = "muon"
    muon_lr: float = 3.0e-3
    muon_update_scale: str = "match_rms_adamw"
    adamw_lr: float = 3.0e-3
    alpha_lr: float = 1e-5
    norm_lr: float = 2e-4
    attention_temperature_lr: float = 1e-4
    # Norm gains require their own low-LR AdamW group. Letting them fall into
    # the general no-decay group caused the Terminus Flagship 30B Q/K gain runaway.
    split_norm_parameters: bool = True
    muon_momentum: float = 0.95
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-10
    adamw_weight_decay: float = 0.1
    muon_weight_decay: float = 0.1


def muon_update_scale_for_param(
    param: torch.nn.Parameter,
    mode: str,
) -> float:
    if mode == "none":
        return 1.0
    if mode != "match_rms_adamw":
        raise ValueError(f"unknown muon_update_scale: {mode!r}")
    if param.ndim != 2:
        raise ValueError(f"Muon RMS matching expects a 2D matrix, got shape {tuple(param.shape)}")
    fan_out, fan_in = param.shape[:2]
    return 0.2 * math.sqrt(max(fan_out, fan_in))


def import_custom_muon() -> Type[torch.optim.Optimizer]:
    override = os.environ.get(MUON_IMPORT_ENV)
    candidates = [override] if override else [DEFAULT_MUON_IMPORT, "muon:MuonWithAuxAdam"]
    errors = []

    for import_path in candidates:
        if import_path is None:
            continue
        if ":" not in import_path:
            raise MuonImportError(
                f"{MUON_IMPORT_ENV} must be formatted as 'module:Class', got {import_path!r}"
            )
        module_name, class_name = import_path.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{import_path!r}: import failed: {exc}")
            if override:
                raise MuonImportError(
                    "Custom-kernel Muon is required but could not be imported. "
                    f"Tried {import_path!r}. Install/provide Muon on the training machine, "
                    f"or set {MUON_IMPORT_ENV}='module:Class'."
                ) from exc
            continue
        try:
            optimizer_cls = getattr(module, class_name)
        except AttributeError as exc:
            errors.append(f"{import_path!r}: class missing")
            if override:
                raise MuonImportError(
                    f"Imported {module_name!r}, but it does not define {class_name!r}."
                ) from exc
            continue
        if not issubclass(optimizer_cls, torch.optim.Optimizer):
            errors.append(f"{import_path!r}: not a torch Optimizer")
            if override:
                raise MuonImportError(
                    f"{import_path!r} resolved to {optimizer_cls!r}, which is not a torch Optimizer."
                )
            continue
        return optimizer_cls

    raise MuonImportError(
        "Custom-kernel Muon is required but no compatible optimizer class could be loaded. "
        + " ".join(errors)
    )


def current_muon_import_target() -> str:
    return os.environ.get(MUON_IMPORT_ENV, DEFAULT_MUON_IMPORT)


def alpha_layer_groups(model, config: OptimizerConfig, *, include_use_muon: bool) -> List[Dict[str, object]]:
    buckets: Dict[str, List[torch.nn.Parameter]] = {}
    if not hasattr(model, "named_alpha_parameters"):
        return []
    for _, _, group_name, param in model.named_alpha_parameters():
        buckets.setdefault(group_name, []).append(param)
    groups: List[Dict[str, object]] = []
    ordered_keys = sorted(buckets, key=_transformer_group_sort_key)
    for key in ordered_keys:
        group: Dict[str, object] = {
            "name": key,
            "params": buckets[key],
            "lr": config.alpha_lr,
            "betas": config.adamw_betas,
            "eps": config.adamw_eps,
            "weight_decay": 0.0,
        }
        if include_use_muon:
            group["use_muon"] = False
        groups.append(group)
    return groups


def named_special_parameter_groups(
    model,
    *,
    include_out_proj_in_adamw: bool = False,
    split_norm_parameters: bool = True,
) -> Dict[str, List[torch.nn.Parameter]]:
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    muon_transformer_by_matrix: Dict[str, List[torch.nn.Parameter]] = {}
    transformer_matrices: List[torch.nn.Parameter] = []
    claimed_ids = set()

    for name, param in named_params:
        if name.endswith(_TRANSFORMER_WEIGHT_SUFFIXES) or name.endswith(_MUON_EMBEDDING_PROJECTION_SUFFIXES):
            if include_out_proj_in_adamw:
                transformer_matrices.append(param)
            else:
                muon_transformer_by_matrix.setdefault(
                    _transformer_group_name_for_param(name),
                    [],
                ).append(param)
            claimed_ids.add(id(param))

    alpha_ids = {id(param) for param in model.alpha_parameters()}
    claimed_ids.update(alpha_ids)
    temperature_ids = {
        id(param)
        for _, param in getattr(model, "named_attention_temperature_parameters", lambda: [])()
    }
    claimed_ids.update(temperature_ids)
    norm_ids = set()
    if split_norm_parameters:
        norm_ids = {
            id(param)
            for name, param in named_params
            if name == "norm.weight"
            or name.endswith("_norm.weight")
            or ".norm.weight" in name
        }
        claimed_ids.update(norm_ids)
    adamw_decay = [
        param
        for _, param in named_params
        if id(param) not in claimed_ids and param.ndim >= 2
    ]
    adamw_no_decay = [
        param
        for _, param in named_params
        if id(param) not in claimed_ids and param.ndim < 2
    ]

    groups = {
        "muon": (
            [param for params in muon_transformer_by_matrix.values() for param in params]
            if not include_out_proj_in_adamw
            else []
        ),
        "adamw_decay": adamw_decay,
        "adamw_no_decay": adamw_no_decay,
        "ternary_matrices": transformer_matrices,
        "norm": [param for _, param in named_params if id(param) in norm_ids],
        "attention_temperature": [
            param for _, param in named_params if id(param) in temperature_ids
        ],
    }
    if not include_out_proj_in_adamw:
        groups.update(muon_transformer_by_matrix)
    return groups


def muon_aux_adam_groups(model, config: OptimizerConfig) -> List[Dict[str, object]]:
    special = named_special_parameter_groups(
        model,
        split_norm_parameters=config.split_norm_parameters,
    )
    groups = [
        _muon_group_dict(group_name, special[group_name], config)
        for group_name in sorted(
            (key for key in special if key.startswith("muon_transformer")),
            key=_transformer_group_sort_key,
        )
    ]
    groups.extend(
        [
            {
                "name": "adamw_decay",
                "params": special["adamw_decay"],
                "lr": config.adamw_lr,
                "betas": config.adamw_betas,
                "eps": config.adamw_eps,
                "weight_decay": config.adamw_weight_decay,
                "use_muon": False,
            },
            {
                "name": "adamw_no_decay",
                "params": special["adamw_no_decay"],
                "lr": config.adamw_lr,
                "betas": config.adamw_betas,
                "eps": config.adamw_eps,
                "weight_decay": 0.0,
                "use_muon": False,
            },
            {
                "name": "adamw_norm",
                "params": special["norm"],
                "lr": config.norm_lr,
                "betas": config.adamw_betas,
                "eps": config.adamw_eps,
                "weight_decay": 0.0,
                "use_muon": False,
            },
            {
                "name": "adamw_attention_temperature",
                "params": special["attention_temperature"],
                "lr": config.attention_temperature_lr,
                "betas": config.adamw_betas,
                "eps": config.adamw_eps,
                "weight_decay": 0.0,
                "use_muon": False,
            },
        ]
    )
    groups.extend(alpha_layer_groups(model, config, include_use_muon=True))
    return nonempty_groups(groups)


def _muon_group_dict(
    group_name: str,
    params: Sequence[torch.nn.Parameter],
    config: OptimizerConfig,
) -> Dict[str, object]:
    if len(params) != 1:
        raise ValueError(
            f"Muon RMS matching requires one matrix per Muon group; {group_name!r} has {len(params)} params"
        )
    param = params[0]
    return {
        "name": group_name,
        "params": list(params),
        "lr": config.muon_lr,
        "momentum": config.muon_momentum,
        "weight_decay": config.muon_weight_decay,
        "muon_update_scale": muon_update_scale_for_param(param, config.muon_update_scale),
        "use_muon": True,
    }


def adamw_control_groups(model, config: OptimizerConfig) -> List[Dict[str, object]]:
    special = named_special_parameter_groups(
        model,
        include_out_proj_in_adamw=True,
        split_norm_parameters=config.split_norm_parameters,
    )
    groups = [
        {
            "name": "adamw_ternary_matrices",
            "params": special["ternary_matrices"],
            "lr": config.adamw_lr,
            "betas": config.adamw_betas,
            "eps": config.adamw_eps,
            "weight_decay": config.adamw_weight_decay,
        },
        {
            "name": "adamw_decay",
            "params": special["adamw_decay"],
            "lr": config.adamw_lr,
            "betas": config.adamw_betas,
            "eps": config.adamw_eps,
            "weight_decay": config.adamw_weight_decay,
        },
        {
            "name": "adamw_no_decay",
            "params": special["adamw_no_decay"],
            "lr": config.adamw_lr,
            "betas": config.adamw_betas,
            "eps": config.adamw_eps,
            "weight_decay": 0.0,
        },
        {
            "name": "adamw_norm",
            "params": special["norm"],
            "lr": config.norm_lr,
            "betas": config.adamw_betas,
            "eps": config.adamw_eps,
            "weight_decay": 0.0,
        },
        {
            "name": "adamw_attention_temperature",
            "params": special["attention_temperature"],
            "lr": config.attention_temperature_lr,
            "betas": config.adamw_betas,
            "eps": config.adamw_eps,
            "weight_decay": 0.0,
        },
    ]
    groups.extend(alpha_layer_groups(model, config, include_use_muon=False))
    return nonempty_groups(groups)


def nonempty_groups(groups: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [group for group in groups if len(group.get("params", [])) > 0]


def build_optimizer(model, config: OptimizerConfig | None = None) -> torch.optim.Optimizer:
    config = config or OptimizerConfig()
    if config.kind == "adamw":
        groups = adamw_control_groups(model, config)
        _validate_optimizer_groups(groups, model)
        return torch.optim.AdamW(groups)
    if config.kind != "muon":
        raise ValueError(f"unknown optimizer kind: {config.kind!r}")
    muon_cls = import_custom_muon()
    groups = muon_aux_adam_groups(model, config)
    _validate_optimizer_groups(groups, model)
    optimizer_groups, group_names = prepare_keller_muon_groups(groups)
    optimizer = muon_cls(optimizer_groups)
    attach_group_names(optimizer, group_names)
    return optimizer


def prepare_keller_muon_groups(groups: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    prepared: List[Dict[str, object]] = []
    names: List[str] = []
    for group in groups:
        name = str(group["name"])
        names.append(name)
        if group["use_muon"]:
            scale = float(group.get("muon_update_scale", 1.0))
            if scale <= 0.0:
                raise ValueError(f"Muon update scale must be positive, got {scale} for {name}")
            prepared.append(
                {
                    "params": group["params"],
                    "lr": float(group["lr"]) * scale,
                    "momentum": group["momentum"],
                    "weight_decay": float(group["weight_decay"]) / scale,
                    "use_muon": True,
                }
            )
        else:
            prepared.append(
                {
                    "params": group["params"],
                    "lr": group["lr"],
                    "betas": group["betas"],
                    "eps": group["eps"],
                    "weight_decay": group["weight_decay"],
                    "use_muon": False,
                }
            )
    return prepared, names


def attach_group_names(optimizer: torch.optim.Optimizer, group_names: Sequence[str]) -> None:
    if len(optimizer.param_groups) != len(group_names):
        raise ValueError("optimizer changed the number of parameter groups")
    for group, name in zip(optimizer.param_groups, group_names):
        group["name"] = name


def _validate_optimizer_groups(groups: Sequence[Dict[str, object]], model) -> None:
    param_ids = []
    for group in groups:
        params = group.get("params", [])
        param_ids.extend(id(param) for param in params)
    if len(param_ids) != len(set(param_ids)):
        raise ValueError("optimizer groups contain duplicate parameters")
    grouped_params = sum(param.numel() for group in groups for param in group.get("params", []))
    model_params = sum(param.numel() for param in model.parameters())
    if grouped_params != model_params:
        raise ValueError(
            f"optimizer groups cover {grouped_params:,} params, model has {model_params:,}"
        )


def describe_optimizer_groups(model, config: OptimizerConfig | None = None) -> Dict[str, int]:
    config = config or OptimizerConfig()
    if config.kind == "adamw":
        groups = adamw_control_groups(model, config)
    elif config.kind == "muon":
        groups = muon_aux_adam_groups(model, config)
    else:
        raise ValueError(f"unknown optimizer kind: {config.kind!r}")
    _validate_optimizer_groups(groups, model)
    return {
        str(group["name"]): sum(param.numel() for param in group["params"])
        for group in groups
    }
