from __future__ import annotations

import os
from collections import OrderedDict
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


ATTENTION_BACKEND_ENV = "BITNET_ATTENTION_BACKEND"
VALID_ATTENTION_BACKENDS = {"fa3", "flex", "sdpa_reference"}


class KernelBackendError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def flash_attn3_func() -> Optional[Callable]:
    """Return the Hopper FlashAttention-3 interface used by Terminus Flagship 30B."""

    try:
        import flash_attn_interface
    except Exception:
        return None
    return getattr(flash_attn_interface, "flash_attn_func", None)


@lru_cache(maxsize=1)
def flex_attention_ops() -> Optional[Tuple[Callable, Callable]]:
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception:
        return None
    return flex_attention, create_block_mask


def flash_attention3_available() -> bool:
    return flash_attn3_func() is not None


def flex_attention_available() -> bool:
    return flex_attention_ops() is not None


def kernel_policy() -> Dict[str, object]:
    """Describe Terminus Flagship 30B's strict accelerator-kernel policy."""

    return {
        "attention": {
            "production_backends": ["fa3", "flex"],
            "default_backend": "fa3",
            "global_pattern": "full causal",
            "local_pattern": "causal window passed as window_size=(window-1, 0)",
            "fa3": "primary path for full-causal and fixed-window attention",
            "flex": "explicit alternative for block-sparse or irregular patterns",
            "sdpa_reference": "test/reference only; forbidden as an automatic fallback",
            "silent_fallback": False,
        },
        "optimizer": {
            "muon": "external KellerJordan/Muon import via BITNET_MUON_IMPORT",
            "local_foundations_muon_used": False,
        },
        "custom_kernels": {
            "custom_ternary_matmul": False,
            "custom_rope": False,
            "custom_attention": "external_fa3_or_pytorch_flex",
            "packed_bitnet_inference": False,
        },
        "systems_claim_gates": {
            "packed_ternary_compute_ready": False,
            "packed_checkpoint_ready": False,
            "distributed_alpha_grouping_validated": False,
            "training_memory_is_conventional_latent_optimizer_memory": True,
            "allow_1_58_bit_cost_or_memory_claim": False,
        },
    }


def backend_runtime_summary(requested: Optional[str] = None) -> Dict[str, object]:
    requested = (
        os.environ.get(ATTENTION_BACKEND_ENV, "fa3")
        if requested is None
        else requested
    ).lower()
    return {
        "attention_backend_requested": requested,
        "flash_attention3_available": flash_attention3_available(),
        "flex_attention_available": flex_attention_available(),
        "strict_no_sdpa_fallback": True,
        "kernel_policy": kernel_policy(),
    }


def _is_hopper_tensor(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(tensor.device)
    except Exception:
        return False
    return major >= 9


def assert_backend_usable(requested: str, *, q: torch.Tensor, k: torch.Tensor) -> str:
    """Resolve a requested backend or fail; never silently return SDPA."""

    requested = requested.lower()
    if requested not in VALID_ATTENTION_BACKENDS:
        raise KernelBackendError(f"unknown attention backend: {requested!r}")
    if requested == "sdpa_reference":
        return requested
    if not q.is_cuda or not k.is_cuda:
        raise KernelBackendError(f"{requested} requires CUDA tensors")
    if q.dtype not in {torch.float16, torch.bfloat16} or k.dtype != q.dtype:
        raise KernelBackendError(
            f"{requested} requires matching fp16/bf16 Q/K tensors, got q={q.dtype} k={k.dtype}"
        )
    if q.shape[-1] not in {16, 32, 64, 128, 256}:
        raise KernelBackendError(f"unsupported attention head dimension: {q.shape[-1]}")
    if q.shape[2] % k.shape[2] != 0:
        raise KernelBackendError(
            f"query heads ({q.shape[2]}) must be divisible by KV heads ({k.shape[2]})"
        )
    if requested == "fa3":
        if not _is_hopper_tensor(q):
            raise KernelBackendError("Terminus Flagship 30B FA3 production path requires Hopper-or-newer CUDA hardware")
        if not flash_attention3_available():
            raise KernelBackendError(
                "FlashAttention-3 is required but flash_attn_interface.flash_attn_func is unavailable"
            )
    elif requested == "flex" and not flex_attention_available():
        raise KernelBackendError("FlexAttention is required but unavailable in this PyTorch build")
    return requested


def flash_attention3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    causal: bool,
    window_size: Tuple[int, int],
    softcap: float = 0.0,
) -> torch.Tensor:
    """Run FA3 with an explicit full or local causal window.

    Q/K/V use the external FA3 layout [batch, sequence, heads, head_dim].
    The window argument is mandatory so a local layer can never accidentally
    become full attention.
    """

    func = flash_attn3_func()
    if func is None:
        raise KernelBackendError("flash_attn_interface.flash_attn_func is unavailable")
    kwargs = {
        "causal": bool(causal),
        "window_size": (int(window_size[0]), int(window_size[1])),
        "softcap": float(softcap),
    }
    if dropout_p != 0.0:
        kwargs["dropout_p"] = float(dropout_p)
    try:
        return func(q, k, v, **kwargs)
    except TypeError as exc:
        raise KernelBackendError(
            "installed FA3 interface rejected the required causal/window_size call; "
            "run attention_preflight.py against the pinned Hopper build"
        ) from exc


_FLEX_BLOCK_MASK_CACHE_MAX_SIZE = 32
_FLEX_BLOCK_MASK_CACHE: "OrderedDict[Tuple[str, int, int, int, int, int, int, int], object]" = OrderedDict()


def clear_flex_block_mask_cache() -> None:
    _FLEX_BLOCK_MASK_CACHE.clear()


def flex_block_mask_cache_size() -> int:
    return len(_FLEX_BLOCK_MASK_CACHE)


def _flex_block_mask(
    *,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
    local_window: Optional[int],
    query_position_offset: int = 0,
):
    ops = flex_attention_ops()
    if ops is None:
        raise KernelBackendError("FlexAttention is unavailable")
    _, create_block_mask = ops
    device_index = -1 if device.index is None else int(device.index)
    window_key = -1 if local_window is None else int(local_window)
    key = (
        device.type,
        device_index,
        batch,
        heads,
        q_len,
        kv_len,
        window_key,
        int(query_position_offset),
    )
    cached = _FLEX_BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        _FLEX_BLOCK_MASK_CACHE.move_to_end(key)
        return cached

    if local_window is None:
        def mask_mod(_b, _h, q_idx, kv_idx):
            return q_idx + query_position_offset >= kv_idx
    else:
        window = int(local_window)

        def mask_mod(_b, _h, q_idx, kv_idx):
            distance = q_idx + query_position_offset - kv_idx
            return (distance >= 0) & (distance < window)

    block_mask = create_block_mask(
        mask_mod,
        B=int(batch),
        H=int(heads),
        Q_LEN=int(q_len),
        KV_LEN=int(kv_len),
        device=str(device),
        BLOCK_SIZE=128,
        _compile=True,
    )
    _FLEX_BLOCK_MASK_CACHE[key] = block_mask
    _FLEX_BLOCK_MASK_CACHE.move_to_end(key)
    while len(_FLEX_BLOCK_MASK_CACHE) > _FLEX_BLOCK_MASK_CACHE_MAX_SIZE:
        _FLEX_BLOCK_MASK_CACHE.popitem(last=False)
    return block_mask


def flex_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    local_window: Optional[int],
    query_position_offset: Optional[int] = None,
    causal: bool = True,
    softcap: float = 0.0,
) -> torch.Tensor:
    """Run block-sparse FlexAttention without constructing a dense mask.

    Q/K/V use PyTorch's [batch, heads, sequence, head_dim] layout.
    """

    ops = flex_attention_ops()
    if ops is None:
        raise KernelBackendError("FlexAttention is unavailable")
    flex_attention, _ = ops
    if softcap < 0.0:
        raise KernelBackendError("attention softcap must be non-negative")
    score_mod = None
    if softcap > 0.0:
        cap = float(softcap)

        def score_mod(score, _b, _h, _q_idx, _kv_idx):
            return cap * torch.tanh(score / cap)

    if not causal:
        if local_window is not None:
            raise KernelBackendError("non-causal FlexAttention must not request a local mask")
        return flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            enable_gqa=q.shape[1] != k.shape[1],
        )
    if query_position_offset is None:
        query_position_offset = int(k.shape[2]) - int(q.shape[2])
    if query_position_offset < 0:
        raise KernelBackendError("FlexAttention query_position_offset must be non-negative")
    block_mask = _flex_block_mask(
        batch=int(q.shape[0]),
        heads=int(q.shape[1]),
        q_len=int(q.shape[2]),
        kv_len=int(k.shape[2]),
        device=q.device,
        local_window=local_window,
        query_position_offset=query_position_offset,
    )
    return flex_attention(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=q.shape[1] != k.shape[1],
    )


def sdpa_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    local_window: Optional[int],
    dropout_p: float = 0.0,
    query_position_offset: Optional[int] = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    """Dense correctness reference. Never selected automatically."""

    if q.shape[1] != k.shape[1]:
        repeats = q.shape[1] // k.shape[1]
        k = k[:, :, None, :, :].expand(-1, -1, repeats, -1, -1).reshape(
            k.shape[0], q.shape[1], k.shape[2], k.shape[3]
        )
        v = v[:, :, None, :, :].expand(-1, -1, repeats, -1, -1).reshape(
            v.shape[0], q.shape[1], v.shape[2], v.shape[3]
        )
    if query_position_offset is None:
        query_position_offset = int(k.shape[2]) - int(q.shape[2])
    if query_position_offset < 0:
        raise KernelBackendError("SDPA reference query_position_offset must be non-negative")
    q_pos = torch.arange(q.shape[2], device=q.device) + int(query_position_offset)
    kv_pos = torch.arange(k.shape[2], device=q.device)
    distance = q_pos[:, None] - kv_pos[None, :]
    mask = distance >= 0
    if local_window is not None:
        mask = mask & (distance < int(local_window))
    if softcap < 0.0:
        raise KernelBackendError("attention softcap must be non-negative")
    if softcap == 0.0:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

    # SDPA does not expose a score transform. This deliberately dense oracle
    # implements the exact FA3/Flex soft-cap contract for correctness tests;
    # production paths remain kernel sparse and never enter this branch.
    input_dtype = q.dtype
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    logits = float(softcap) * torch.tanh(logits / float(softcap))
    logits = logits.masked_fill(~mask.view(1, 1, mask.shape[0], mask.shape[1]), float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        probabilities = F.dropout(probabilities, p=dropout_p, training=True)
    return torch.matmul(probabilities, v.float()).to(dtype=input_dtype)


def flex_block_mask_stats(
    *,
    batch: int,
    heads: int,
    seq_len: int,
    device: torch.device,
    local_window: int,
) -> Dict[str, float]:
    """Expose concrete sparsity metadata for the preflight scaling gate."""

    block_mask = _flex_block_mask(
        batch=batch,
        heads=heads,
        q_len=seq_len,
        kv_len=seq_len,
        device=device,
        local_window=local_window,
    )
    sparsity = float(block_mask.sparsity())
    partial_blocks = block_mask.kv_num_blocks
    full_blocks = getattr(block_mask, "full_kv_num_blocks", None)
    active_blocks = (
        partial_blocks
        if full_blocks is None
        else partial_blocks + full_blocks
    )
    return {
        "sparsity_percent": sparsity,
        "max_active_kv_blocks_per_row": float(active_blocks.max().item()),
        "mean_active_kv_blocks_per_row": float(active_blocks.float().mean().item()),
    }
