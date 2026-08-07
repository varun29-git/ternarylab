from __future__ import annotations

import json


def transformerpp_counts(
    *,
    vocab_size: int = 32_000,
    dim: int = 768,
    n_layers: int = 20,
    n_heads: int = 12,
    n_kv_heads: int = 4,
    ffn_dim: int = 2048,
    alpha_mode: str = "deterministic_grouped_absmean",
    use_subln: bool = True,
    ffn_activation: str = "swiglu",
    use_embedding_projection: bool = False,
) -> dict[str, object]:
    if dim % n_heads != 0:
        raise ValueError("dim must be divisible by n_heads")
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    if ffn_activation not in {"swiglu", "relu2"}:
        raise ValueError("invalid ffn_activation")
    per_layer = {
        "attn.q_proj.weight": dim * dim,
        "attn.k_proj.weight": dim * kv_dim,
        "attn.v_proj.weight": dim * kv_dim,
        "attn.o_proj.weight": dim * dim,
        "ffn.up_proj.weight": dim * ffn_dim,
        "ffn.down_proj.weight": ffn_dim * dim,
    }
    if ffn_activation == "swiglu":
        per_layer["ffn.gate_proj.weight"] = dim * ffn_dim
    muon_per_layer = sum(per_layer.values())
    muon_total = n_layers * muon_per_layer
    embedding_projection_params = 2 * dim * dim if use_embedding_projection else 0
    block_norm_per_layer = 2 * dim
    final_norm_params = dim
    # Q/K normalization stays parameter-free. SubLN uses one learned scalar per site.
    qk_norm_per_layer = 0
    subln_per_layer = 2 if use_subln else 0
    embedding_params = vocab_size * dim
    norm_params = n_layers * (
        block_norm_per_layer + qk_norm_per_layer + subln_per_layer
    ) + final_norm_params
    adamw_total = embedding_params + norm_params
    alpha_trainable_params = n_layers * len(per_layer) if alpha_mode == "learned_log_gain" else 0
    if alpha_mode == "learned_grouped_log_gain":
        group_counts = {name: (count + 128 - 1) // 128 for name, count in per_layer.items()}
        alpha_trainable_params = n_layers * sum(group_counts.values())
    non_alpha_total = muon_total + embedding_projection_params + adamw_total
    total = non_alpha_total + alpha_trainable_params
    return {
        "config": {
            "vocab_size": vocab_size,
            "dim": dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "kv_dim": kv_dim,
            "ffn_dim": ffn_dim,
            "alpha_group_size": 128,
            "alpha_mode": alpha_mode,
            "use_subln": use_subln,
            "qk_norm_mode": "unit_rms",
            "residual_norm_mode": "affine_rms",
            "subln_mode": "single_scale_rms" if use_subln else "disabled",
            "ffn_activation": ffn_activation,
            "use_embedding_projection": use_embedding_projection,
        },
        "per_layer_muon_params": per_layer,
        "muon_per_layer": muon_per_layer,
        "muon_total": muon_total,
        "muon_embedding_projection": embedding_projection_params,
        "adamw": {
            "token_embeddings": embedding_params,
            "block_and_final_norms": n_layers * block_norm_per_layer + final_norm_params,
            "all_layer_qk_norm_gains": n_layers * qk_norm_per_layer,
            "all_layer_subln_gains": n_layers * subln_per_layer,
            "norms": norm_params,
            "total": adamw_total,
        },
        "alpha_trainable_params": alpha_trainable_params,
        "non_alpha_total_params": non_alpha_total,
        "total_params": total,
    }


def main() -> None:
    print(json.dumps(transformerpp_counts(), indent=2))


if __name__ == "__main__":
    main()
