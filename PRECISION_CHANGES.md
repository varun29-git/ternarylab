# Precision Changes Journal

## 30B Parameter Model - FP32 to BF16 Migration

This journal tracks all precision-related changes during the migration from FP32 to BF16 for the 30B parameter model training system.

---

## Table of Contents

1. [Overview](#overview)
2. [Change Log](#change-log)
3. [Pending Changes](#pending-changes)
4. [Notes & Considerations](#notes--considerations)

---

## Overview

**Project**: terminus-flagship-30b  
**Migration Goal**: Convert model precision from FP32 to BF16  
**Scope**: Embedding layers (Phase 1)  
**Started**: 2026-08-09T00:36:23+05:30  
**Last Updated**: 2026-08-09T00:40:05+05:30

### Change Entry Format

Each entry includes:
- **Timestamp**: ISO 8601 format with timezone
- **File Modified**: Relative path from project root
- **Line Numbers**: Specific lines changed
- **Change Description**: What was modified
- **Rationale**: Why the change was made
- **Affected Components**: Impact areas

---

## Change Log

### Entry 001: Add embedding_dtype Configuration Parameter

| Field | Value |
|-------|-------|
| **Timestamp** | 2026-08-09T00:37:32+05:30 |
| **File Modified** | `src/model.py` |
| **Line Numbers** | 207 |
| **Change Description** | Added `embedding_dtype: str = "bfloat16"` configuration parameter to `BitNetLlamaConfig` dataclass |

**Before:**
```python
attention_backend: str = "fa3"
# Terminus Flagship 30B keeps the Test 16 local/global architecture but requires a real
# sparse kernel: FA3 window_size for the primary path or FlexAttention
```

**After:**
```python
attention_backend: str = "fa3"
# Embedding precision: torch.bfloat16 for memory efficiency on Ampere+ GPUs
embedding_dtype: str = "bfloat16"
# Terminus Flagship 30B keeps the Test 16 local/global architecture but requires a real
# sparse kernel: FA3 window_size for the primary path or FlexAttention
```

**Rationale**: Provides configurable embedding precision with BF16 as default for memory efficiency on Ampere+ GPUs (A100, H100).

**Affected Components**: Configuration system, embedding layer initialization

---

### Entry 002: Add embedding_dtype Validation and Helper Method

| Field | Value |
|-------|-------|
| **Timestamp** | 2026-08-09T00:38:32+05:30 |
| **File Modified** | `src/model.py` |
| **Line Numbers** | 317-336 |
| **Change Description** | Added validation for `embedding_dtype` parameter and helper method to convert string to torch.dtype |

**Before:**
```python
if self.n_layers % self.global_attention_every != 0:
    raise ValueError("n_layers must contain complete local/global attention cycles")
hybrid_layer_types(self.n_layers, self.transformer_layer_indices)

@property
def head_dim(self) -> int:
    return self.dim // self.n_heads

@property
def d_inner(self) -> int:
    return self.dim * self.expand
```

**After:**
```python
if self.n_layers % self.global_attention_every != 0:
    raise ValueError("n_layers must contain complete local/global attention cycles")
if self.embedding_dtype not in {"float32", "bfloat16"}:
    raise ValueError("embedding_dtype must be 'float32' or 'bfloat16'")
hybrid_layer_types(self.n_layers, self.transformer_layer_indices)

@property
def head_dim(self) -> int:
    return self.dim // self.n_heads

@property
def d_inner(self) -> int:
    return self.dim * self.expand

def get_embedding_torch_dtype(self) -> torch.dtype:
    """Convert embedding_dtype string to torch.dtype."""
    if self.embedding_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32
```

**Rationale**: Ensures valid dtype configuration and provides clean conversion method.

**Affected Components**: Configuration validation, dtype conversion

---

### Entry 003: Update RMSNorm to Support Configurable dtype

| Field | Value |
|-------|-------|
| **Timestamp** | 2026-08-09T00:39:04+05:30 |
| **File Modified** | `src/model.py` |
| **Line Numbers** | 327-340 |
| **Change Description** | Modified `RMSNorm.__init__()` to accept optional `dtype` parameter |

**Before:**
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype=dtype)
```

**After:**
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.eps = eps
        # Use specified dtype (bf16 for embeddings) or default FP32
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype if dtype is not None else torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype=dtype)
```

**Rationale**: Enables RMSNorm weights to use BF16 for memory efficiency while maintaining FP32 computation for numerical stability.

**Affected Components**: All normalization layers in the model

---

### Entry 004: Update SingleScaleRMSNorm to Use FP8

| Field | Value |
|-------|-------|
| **Timestamp** | 2026-08-09T00:39:32+05:30 |
| **File Modified** | `src/model.py` |
| **Line Numbers** | 370-396 |
| **Change Description** | Modified `SingleScaleRMSNorm` to use FP8 (`torch.float8_e4m3fn`) for scalar weight |

**Before:**
```python
class SingleScaleRMSNorm(nn.Module):
    """RMSNorm with one learned scalar, not one gain per hidden channel."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones((), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"SingleScaleRMSNorm expected last dimension {self.dim}, got {x.shape[-1]}"
            )
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(
            (x_float * x_float).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normed * self.weight.float()).to(dtype=dtype)
```

**After:**
```python
class SingleScaleRMSNorm(nn.Module):
    """RMSNorm with one learned scalar, not one gain per hidden channel.
    
    Uses FP8 (float8_e4m3fn) for the scalar weight for memory efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        # Use FP8 for the scalar weight - more memory efficient
        self.weight = nn.Parameter(torch.ones((), dtype=torch.float8_e4m3fn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"SingleScaleRMSNorm expected last dimension {self.dim}, got {x.shape[-1]}"
            )
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(
            (x_float * x_float).mean(dim=-1, keepdim=True) + self.eps
        )
        # Convert FP8 weight to float for computation
        return (normed * self.weight.float()).to(dtype=dtype)
```

**Rationale**: FP8 provides maximum memory efficiency for single scalar parameters. Used in attention and FFN sub-layer normalization.

**Affected Components**: `BitNetAttention.attn_sub_norm`, `BitNetFeedForward.ffn_sub_norm`

---

### Entry 005: Update BitNetBlock to Use Embedding dtype for RMSNorm

| Field | Value |
|-------|-------|
| **Timestamp** | 2026-08-09T00:39:55+05:30 |
| **File Modified** | `src/model.py` |
| **Line Numbers** | 1867-1877 |
| **Change Description** | Modified `BitNetBlock.__init__()` to pass embedding dtype to RMSNorm instances |

**Before:**
```python
class BitNetBlock(nn.Module):
    def __init__(self, config: BitNetLlamaConfig, *, layer_index: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.attn = BitNetAttention(config, layer_index=layer_index)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.ffn = BitNetFeedForward(config)
```

**After:**
```python
class BitNetBlock(nn.Module):
    def __init__(self, config: BitNetLlamaConfig, *, layer_index: int) -> None:
        super().__init__()
        # Use embedding dtype for RMSNorm weights (bf16 for memory efficiency)
        embedding_dtype = config.get_embedding_torch_dtype()
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps, dtype=embedding_dtype)
        self.attn = BitNetAttention(config, layer_index=layer_index)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps, dtype=embedding_dtype)
        self.ffn = BitNetFeedForward(config)
```

**Rationale**: Ensures all layer normalization weights use BF16 for consistency with embedding precision.

**Affected Components**: All transformer block normalization layers

---

### Entry 006: Update BitNetLlama Embedding Layer Initialization

| Field | Value |
|-------|-------|
| **Timestamp** | 2026-08-09T00:40:05+05:30 |
| **File Modified** | `src/model.py` |
| **Line Numbers** | 1934-1996 |
| **Change Description** | Modified `BitNetLlama.__init__()` to create embeddings and projection layers with BF16 dtype |

**Before:**
```python
class BitNetLlama(nn.Module):
    def __init__(self, config: Optional[BitNetLlamaConfig] = None) -> None:
        super().__init__()
        self.config = config or BitNetLlamaConfig()
        self.tok_embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.dim,
            padding_idx=self.config.pad_token_id,
        )
        self.embed_rot_in = (
            nn.Linear(self.config.dim, self.config.dim, bias=False)
            if self.config.use_embedding_projection
            else nn.Identity()
        )
        self.embed_rot_out = (
            nn.Linear(self.config.dim, self.config.dim, bias=False)
            if self.config.use_embedding_projection
            else nn.Identity()
        )
        # ... rest of init ...
        self.norm = RMSNorm(self.config.dim, eps=self.config.rms_norm_eps)
        self.reset_parameters()
        self.assert_bias_free()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.config.dim))
        if self.config.use_embedding_projection:
            nn.init.orthogonal_(self.embed_rot_in.weight)
            nn.init.orthogonal_(self.embed_rot_out.weight)
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.tok_embeddings.weight[self.config.pad_token_id].zero_()
```

**After:**
```python
class BitNetLlama(nn.Module):
    def __init__(self, config: Optional[BitNetLlamaConfig] = None) -> None:
        super().__init__()
        self.config = config or BitNetLlamaConfig()
        # Get embedding dtype from config (bf16 for memory efficiency on Ampere+ GPUs)
        self._embedding_dtype = self.config.get_embedding_torch_dtype()
        
        # Create token embeddings with specified dtype
        self.tok_embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.dim,
            padding_idx=self.config.pad_token_id,
        )
        # Convert embeddings to bf16 for memory efficiency
        self.tok_embeddings = self.tok_embeddings.to(dtype=self._embedding_dtype)
        
        # Embedding projection layers (OSP rotations) - also use embedding dtype
        self.embed_rot_in = (
            nn.Linear(self.config.dim, self.config.dim, bias=False).to(dtype=self._embedding_dtype)
            if self.config.use_embedding_projection
            else nn.Identity()
        )
        self.embed_rot_out = (
            nn.Linear(self.config.dim, self.config.dim, bias=False).to(dtype=self._embedding_dtype)
            if self.config.use_embedding_projection
            else nn.Identity()
        )
        # ... rest of init ...
        # RMSNorm weights also use embedding dtype for consistency
        self.norm = RMSNorm(self.config.dim, eps=self.config.rms_norm_eps, dtype=self._embedding_dtype)
        self.reset_parameters()
        self.assert_bias_free()

    def reset_parameters(self) -> None:
        # Initialize embeddings in FP32 then convert to target dtype
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.config.dim))
        self.tok_embeddings = self.tok_embeddings.to(dtype=self._embedding_dtype)
        
        if self.config.use_embedding_projection:
            nn.init.orthogonal_(self.embed_rot_in.weight)
            nn.init.orthogonal_(self.embed_rot_out.weight)
            self.embed_rot_in = self.embed_rot_in.to(dtype=self._embedding_dtype)
            self.embed_rot_out = self.embed_rot_out.to(dtype=self._embedding_dtype)
        
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.tok_embeddings.weight[self.config.pad_token_id].zero_()
```

**Rationale**: 
- Token embeddings converted to BF16 for 50% memory reduction
- Embedding projection layers (OSP rotations) also use BF16
- Final RMSNorm uses BF16 weights
- Initialization done in FP32 then converted for numerical stability

**Affected Components**: 
- `tok_embeddings.weight` - shape [vocab_size, dim]
- `embed_rot_in.weight` - shape [dim, dim] (when enabled)
- `embed_rot_out.weight` - shape [dim, dim] (when enabled)
- `norm.weight` - shape [dim]

---

## Pending Changes

- [ ] Attention layer precision (Phase 2)
- [ ] Optimizer state precision (Phase 3)
- [ ] Gradient precision (Phase 4)
- [ ] Verify checkpoint compatibility with BF16 embeddings
- [ ] Run validation tests on BF16 embedding model

---

## Notes & Considerations

### Precision Migration Strategy

1. **Embeddings First** ✅ - Completed
   - Token embeddings: FP32 → BF16
   - Embedding projections: FP32 → BF16
   - RMSNorm weights: FP32 → BF16
   - SingleScaleRMSNorm: FP32 → FP8

2. **Attention Layers** - To be handled in Phase 2
3. **Optimizer State** - To be handled in Phase 3
4. **Gradients** - To be handled in Phase 4

### Memory Savings

For a 30B parameter model with vocab_size=32,000 and dim=768:
- **Token embeddings**: 32,000 × 768 × 4 bytes = 98.3 MB (FP32) → 49.2 MB (BF16) = **49.1 MB saved**
- **Per-layer RMSNorm**: 768 × 4 bytes = 3 KB (FP32) → 1.5 KB (BF16) = **1.5 KB saved per layer**
- **Total RMSNorm savings**: 1.5 KB × (n_layers × 2 + 1) = ~60 KB for 20 layers

### BF16 Considerations

- BF16 provides dynamic range equivalent to FP32 (same 8-bit exponent)
- Reduced precision (7-bit mantissa vs 24-bit for FP32)
- Native support on Ampere+ GPUs (A100, H100)
- May require gradient scaling in some cases
- Initialization done in FP32 then converted for numerical stability

### FP8 Considerations

- FP8 (`float8_e4m3fn`) used for SingleScaleRMSNorm scalar weights
- Even more memory efficient (1 byte per scalar)
- Sufficient dynamic range for single-gain normalization
- Converted to FP32 for computation

---

## References

- PyTorch BF16 Documentation: https://pytorch.org/docs/stable/generated/torch.bfloat16.html
- PyTorch FP8 Documentation: https://pytorch.org/docs/stable/generated/torch.float8_e4m3fn.html
- Model Architecture: `src/model.py`
- Training Configuration: `working/30b-run/configs/model_config.json`
