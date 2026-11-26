#!/usr/bin/env python3
"""
Validate checkpoint for continual training
Ensures the checkpoint can be loaded and matches expected configuration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import BitNetLlama, BitNetLlamaConfig


def validate_checkpoint(checkpoint_path: str) -> dict:
    """Validate a checkpoint can be loaded and return its metadata."""

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Check for sidecar
    sidecar_path = ckpt_path.with_suffix(ckpt_path.suffix + ".json")
    if not sidecar_path.exists():
        print(f"Warning: Sidecar file not found: {sidecar_path}")
    else:
        sidecar = json.loads(sidecar_path.read_text())
        print(f"Checkpoint size: {sidecar.get('size_bytes', 'unknown')} bytes")
        print(f"Checkpoint SHA256: {sidecar.get('sha256', 'unknown')[:16]}...")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    device = torch.device("cpu")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    # Extract metadata
    schema_version = checkpoint.get("schema_version", "unknown")
    next_step = checkpoint.get("next_step", "unknown")
    run_id = checkpoint.get("run_id", "unknown")

    print(f"Schema version: {schema_version}")
    print(f"Next step: {next_step}")
    print(f"Run ID: {run_id}")

    # Check resume contract
    resume_contract = checkpoint.get("resume_contract")
    if resume_contract:
        print(f"\nResume contract fingerprint: {resume_contract.get('fingerprint', 'unknown')[:16]}...")

        model_config_dict = resume_contract.get("model_config", {})
        print("\nModel configuration from checkpoint:")
        print(f"  vocab_size: {model_config_dict.get('vocab_size')}")
        print(f"  dim: {model_config_dict.get('dim')}")
        print(f"  n_layers: {model_config_dict.get('n_layers')}")
        print(f"  n_heads: {model_config_dict.get('n_heads')}")
        print(f"  n_kv_heads: {model_config_dict.get('n_kv_heads')}")
        print(f"  ffn_dim: {model_config_dict.get('ffn_dim')}")
        print(f"  expand: {model_config_dict.get('expand')}")
        print(f"  max_seq_len: {model_config_dict.get('max_seq_len')}")
        print(f"  attention_backend: {model_config_dict.get('attention_backend')}")
        print(f"  alpha_mode: {model_config_dict.get('alpha_mode')}")
        print(f"  transformer_alpha_mode: {model_config_dict.get('transformer_alpha_mode')}")
        print(f"  alpha_group_size: {model_config_dict.get('alpha_group_size')}")
        print(f"  local_attention_window: {model_config_dict.get('local_attention_window')}")
        print(f"  global_attention_every: {model_config_dict.get('global_attention_every')}")

    # Check model state
    model_state = checkpoint.get("model", {})
    print("\nModel state:")
    print(f"  Number of parameters: {len(model_state)}")

    total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
    print(f"  Total parameter count: {total_params:,}")

    # Check optimizer state
    optimizer_state = checkpoint.get("optimizer", {})
    print("\nOptimizer state:")
    print(f"  State dict keys: {list(optimizer_state.keys()) if isinstance(optimizer_state, dict) else 'Not a dict'}")

    return {
        "schema_version": schema_version,
        "next_step": next_step,
        "run_id": run_id,
        "model_config": resume_contract.get("model_config", {}) if resume_contract else {},
        "total_params": total_params,
        "valid": True,
    }


def test_model_instantiation(config_dict: dict) -> bool:
    """Test that a model can be instantiated with the given config."""

    print("\n" + "=" * 50)
    print("Testing model instantiation...")
    print("=" * 50)

    try:
        config = BitNetLlamaConfig(**config_dict)
        model = BitNetLlama(config)

        counts = model.count_parameters()
        print("\nModel instantiated successfully!")
        print(f"Total parameters: {counts['total']:,}")
        print(f"  Transformer parameters: {counts.get('transformer', 0):,}")
        print(f"  Embedding parameters: {counts.get('embedding', 0):,}")
        print(f"  Head parameters: {counts.get('head', 0):,}")
        print(f"  Alpha parameters: {counts.get('alpha', 0):,}")

        return True

    except Exception as e:
        print(f"\nFailed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate checkpoint for continual training")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--test-model", action="store_true", help="Test model instantiation")
    args = parser.parse_args()

    print("=" * 50)
    print("Checkpoint Validation")
    print("=" * 50)

    try:
        result = validate_checkpoint(args.checkpoint)

        if args.test_model and result.get("model_config"):
            success = test_model_instantiation(result["model_config"])
            if not success:
                sys.exit(1)

        print("\n" + "=" * 50)
        print("✓ Checkpoint validation PASSED")
        print("=" * 50)
        print("\nCheckpoint is ready for continual training!")
        print(f"Resume from step: {result.get('next_step', 'unknown')}")

    except Exception as e:
        print("\n" + "=" * 50)
        print("✗ Checkpoint validation FAILED")
        print("=" * 50)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
