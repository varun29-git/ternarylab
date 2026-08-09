#!/usr/bin/env bash
# Run on the already-provisioned Azure H100.  This is a kernel qualification
# gate, not a data-backed training launcher.
set -euo pipefail
set +x

if [[ $# -ne 3 ]]; then
  echo "usage: $0 REPOSITORY RESULTS_DIRECTORY DEEPGEMM_COMMIT" >&2
  exit 2
fi

REPOSITORY=$1
RESULTS_DIRECTORY=$2
DEEPGEMM_COMMIT=$3
EXPECTED_DEEPGEMM_VERSION=2.6.1
MUON_EXPECTED_COMMIT=f98f1cacc0263b04290753e32be8d498c1efc806

# Detect and configure CUDA for DeepGEMM JIT compilation
# DeepGEMM requires NVCC 12.9+ for assembly constraints
# Use CUDA 13.2 for nvcc (DeepGEMM JIT), create nvrtc.h symlink for Triton
CUDA_HOME="${CUDA_HOME:-}"

# Find CUDA with nvcc 12.9+ for DeepGEMM JIT compilation
if [[ -z "$CUDA_HOME" ]] || [[ ! -d "$CUDA_HOME" ]]; then
  # Prefer CUDA 13.2 which has nvcc 13.2 (>= 12.9 required by DeepGEMM)
  for cuda_path in /usr/local/cuda-13.2 /usr/local/cuda-13 /usr/local/cuda; do
    if [[ -d "$cuda_path" ]] && [[ -x "$cuda_path/bin/nvcc" ]]; then
      CUDA_HOME="$cuda_path"
      break
    fi
  done
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# Ensure nvrtc.h is findable - create symlink if needed
# CUDA 13.2 doesn't have nvrtc.h, but CUDA 12.6 has it in targets/x86_64-linux/include/
if [[ ! -f "$CUDA_HOME/include/nvrtc.h" ]]; then
  # Try to find nvrtc.h in CUDA 12.6
  for nvrtc_path in /usr/local/cuda-12.6 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -f "$nvrtc_path/targets/x86_64-linux/include/nvrtc.h" ]]; then
      ln -sf "$nvrtc_path/targets/x86_64-linux/include/nvrtc.h" "$CUDA_HOME/include/nvrtc.h" 2>/dev/null || true
      break
    elif [[ -f "$nvrtc_path/include/nvrtc.h" ]]; then
      ln -sf "$nvrtc_path/include/nvrtc.h" "$CUDA_HOME/include/nvrtc.h" 2>/dev/null || true
      break
    fi
  done
fi

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command is unavailable: $1" >&2
    exit 1
  }
}

require_command git
require_command python3
require_command nvidia-smi
require_command nvcc
[[ -d "$REPOSITORY/.git" ]] || {
  echo "repository must be an existing git checkout: $REPOSITORY" >&2
  exit 2
}
[[ -f "$REPOSITORY/src/ternary_training_preflight.py" ]] || {
  echo "repository does not contain the BitLinear qualification entrypoint" >&2
  exit 2
}

# The first kernel-evolution phase always targets this exact 400M model.  The
# remote Azure shell does not inherit arbitrary local variables through az ssh,
# so define and validate the public, non-secret profile here rather than asking
# an operator to rediscover it.  This uses the tied 64k token embedding/LM head.
TERMINUS_EVOLVE_PROFILE="${TERMINUS_EVOLVE_PROFILE:-terminus_400m_64k_h100_v1}"
[[ "$TERMINUS_EVOLVE_PROFILE" == "terminus_400m_64k_h100_v1" ]] || {
  echo "only the frozen terminus_400m_64k_h100_v1 profile is valid for this evolution phase" >&2
  exit 2
}
declare -A target_shape=(
  [TERMINUS_EVOLVE_DIM]=1024
  [TERMINUS_EVOLVE_LAYERS]=20
  [TERMINUS_EVOLVE_FFN_DIM]=4096
  [TERMINUS_EVOLVE_N_HEADS]=16
  [TERMINUS_EVOLVE_N_KV_HEADS]=4
  [TERMINUS_EVOLVE_BATCH]=14
  [TERMINUS_EVOLVE_SEQ_LEN]=4096
  [TERMINUS_EVOLVE_VOCAB_SIZE]=64000
)
for name in "${!target_shape[@]}"; do
  actual="${!name:-${target_shape[$name]}}"
  [[ "$actual" == "${target_shape[$name]}" ]] || {
    echo "$TERMINUS_EVOLVE_PROFILE requires $name=${target_shape[$name]}, got $actual" >&2
    exit 2
  }
  export "$name=$actual"
done

CUDA_VERSION_OUTPUT=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
if [[ -z "$CUDA_VERSION_OUTPUT" ]]; then
  echo "Could not parse CUDA version from nvcc --version" >&2
  exit 1
fi
IFS='.' read -r CUDA_MAJOR CUDA_MINOR <<< "$CUDA_VERSION_OUTPUT"
if [[ -z "${CUDA_MAJOR:-}" || -z "${CUDA_MINOR:-}" || "$CUDA_MAJOR" -lt 12 || ( "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -lt 3 ) ]]; then
  echo "DeepGEMM SM90 requires CUDA Toolkit 12.3+; install/select a newer toolkit before continuing." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIRECTORY"
umask 077
RUNTIME_ROOT="$RESULTS_DIRECTORY/runtime"
VENV="$RUNTIME_ROOT/venv"
DEEPGEMM_PATH="$RUNTIME_ROOT/DeepGEMM"
mkdir -p "$RUNTIME_ROOT" "$RESULTS_DIRECTORY/artifacts"
export FLA_TILELANG="${FLA_TILELANG:-0}"
export DG_JIT_CACHE_DIR="$RUNTIME_ROOT/deepgemm-jit"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/torchinductor"
export BITNET_RUNTIME_CACHE_ROOT="$RUNTIME_ROOT/bitnet"

patch_fla_tilelang_guard() {
  python - <<'PY'
from pathlib import Path
import sysconfig

path = Path(sysconfig.get_paths()["purelib"]) / "fla/ops/common/backends/tilelang/__init__.py"
if not path.exists():
    raise SystemExit(0)

text = path.read_text(encoding="utf-8")
needle = """    @classmethod
    def is_available(cls) -> bool:
        try:
            import tilelang  # noqa: F401
            return True
        except ImportError:
            return False
"""
replacement = """    @classmethod
    def is_available(cls) -> bool:
        if not cls.is_enabled():
            return False
        try:
            import tilelang  # noqa: F401
            return True
        except ImportError:
            return False
"""
if replacement not in text and needle in text:
    path.write_text(text.replace(needle, replacement), encoding="utf-8")
PY
}

python3 -m venv "$VENV"

# Write CUDA paths to venv activate script for DeepGEMM JIT compilation
# DeepGEMM requires NVCC 12.9+ for assembly constraints
# Set CUDA_HOME to CUDA 13.2 for DeepGEMM's nvcc, create nvrtc.h symlink for Triton
cat >> "$VENV/bin/activate" << 'CUDA_ACTIVATE'
# CUDA paths for DeepGEMM JIT compilation (requires NVCC 12.9+)
# DeepGEMM uses CUDA_HOME to find nvcc, so set it to CUDA 13.2
if [[ -z "${CUDA_HOME:-}" ]]; then
  for cuda_path in /usr/local/cuda-13.2 /usr/local/cuda-13 /usr/local/cuda; do
    if [[ -d "$cuda_path" ]] && [[ -x "$cuda_path/bin/nvcc" ]]; then
      export CUDA_HOME="$cuda_path"
      break
    fi
  done
fi
# Ensure nvcc is in PATH
if [[ -n "${CUDA_HOME:-}" ]] && [[ -d "$CUDA_HOME/bin" ]]; then
  case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
  esac
fi
CUDA_ACTIVATE

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet setuptools wheel ninja packaging psutil
python -m pip install --quiet --index-url https://download.pytorch.org/whl/cu126 torch==2.8.0
python -m pip install --quiet triton==3.4.0 numpy==1.26.4
python -m pip install --quiet datasets==2.18.0 transformers==4.40.2 tokenizers==0.19.1 tqdm==4.66.4 sentencepiece==0.2.0
python -m pip install --quiet --no-deps fla-core==0.5.0 flash-linear-attention==0.5.0 tilelang==0.1.10
python -m pip install --quiet --no-deps "muon-optimizer @ git+https://github.com/KellerJordan/Muon.git@${MUON_EXPECTED_COMMIT}"
patch_fla_tilelang_guard

if [[ ! -d "$DEEPGEMM_PATH/.git" ]]; then
  git clone --quiet --recursive https://github.com/deepseek-ai/DeepGEMM.git "$DEEPGEMM_PATH"
fi
git -C "$DEEPGEMM_PATH" fetch --quiet --tags origin
git -C "$DEEPGEMM_PATH" checkout --quiet --detach "$DEEPGEMM_COMMIT"
git -C "$DEEPGEMM_PATH" submodule update --init --recursive --quiet

# Ensure nvrtc.h symlink exists before building DeepGEMM
# CUDA 13.2 doesn't have nvrtc.h, but DeepGEMM's C++ extension requires it
if [[ ! -f "$CUDA_HOME/include/nvrtc.h" ]]; then
  for nvrtc_path in /usr/local/cuda-12.6 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -f "$nvrtc_path/targets/x86_64-linux/include/nvrtc.h" ]]; then
      ln -sf "$nvrtc_path/targets/x86_64-linux/include/nvrtc.h" "$CUDA_HOME/include/nvrtc.h" 2>/dev/null || true
      break
    elif [[ -f "$nvrtc_path/include/nvrtc.h" ]]; then
      ln -sf "$nvrtc_path/include/nvrtc.h" "$CUDA_HOME/include/nvrtc.h" 2>/dev/null || true
      break
    fi
  done
fi

(cd "$DEEPGEMM_PATH" && ./install.sh)

# Install Flash Attention for H100 SM90
# flash-attn provides flash_attn.flash_attn_interface module
python -m pip install --quiet flash-attn --no-build-isolation

python - "$EXPECTED_DEEPGEMM_VERSION" "$DEEPGEMM_COMMIT" "$MUON_EXPECTED_COMMIT" "$RESULTS_DIRECTORY/artifacts/runtime.json" <<'PY'
import json
from importlib import metadata
import importlib.util
import sys
from pathlib import Path

import deep_gemm
import torch
import triton

expected_version, expected_commit, expected_muon_commit, output = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the pinned runtime")
capability = torch.cuda.get_device_capability()
if capability[0] != 9:
    raise SystemExit(f"expected H100-class SM90, got {capability}")
if deep_gemm.__version__ != expected_version:
    raise SystemExit(f"expected DeepGEMM {expected_version}, got {deep_gemm.__version__}")
for api in ("bf16_gemm_tn", "fp8_gemm_nt"):
    if not callable(getattr(deep_gemm, api, None)):
        raise SystemExit(f"required DeepGEMM API missing: {api}")
if importlib.util.find_spec("flash_attn.flash_attn_interface") is None:
  raise SystemExit("flash_attn.flash_attn_interface is unavailable; the FA3 production path is not qualified")
muon_dist = metadata.distribution("muon-optimizer")
direct_url = muon_dist.read_text("direct_url.json")
if not direct_url:
    raise SystemExit("muon-optimizer direct_url.json provenance is unavailable")
actual_muon_commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
if actual_muon_commit != expected_muon_commit:
    raise SystemExit(f"expected Muon commit {expected_muon_commit}, got {actual_muon_commit}")
Path(output).write_text(json.dumps({
    "deepgemm_commit": expected_commit,
    "deepgemm_version": deep_gemm.__version__,
    "device": torch.cuda.get_device_name(0),
    "capability": capability,
    "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "flash_attn_interface": True,
    "muon_commit": actual_muon_commit,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
}, indent=2, sort_keys=True) + "\n")
PY

cd "$REPOSITORY"
python src/ternary_training_preflight.py \
  --dim "$TERMINUS_EVOLVE_DIM" \
  --ffn-dim "$TERMINUS_EVOLVE_FFN_DIM" \
  --n-heads "$TERMINUS_EVOLVE_N_HEADS" \
  --n-kv-heads "$TERMINUS_EVOLVE_N_KV_HEADS" \
  --group-size 128 \
  --wgrad-backend deepgemm_fp8 \
  --fp8-wgrad-roles ffn_gate,ffn_up,ffn_down \
  --allow-fp8-wgrad \
  --warmup 3 --iterations 10 \
  --json-output "$RESULTS_DIRECTORY/artifacts/baseline_preflight.json"

python src/throughput_probe.py \
  --modes transformerpp \
  --batch-sizes "$TERMINUS_EVOLVE_BATCH" \
  --seq-len "$TERMINUS_EVOLVE_SEQ_LEN" \
  --vocab-size "$TERMINUS_EVOLVE_VOCAB_SIZE" \
  --dim "$TERMINUS_EVOLVE_DIM" \
  --layers "$TERMINUS_EVOLVE_LAYERS" \
  --ffn-dim "$TERMINUS_EVOLVE_FFN_DIM" \
  --heads "$TERMINUS_EVOLVE_N_HEADS" \
  --kv-heads "$TERMINUS_EVOLVE_N_KV_HEADS" \
  --steps 8 --warmup-steps 4 \
  --compile --compile-mode default --optimizer muon \
  --attention-backend fa3 \
  --bitlinear-kernel-backend triton_packed \
  --bitlinear-wgrad-backend deepgemm_fp8 \
  --fp8-wgrad-roles ffn_gate,ffn_up,ffn_down \
  --run-label baseline \
  --jsonl-output "$RESULTS_DIRECTORY/artifacts/baseline_throughput.jsonl"

git rev-parse HEAD > "$RESULTS_DIRECTORY/artifacts/baseline_commit.txt"
echo "H100 qualification completed: $RESULTS_DIRECTORY/artifacts"
