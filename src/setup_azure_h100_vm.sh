#!/usr/bin/env bash
# Azure H100 VM Setup Script
# Run this on the Azure H100 VM to prepare the environment

set -euo pipefail

echo "=========================================="
echo "Azure H100 VM Setup for Continual Training"
echo "=========================================="

# =============================================================================
# CHECK VM TYPE
# =============================================================================

echo ""
echo "Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is this an GPU VM?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
echo "GPU: ${GPU_NAME}"
echo "GPU Memory: ${GPU_MEMORY}"

# Check if H100
if [[ ! "${GPU_NAME}" =~ H100 ]]; then
    echo "WARNING: This script is optimized for H100. Detected: ${GPU_NAME}"
fi

# =============================================================================
# SETUP DIRECTORIES
# =============================================================================

RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/continual-training-runtime}"
DATA_DIR="${DATA_DIR:-/mnt/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/checkpoints}"

echo ""
echo "Setting up directories..."
sudo mkdir -p "${RUNTIME_ROOT}" "${DATA_DIR}" "${CHECKPOINT_DIR}"
sudo chmod 777 "${RUNTIME_ROOT}" "${DATA_DIR}" "${CHECKPOINT_DIR}"

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

echo ""
echo "Installing dependencies..."

# Update package list
sudo apt-get update -y

# Install essential packages
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    htop \
    nvtop \
    unzip \
    tmux

# Install AzCopy if not present
if ! command -v azcopy &> /dev/null; then
    echo "Installing AzCopy..."
    cd /tmp
    wget -q https://aka.ms/downloadazcopy-v10-linux -O azcopy.tgz
    tar -xzf azcopy.tgz
    sudo cp azcopy_linux_amd64_*/azcopy /usr/local/bin/
    sudo chmod +x /usr/local/bin/azcopy
    rm -rf azcopy.tgz azcopy_linux_amd64_*
    echo "AzCopy installed successfully"
fi

# =============================================================================
# SETUP PYTHON ENVIRONMENT
# =============================================================================

echo ""
echo "Setting up Python virtual environment..."
VENV_DIR="${RUNTIME_ROOT}/venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.6 (for H100)
echo "Installing PyTorch..."
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
echo "Installing dependencies..."
pip install \
    numpy==1.26.4 \
    datasets==2.18.0 \
    transformers==4.40.2 \
    tokenizers==0.19.1 \
    tqdm==4.66.4 \
    sentencepiece==0.2.0

# Install flash-linear-attention
echo "Installing flash-linear-attention..."
pip install --no-deps \
    fla-core==0.5.0 \
    flash-linear-attention==0.5.0 \
    tilelang==0.1.10

# Install Muon optimizer
echo "Installing Muon optimizer..."
pip install --no-deps \
    "muon-optimizer @ git+https://github.com/KellerJordan/Muon.git@f98f1cacc0263b04290753e32be8d498c1efc806"

echo "Python environment setup complete!"

# =============================================================================
# VERIFY INSTALLATION
# =============================================================================

echo ""
echo "Verifying installation..."

python3 << 'PYEOF'
import sys
import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    BF16 supported: {torch.cuda.is_bf16_supported()}")

# Check flash-attention
try:
    import flash_attn_interface
    print("FlashAttention-3: Available")
except ImportError:
    print("FlashAttention-3: Not available")

# Check muon
try:
    import muon
    print("Muon optimizer: Available")
except ImportError:
    print("Muon optimizer: Not available")

print("\nAll checks complete!")
PYEOF

# =============================================================================
# DOWNLOAD DATA (if URLs provided)
# =============================================================================

if [[ -n "${COSMO_URL:-}" ]]; then
    echo ""
    echo "Downloading Cosmopedia data..."
    azcopy copy "${COSMO_URL}" "${DATA_DIR}/cosmopedia_v2_mincols_train.bin"
fi

if [[ -n "${VAL_URL:-}" ]]; then
    echo ""
    echo "Downloading validation data..."
    azcopy copy "${VAL_URL}" "${DATA_DIR}/cosmopedia_v2_mincols_val.bin"
fi

if [[ -n "${CHECKPOINT_URL:-}" ]]; then
    echo ""
    echo "Downloading checkpoint..."
    azcopy copy "${CHECKPOINT_URL}" "${CHECKPOINT_DIR}/"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Runtime root: ${RUNTIME_ROOT}"
echo "Data directory: ${DATA_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Virtual environment: ${VENV_DIR}"
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Copy your checkpoint to: ${CHECKPOINT_DIR}/"
echo "  2. Copy your data to: ${DATA_DIR}/"
echo "  3. Run: ./run_azure_h100_continual_training.sh"
echo ""
