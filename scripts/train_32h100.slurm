#!/usr/bin/env bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=7-12:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --job-name=terminus-exact-1p5b

set -euo pipefail
: "${SLURM_JOB_ID:?Run through sbatch on the allocated H100 cluster}"
: "${WORKSPACE:?Absolute checkout path present on every node}"
: "${MANIFEST:?Verified manifest path present on every node}"
: "${DATA_ROOT:?Node-local prepared corpus directory}"
: "${OUTPUT_DIR:?Shared checkpoint and event directory}"
: "${PREFLIGHT_REPORT:?Matching H100 preflight receipt path on every node}"
export CONFIG="${CONFIG:-configs/1.5b_exact_h100_fp8.json}"
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500 NNODES=4 NPROC_PER_NODE=8 OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# NCCL selects topology and algorithms from the actual NVSwitch/IB fabric.
# Do not force TCP, an HCA name, or an algorithm without topology measurements.
cd "$WORKSPACE"
srun --kill-on-bad-exit=1 --cpu-bind=none bash -c \
  'export NODE_RANK="$SLURM_PROCID"; exec bash scripts/train.sh "$@"' bash "$@" &
train_pid=$!
trap 'kill -TERM "$train_pid" 2>/dev/null || true' TERM INT
wait "$train_pid"
