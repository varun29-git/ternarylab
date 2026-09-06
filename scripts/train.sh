#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CONFIG:?Set CONFIG to the reviewed model/training JSON}"
: "${MANIFEST:?Set MANIFEST to the verified manifest}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared dataset directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a unique run directory}"
: "${PREFLIGHT_REPORT:?Run the CUDA preflight and set PREFLIGHT_REPORT}"
exec "${PYTHON_BIN:-python}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-1}" --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" --module terminus.train \
  --config "$CONFIG" --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" --preflight-report "$PREFLIGHT_REPORT" "$@"
