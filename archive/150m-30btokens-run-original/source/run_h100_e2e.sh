#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Test 17 Hopper launch profile. All machine/data locations must
# be supplied by environment variables so tracked files never contain cloud or VM
# details.
#
# Test 17 should not inherit metric-triggered kill behavior from a parent shell.
# The trainer has no built-in metric-based auto-abort path, and this launcher
# forces the external watchdog expectation off as well.
export BITNET_EXTERNAL_WATCHDOG=0

PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${RUNTIME_ROOT:?Set RUNTIME_ROOT to the machine work directory.}"
VENV_DIR="${VENV_DIR:-${RUNTIME_ROOT}/venv}"
DATA_ROOT="${DATA_ROOT:-${RUNTIME_ROOT}}"
: "${COSMO_BIN:?Set COSMO_BIN to the tokenized Cosmopedia memmap path.}"
: "${VAL_BIN:?Set VAL_BIN to a disjoint validation memmap path.}"
FINEWEB_BIN="${FINEWEB_BIN:-}"
FINEWEB_FRACTION="${FINEWEB_FRACTION:-0.20}"
RESUME="${RESUME:-}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_ROOT}/triton_cache}"

SKIP_INSTALL="${SKIP_INSTALL:-0}"
SEQ_LEN="${SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-14}"
COMPILE="${COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
# Test 16 could not answer several scale questions because research telemetry
# was advertised but disabled in the production launcher. Test 17 keeps the
# evidence streams on and refuses to enter production through this launcher if
# they are disabled.
RESEARCH_MODE="${RESEARCH_MODE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
# The selected production backend is the only mandatory preflight target.
# Fixed causal local attention must use FA3's native window_size path.  Flex is
# reserved for a future irregular-mask implementation and remains opt-in: its
# own scaling gate must pass before it can become a production choice.
ATTENTION_PREFLIGHT_BACKENDS="${ATTENTION_PREFLIGHT_BACKENDS:-${ATTENTION_BACKEND}}"
ATTENTION_PREFLIGHT_LENGTHS="${ATTENTION_PREFLIGHT_LENGTHS:-1,1023,1024,1025,4096}"
ATTENTION_SCALING_LENGTHS="${ATTENTION_SCALING_LENGTHS:-2048,4096}"
RUN_INFERENCE_CACHE_PREFLIGHT="${RUN_INFERENCE_CACHE_PREFLIGHT:-1}"
FA3_HOPPER_PATH="${FA3_HOPPER_PATH:-}"
TRANSFORMER_LAYER_INDICES="${TRANSFORMER_LAYER_INDICES:-all}"
LOCAL_ATTENTION_WINDOW="${LOCAL_ATTENTION_WINDOW:-1024}"
GLOBAL_ATTENTION_EVERY="${GLOBAL_ATTENTION_EVERY:-5}"
QK_NORM_MODE="${QK_NORM_MODE:-unit_rms}"
if [[ "${QK_NORM_MODE}" != "unit_rms" ]]; then
  echo "[attention] ERROR: QK_NORM_MODE=unit_rms is mandatory; affine Q/K gains are forbidden." >&2
  exit 2
fi
FFN_ACTIVATION="${FFN_ACTIVATION:-swiglu}"
USE_EMBEDDING_PROJECTION="${USE_EMBEDDING_PROJECTION:-0}"
ATTENTION_TEMPERATURE_MODE="${ATTENTION_TEMPERATURE_MODE:-none}"
ATTENTION_TEMPERATURE_MIN="${ATTENTION_TEMPERATURE_MIN:-0.5}"
ATTENTION_TEMPERATURE_MAX="${ATTENTION_TEMPERATURE_MAX:-2.0}"
ATTENTION_TEMPERATURE_INIT="${ATTENTION_TEMPERATURE_INIT:-1.0}"
ATTENTION_SOFTCAP="${ATTENTION_SOFTCAP:-0.0}"
REQUIRE_SHARD_ALIGNED_ALPHA_GROUPS="${REQUIRE_SHARD_ALIGNED_ALPHA_GROUPS:-0}"
REQUESTED_RUN_ID="${RUN_ID:-}"
RUN_ID="${REQUESTED_RUN_ID:-test17-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ -n "${RESUME}" && -z "${REQUESTED_RUN_ID}" ]]; then
  echo "[setup] ERROR: resume requires the checkpoint's exact RUN_ID." >&2
  exit 2
fi
MANIFEST="${MANIFEST:-${RUNTIME_ROOT}/${RUN_ID}_manifest.json}"
CKPT_STORE="${CKPT_STORE:-${RUNTIME_ROOT}/${RUN_ID}_checkpoints}"
TOTAL_TOKENS="${TOTAL_TOKENS:-10000000000}"
SCHEDULE_HORIZON_TOKENS="${SCHEDULE_HORIZON_TOKENS:-}"
MAX_STEPS="${MAX_STEPS:-}"
CHECKPOINT_TOKENS="${CHECKPOINT_TOKENS:-200000000}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_ID}}"
DTYPE="${DTYPE:-bf16}"
LOSS_SCALE_INIT="${LOSS_SCALE_INIT:-128}"
MIN_LOSS_SCALE="${MIN_LOSS_SCALE:-1}"
ENABLE_ACTIVATION_QUANTIZATION="${ENABLE_ACTIVATION_QUANTIZATION:-0}"
if [[ "${ENABLE_ACTIVATION_QUANTIZATION}" != "0" ]]; then
  echo "[quantization] ERROR: activation quantization uses integer saturation and is forbidden in the no-hard-cap training path." >&2
  exit 2
fi
TOKENIZER_NAME="${TOKENIZER_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-fe8a4ea1ffedaf415f4da2f062534de366a451e6}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
MODEL_DIM="${MODEL_DIM:-768}"
N_LAYERS="${N_LAYERS:-20}"
N_HEADS="${N_HEADS:-12}"
N_KV_HEADS="${N_KV_HEADS:-4}"
FFN_DIM="${FFN_DIM:-2048}"
EXPAND="${EXPAND:-2}"
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-}"
RUN_SYNC_URL="${RUN_SYNC_URL:-}"
RUN_SYNC_BASE_URL="${RUN_SYNC_BASE_URL:-}"
RUN_SYNC_INTERVAL_SECONDS="${RUN_SYNC_INTERVAL_SECONDS:-180}"
RUN_SYNC_CHECKPOINTS="${RUN_SYNC_CHECKPOINTS:-1}"
REQUIRE_RUN_SYNC="${REQUIRE_RUN_SYNC:-1}"
POST_RUN_EVALUATOR="${POST_RUN_EVALUATOR:-}"
NVIDIA_CONF_COMPUTE_SET_READY="${NVIDIA_CONF_COMPUTE_SET_READY:-1}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-8}"
VAL_BATCHES="${VAL_BATCHES:-32}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-250}"
BOOTSTRAP_LOG_STEPS="${BOOTSTRAP_LOG_STEPS:-5}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-1}"
FLIP_INTERVAL="${FLIP_INTERVAL:-2000}"
GRAD_DETAIL_INTERVAL="${GRAD_DETAIL_INTERVAL:-1000}"
OPTIMIZER_STATS_INTERVAL="${OPTIMIZER_STATS_INTERVAL:-1000}"
SYSTEM_INTERVAL="${SYSTEM_INTERVAL:-1000}"
QUANT_ERROR_INTERVAL="${QUANT_ERROR_INTERVAL:-1000}"
MAX_QUANTIZATION_RELATIVE_MSE="${MAX_QUANTIZATION_RELATIVE_MSE:-}"
MIN_TERNARY_ZERO_FRACTION="${MIN_TERNARY_ZERO_FRACTION:-}"
MAX_TERNARY_ZERO_FRACTION="${MAX_TERNARY_ZERO_FRACTION:-}"
QUANTIZATION_HEALTH_GRACE_FRACTION="${QUANTIZATION_HEALTH_GRACE_FRACTION:-0.10}"
ACTIVATION_STATS_INTERVAL="${ACTIVATION_STATS_INTERVAL:-1000}"
SSM_STATS_INTERVAL="${SSM_STATS_INTERVAL:-1000}"
ATTENTION_STATS_INTERVAL="${ATTENTION_STATS_INTERVAL:-1000}"
ATTENTION_STATS_MODE="${ATTENTION_STATS_MODE:-sampled}"
ATTENTION_STATS_SAMPLE_SEQ_LEN="${ATTENTION_STATS_SAMPLE_SEQ_LEN:-256}"
ATTENTION_STATS_QUERY_TILE_SIZE="${ATTENTION_STATS_QUERY_TILE_SIZE:-128}"
DYNAMICS_INTERVAL="${DYNAMICS_INTERVAL:-1000}"
UPDATE_SAMPLE_SIZE="${UPDATE_SAMPLE_SIZE:-4096}"
TELEMETRY_FAIL_HARD="${TELEMETRY_FAIL_HARD:-0}"
HEALTH_GATE_MODE="${HEALTH_GATE_MODE:-observe}"
# This is an abort-only telemetry threshold. It never clips or modifies logits.
ATTENTION_LOGIT_ABORT_THRESHOLD="${ATTENTION_LOGIT_ABORT_THRESHOLD:-}"
MAX_ACTIVATION_ABS="${MAX_ACTIVATION_ABS:-}"
MAX_ACTIVATION_MAX_P999_RATIO="${MAX_ACTIVATION_MAX_P999_RATIO:-}"
ACTIVATION_BASELINE_SAMPLES="${ACTIVATION_BASELINE_SAMPLES:-3}"
# Use ${var-...}, not ${var:-...}: an explicitly empty value disables this
# optional threshold. Test 20 exposed that collapsing empty to 2.0 can silently
# turn an observation-only run back into an aborting run.
MAX_ACTIVATION_RMS_BASELINE_RATIO="${MAX_ACTIVATION_RMS_BASELINE_RATIO-2.0}"
TELEMETRY_PROBE_SOURCE="${TELEMETRY_PROBE_SOURCE:-training_batch}"
ACTIVATION_GATE_CONSECUTIVE_BREACHES="${ACTIVATION_GATE_CONSECUTIVE_BREACHES:-1}"
if [[ "${RESEARCH_MODE}" != "1" ]]; then
  echo "[telemetry] ERROR: Test 17 production requires RESEARCH_MODE=1." >&2
  exit 2
fi
if [[ ! "${UPDATE_SAMPLE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[telemetry] ERROR: UPDATE_SAMPLE_SIZE must be a positive integer." >&2
  exit 2
fi
FIXED_VAL_SLICES="${FIXED_VAL_SLICES:-4}"
MILESTONE_EVAL_BATCHES="${MILESTONE_EVAL_BATCHES:-32}"
MILESTONE_SAMPLE_MAX_NEW_TOKENS="${MILESTONE_SAMPLE_MAX_NEW_TOKENS:-96}"
MILESTONE_SAMPLE_TEMPERATURE="${MILESTONE_SAMPLE_TEMPERATURE:-0.8}"
MILESTONE_SAMPLE_TOP_K="${MILESTONE_SAMPLE_TOP_K:-50}"
MILESTONE_SAMPLE_TOP_P="${MILESTONE_SAMPLE_TOP_P:-0.95}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.0}"
LR_SCHEDULE="${LR_SCHEDULE:-phase_cosine}"
DECAY_FRACTION="${DECAY_FRACTION:-1.00}"
FINAL_LR_FRACTION="${FINAL_LR_FRACTION:-0.10}"
PHASE1_FRACTION_OF_TOTAL="${PHASE1_FRACTION_OF_TOTAL:-0.25}"
PHASE2_FRACTION_OF_TOTAL="${PHASE2_FRACTION_OF_TOTAL:-0.50}"
PHASE_LR_START_FRACTION="${PHASE_LR_START_FRACTION:-0.25}"
PHASE1_END_LR_FRACTION="${PHASE1_END_LR_FRACTION:-0.75}"
PHASE2_PEAK_LR_FRACTION="${PHASE2_PEAK_LR_FRACTION:-1.00}"
PHASE2_PEAK_RAMP_FRACTION="${PHASE2_PEAK_RAMP_FRACTION:-0.20}"
TERMINAL_COOLDOWN_FRACTION="${TERMINAL_COOLDOWN_FRACTION:-0.0}"
LINEAR_LR_HALFWAY_FRACTION="${LINEAR_LR_HALFWAY_FRACTION:-0.50}"
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.10}"
ADAMW_WEIGHT_DECAY="${ADAMW_WEIGHT_DECAY:-0.10}"
TORCH_INSTALL_SPEC="${TORCH_INSTALL_SPEC:-torch==2.8.0}"
TRITON_INSTALL_SPEC="${TRITON_INSTALL_SPEC:-triton==3.4.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_EXPECTED_VERSION="${TORCH_EXPECTED_VERSION:-2.8.0+cu126}"
TRITON_EXPECTED_VERSION="${TRITON_EXPECTED_VERSION:-3.4.0}"
FLA_CORE_SPEC="${FLA_CORE_SPEC:-fla-core==0.5.0}"
FLASH_LINEAR_ATTENTION_SPEC="${FLASH_LINEAR_ATTENTION_SPEC:-flash-linear-attention==0.5.0}"
TILELANG_SPEC="${TILELANG_SPEC:-tilelang==0.1.10}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy==1.26.4}"
DATASETS_SPEC="${DATASETS_SPEC:-datasets==2.18.0}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.40.2}"
TOKENIZERS_SPEC="${TOKENIZERS_SPEC:-tokenizers==0.19.1}"
TQDM_SPEC="${TQDM_SPEC:-tqdm==4.66.4}"
SENTENCEPIECE_SPEC="${SENTENCEPIECE_SPEC:-sentencepiece==0.2.0}"
MUON_EXPECTED_COMMIT="${MUON_EXPECTED_COMMIT:-f98f1cacc0263b04290753e32be8d498c1efc806}"
FA3_EXPECTED_COMMIT="${FA3_EXPECTED_COMMIT:-2402cb0bed7a2185cb9ddbe88fb998656cf73066}"
MUON_LR="${MUON_LR:-0.003}"
ADAMW_LR="${ADAMW_LR:-0.003}"
OPTIMIZER_KIND="${OPTIMIZER_KIND:-muon}"
if [[ "${OPTIMIZER_KIND}" != "muon" && "${OPTIMIZER_KIND}" != "adamw" ]]; then
  echo "[optimizer] ERROR: OPTIMIZER_KIND must be 'muon' or 'adamw'." >&2
  exit 2
fi
RESIDUAL_INIT_SCALE="${RESIDUAL_INIT_SCALE:-1.0}"
USE_SUBLN="${USE_SUBLN:-1}"
# Deterministic local alpha: grouped absmean recomputed from the current
# weights, shared across 128 parameters per alpha group.
ALPHA_MODE="${ALPHA_MODE:-deterministic_grouped_absmean}"
TRANSFORMER_ALPHA_MODE="${TRANSFORMER_ALPHA_MODE:-deterministic_grouped_absmean}"
ALPHA_GROUP_SIZE="${ALPHA_GROUP_SIZE:-128}"
ALPHA_LR="${ALPHA_LR:-0.00001}"
NORM_LR="${NORM_LR:-0.0002}"
ATTENTION_TEMPERATURE_LR="${ATTENTION_TEMPERATURE_LR:-0.0001}"
SPLIT_NORM_PARAMETERS="${SPLIT_NORM_PARAMETERS:-1}"
if [[ "${SPLIT_NORM_PARAMETERS}" != "1" ]]; then
  echo "[optimizer] ERROR: SPLIT_NORM_PARAMETERS=1 is mandatory; refusing to route norm gains through the general AdamW group." >&2
  exit 2
fi
ALPHA_REG_LAMBDA="${ALPHA_REG_LAMBDA:-0.0}"
QUANTIZATION_DECOMPOSITION_INTERVAL="${QUANTIZATION_DECOMPOSITION_INTERVAL:-0}"
FUNCTIONAL_QUANTIZATION_INTERVAL="${FUNCTIONAL_QUANTIZATION_INTERVAL:-0}"
FUNCTIONAL_QUANTIZATION_MAX_ROWS="${FUNCTIONAL_QUANTIZATION_MAX_ROWS:-2048}"
ALPHA_GRAD_CLIP="${ALPHA_GRAD_CLIP:-0.0}"
if [[ "${ALPHA_GRAD_CLIP}" != "0" && "${ALPHA_GRAD_CLIP}" != "0.0" ]]; then
  echo "[optimizer] ERROR: alpha gradient clipping is forbidden; use fail-fast telemetry instead." >&2
  exit 2
fi

TOKENS_PER_STEP=$((BATCH_SIZE * SEQ_LEN))
CHECKPOINT_INTERVAL=$(((CHECKPOINT_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
if (( CHECKPOINT_INTERVAL < 1 )); then
  CHECKPOINT_INTERVAL=1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TQDM_NCOLS="${TQDM_NCOLS:-120}"
export BITNET_MUON_IMPORT="${BITNET_MUON_IMPORT:-muon:SingleDeviceMuonWithAuxAdam}"
export BITNET_ATTENTION_BACKEND="${BITNET_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
export FLA_TILELANG="${FLA_TILELANG:-0}"
export ATTENTION_BACKEND LOCAL_ATTENTION_WINDOW GLOBAL_ATTENTION_EVERY SEQ_LEN
export MUON_EXPECTED_COMMIT FA3_EXPECTED_COMMIT FA3_HOPPER_PATH
if [[ -n "${FA3_HOPPER_PATH}" ]]; then
  # Do not put the Hopper source tree on PYTHONPATH.  It contains a
  # `flash_attn_3` source package without the compiled `_C` extension and
  # would therefore shadow the verified wheel installed from this checkout.
  # The FA3 commit check below pins the source used for the build; the runtime
  # import probe verifies that the installed binary extension is loadable.
  : "${FA3_HOPPER_PATH}"
fi
export TRITON_CACHE_DIR
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUNTIME_ROOT}/pip-cache}"
export AZCOPY_AUTO_LOGIN_TYPE="${AZCOPY_AUTO_LOGIN_TYPE:-${SYNC_AUTO_LOGIN_TYPE:-MSI}}"

RUN_NAME="$(basename "${OUTPUT_DIR}")"
AZURE_LIVE_LOG_BASE="${AZURE_LIVE_LOG_BASE:-}"
if [[ -n "${AZURE_LIVE_LOG_BASE}" && -z "${BITNET_AZURE_LOG_ROOT:-}" ]]; then
  export BITNET_AZURE_LOG_ROOT="${AZURE_LIVE_LOG_BASE%/}/${RUN_NAME}"
fi
if [[ -n "${BITNET_AZURE_LOG_ROOT:-}" ]]; then
  export BITNET_AZURE_TEXT_LOGS="${BITNET_AZURE_TEXT_LOGS:-1}"
fi
if [[ -z "${RUN_SYNC_URL}" && -n "${RUN_SYNC_BASE_URL}" ]]; then
  RUN_SYNC_URL="${RUN_SYNC_BASE_URL%/}/${RUN_NAME}"
fi
if [[ -z "${RUN_SYNC_URL}" && "${REQUIRE_RUN_SYNC}" == "1" ]]; then
  echo "[sync] ERROR: set RUN_SYNC_URL or RUN_SYNC_BASE_URL, or set REQUIRE_RUN_SYNC=0 for local/debug runs." >&2
  exit 2
fi

if [[ -z "${RESUME}" ]] && { [[ -f "${OUTPUT_DIR}/RUN_ID.json" ]] || [[ -f "${OUTPUT_DIR}/logs/events.jsonl" ]] || compgen -G "${CKPT_STORE}/*.pt" >/dev/null; }; then
  echo "[setup] ERROR: refusing to reuse an existing fresh-run output/checkpoint directory." >&2
  echo "[setup] Choose a new RUN_ID/OUTPUT_DIR or set RESUME to an exact checkpoint." >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}/logs" "${CKPT_STORE}" "${TRITON_CACHE_DIR}" "${TMPDIR}" "${PIP_CACHE_DIR}"
if [[ -n "${BITNET_AZURE_LOG_ROOT:-}" ]]; then
  mkdir -p "${BITNET_AZURE_LOG_ROOT}"
fi
LAUNCHER_LOG="${OUTPUT_DIR}/logs/launcher.log"
touch "${LAUNCHER_LOG}"
if [[ -n "${BITNET_AZURE_LOG_ROOT:-}" ]]; then
  LAUNCHER_AZURE_LOG="${BITNET_AZURE_LOG_ROOT}/launcher.log"
  touch "${LAUNCHER_AZURE_LOG}"
  exec > >(tee -a "${LAUNCHER_LOG}" "${LAUNCHER_AZURE_LOG}") 2>&1
else
  exec > >(tee -a "${LAUNCHER_LOG}") 2>&1
fi

AZCOPY_BIN="${AZCOPY_BIN:-}"
RUN_SYNC_PID=""
TRAINING_COMPLETED=0

ensure_azcopy() {
  if [[ -n "${AZCOPY_BIN}" && -x "${AZCOPY_BIN}" ]]; then
    return 0
  fi
  if command -v azcopy >/dev/null 2>&1; then
    AZCOPY_BIN="$(command -v azcopy)"
    return 0
  fi
  local install_dir="${RUNTIME_ROOT}/bin"
  local tmp_dir="${RUNTIME_ROOT}/tmp/azcopy-install"
  mkdir -p "${install_dir}" "${tmp_dir}"
  curl -fsSL "https://aka.ms/downloadazcopy-v10-linux" -o "${tmp_dir}/azcopy.tgz"
  tar -xzf "${tmp_dir}/azcopy.tgz" -C "${tmp_dir}"
  cp "${tmp_dir}"/azcopy_linux_amd64_*/azcopy "${install_dir}/azcopy"
  chmod +x "${install_dir}/azcopy"
  AZCOPY_BIN="${install_dir}/azcopy"
}

blob_sync_once() {
  [[ -z "${RUN_SYNC_URL}" ]] && return 0
  ensure_azcopy || {
    echo "[sync] WARN: azcopy unavailable; sync skipped"
    return 1
  }
  local status=0
  echo "[sync] sync start $(date -Iseconds)"
  # This file is the launcher's own tee target.  It changes while AzCopy reports
  # progress, which makes an otherwise successful sync return
  # CompletedWithErrors. Events, diagnostics, manifests and checkpoints remain
  # in the run sync; the outer launch log is uploaded after the process exits.
  "${AZCOPY_BIN}" sync "${OUTPUT_DIR}" "${RUN_SYNC_URL}" \
    --recursive \
    --exclude-pattern "*.pt;*.tmp;checkpoints/*;torchinductor_cache/*;logs/launcher.log" \
    --exclude-path "logs/launcher.log" \
    --log-level ERROR || {
      echo "[sync] WARN: run-dir sync failed"
      status=1
    }
  if [[ "${RUN_SYNC_CHECKPOINTS}" == "1" ]]; then
    local checkpoint_sync_url
    checkpoint_sync_url="$(append_blob_url_path "${RUN_SYNC_URL}" "checkpoints")"
    "${AZCOPY_BIN}" sync "${CKPT_STORE}" "${checkpoint_sync_url}" \
      --recursive \
      --exclude-pattern "*.tmp" \
      --log-level ERROR || {
        echo "[sync] WARN: checkpoint sync failed"
        status=1
      }
  fi
  echo "[sync] sync end $(date -Iseconds)"
  return "${status}"
}

append_blob_url_path() {
  local url="$1"
  local suffix="${2#/}"
  local base="${url%%\?*}"
  if [[ "${url}" == *"?"* ]]; then
    local query="${url#*\?}"
    printf '%s/%s?%s' "${base%/}" "${suffix}" "${query}"
  else
    printf '%s/%s' "${base%/}" "${suffix}"
  fi
}

start_blob_sync_loop() {
  [[ -z "${RUN_SYNC_URL}" ]] && return 0
  if ! blob_sync_once; then
    if [[ "${REQUIRE_RUN_SYNC}" == "1" ]]; then
      echo "[sync] ERROR: initial run sync failed; refusing to train without recovery artifacts" >&2
      return 1
    fi
    echo "[sync] WARN: initial sync failed; continuing because REQUIRE_RUN_SYNC=0" >&2
  fi
  (
    while true; do
      sleep "${RUN_SYNC_INTERVAL_SECONDS}"
      blob_sync_once || true
    done
  ) &
  RUN_SYNC_PID="$!"
  echo "[sync] periodic sync pid=${RUN_SYNC_PID} interval=${RUN_SYNC_INTERVAL_SECONDS}s checkpoints=${RUN_SYNC_CHECKPOINTS}"
}

stop_blob_sync_loop() {
  if [[ -n "${RUN_SYNC_PID}" ]]; then
    kill "${RUN_SYNC_PID}" >/dev/null 2>&1 || true
    wait "${RUN_SYNC_PID}" >/dev/null 2>&1 || true
    RUN_SYNC_PID=""
  fi
}

ensure_confidential_gpu_ready() {
  [[ "${NVIDIA_CONF_COMPUTE_SET_READY}" == "1" ]] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local cc_state
  cc_state="$(nvidia-smi conf-compute -q 2>/dev/null || true)"
  [[ -n "${cc_state}" ]] || return 0
  if grep -q "CC GPUs Ready State[[:space:]]*:[[:space:]]*Not Ready" <<<"${cc_state}"; then
    echo "[gpu] confidential-compute ready state is Not Ready; setting ready state"
    sudo -n nvidia-smi conf-compute -srs 1
    nvidia-smi conf-compute -q | grep -E "CC State|CC GPUs Ready State" || true
  else
    echo "[gpu] confidential-compute ready state already OK"
  fi
}

patch_fla_tilelang_guard() {
  python - <<'PY'
from pathlib import Path
import sysconfig

path = Path(sysconfig.get_paths()["purelib"]) / "fla/ops/common/backends/tilelang/__init__.py"
if not path.exists():
    print("[fla] tilelang backend guard patch skipped: package file not found")
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
if replacement in text:
    print("[fla] tilelang backend guard patch already present")
elif needle in text:
    path.write_text(text.replace(needle, replacement), encoding="utf-8")
    print("[fla] patched tilelang backend guard to honor FLA_TILELANG=0 before import")
else:
    print("[fla] WARN: tilelang backend guard patch pattern not found")
PY
}

ensure_python_venv_support() {
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  if "${PYTHON_BIN}" -m venv "${tmp_dir}/probe" >/dev/null 2>&1; then
    rm -rf "${tmp_dir}"
    return 0
  fi
  rm -rf "${tmp_dir}"
  if ! sudo -n true >/dev/null 2>&1; then
    echo "[setup] ERROR: python venv support is missing and passwordless sudo is unavailable." >&2
    echo "[setup] Install python3.10-venv (or the matching python3-venv package) before launching." >&2
    exit 2
  fi
  echo "[setup] installing python3.10-venv for launcher-managed virtualenvs"
  sudo apt-get update -y
  sudo apt-get install -y python3.10-venv
}

finish() {
  local exit_code="$?"
  local final_sync_status=0
  trap - EXIT
  set +e
  echo "${exit_code}" > "${OUTPUT_DIR}/logs/exit_code.txt"
  stop_blob_sync_loop
  blob_sync_once || final_sync_status="$?"
  if [[ "${final_sync_status}" != "0" ]]; then
    echo "[sync] ERROR: final run sync failed" >&2
    if [[ "${exit_code}" == "0" && "${REQUIRE_RUN_SYNC}" == "1" ]]; then
      exit_code=1
    fi
  fi
  echo "[deallocate] disabled; launcher never deallocates the VM"
  exit "${exit_code}"
}
trap finish EXIT

echo "[setup] repo=$(pwd)"
echo "[setup] run_id=${RUN_ID}"
echo "[setup] launched_at=$(date -Iseconds)"
echo "[setup] target_tokens=${TOTAL_TOKENS} seq_len=${SEQ_LEN} batch=${BATCH_SIZE} tokens_per_step=${TOKENS_PER_STEP}"
echo "[setup] schedule_horizon_tokens=${SCHEDULE_HORIZON_TOKENS:-<run_budget>}"
echo "[setup] dtype=${DTYPE} disable_cuda_amp=${DISABLE_CUDA_AMP:-0} loss_scale_init=${LOSS_SCALE_INIT} min_loss_scale=${MIN_LOSS_SCALE}"
echo "[setup] train_memmap=${COSMO_BIN} validation_memmap=${VAL_BIN}"
echo "[setup] activation_quantization=${ENABLE_ACTIVATION_QUANTIZATION}"
echo "[setup] compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} loss_chunk_size=${LOSS_CHUNK_SIZE} use_subln=${USE_SUBLN}"
echo "[setup] compile_mode=${COMPILE_MODE}"
echo "[setup] research_mode=${RESEARCH_MODE} attention_backend=${ATTENTION_BACKEND} strict_no_fallback=1"
echo "[setup] residual_norm=affine_rms subln=single_scale_rms qk_norm=unit_rms ffn_activation=${FFN_ACTIVATION} embedding_projection=${USE_EMBEDDING_PROJECTION}"
echo "[setup] attention_preflight_backends=${ATTENTION_PREFLIGHT_BACKENDS} lengths=${ATTENTION_PREFLIGHT_LENGTHS} scaling_lengths=${ATTENTION_SCALING_LENGTHS}"
echo "[setup] inference_cache_preflight=${RUN_INFERENCE_CACHE_PREFLIGHT}"
echo "[setup] transformer_layers=${TRANSFORMER_LAYER_INDICES} global_attention_every=${GLOBAL_ATTENTION_EVERY} local_attention_window=${LOCAL_ATTENTION_WINDOW}"
echo "[setup] log_interval=${LOG_INTERVAL} bootstrap_log_steps=${BOOTSTRAP_LOG_STEPS} val_interval=${VAL_INTERVAL} val_batches=${VAL_BATCHES} flip_interval=${FLIP_INTERVAL} no_progress_bar=${NO_PROGRESS_BAR}"
echo "[setup] grad_detail_interval=${GRAD_DETAIL_INTERVAL} optimizer_stats_interval=${OPTIMIZER_STATS_INTERVAL} system_interval=${SYSTEM_INTERVAL}"
echo "[setup] quant_error_interval=${QUANT_ERROR_INTERVAL} activation_stats_interval=${ACTIVATION_STATS_INTERVAL} ssm_stats_interval=${SSM_STATS_INTERVAL} attention_stats_interval=${ATTENTION_STATS_INTERVAL} attention_stats_mode=${ATTENTION_STATS_MODE} dynamics_interval=${DYNAMICS_INTERVAL}"
echo "[setup] checkpoint_interval=${CHECKPOINT_INTERVAL} checkpoint_tokens=${CHECKPOINT_TOKENS} run_sync_interval_seconds=${RUN_SYNC_INTERVAL_SECONDS}"
echo "[setup] update_sample_size=${UPDATE_SAMPLE_SIZE}"
echo "[setup] fixed_val_slices=${FIXED_VAL_SLICES} milestone_eval_batches=${MILESTONE_EVAL_BATCHES} milestone_sample_max_new_tokens=${MILESTONE_SAMPLE_MAX_NEW_TOKENS}"
echo "[setup] transformer_layer_indices=${TRANSFORMER_LAYER_INDICES}"
echo "[setup] alpha_mode=${ALPHA_MODE} transformer_alpha_mode=${TRANSFORMER_ALPHA_MODE} alpha_group_size=${ALPHA_GROUP_SIZE} alpha_lr=${ALPHA_LR} reg=${ALPHA_REG_LAMBDA} hard_gradient_clipping=disabled"
echo "[setup] decomp_interval=${QUANTIZATION_DECOMPOSITION_INTERVAL} functional_quant_interval=${FUNCTIONAL_QUANTIZATION_INTERVAL}"
echo "[setup] safety auto_abort=disabled reactive_lr_guards=disabled"
echo "[setup] external_watchdog=disabled forced_env_BITNET_EXTERNAL_WATCHDOG=${BITNET_EXTERNAL_WATCHDOG}"
echo "[setup] lr_schedule=${LR_SCHEDULE} warmup_fraction=${WARMUP_FRACTION} decay_fraction=${DECAY_FRACTION} final_lr_fraction=${FINAL_LR_FRACTION}"
echo "[setup] linear_lr_halfway_fraction=${LINEAR_LR_HALFWAY_FRACTION} weight_decay_schedule=constant_through_100_percent muon_weight_decay=${MUON_WEIGHT_DECAY} adamw_weight_decay=${ADAMW_WEIGHT_DECAY}"
echo "[setup] muon_lr=${MUON_LR}"
echo "[setup] adamw_lr=${ADAMW_LR}"
echo "[setup] norm_lr=${NORM_LR} split_norm_parameters=${SPLIT_NORM_PARAMETERS}"
echo "[setup] output_dir=${OUTPUT_DIR}"
if [[ -n "${RUN_SYNC_URL}" ]]; then
  echo "[sync] enabled destination=redacted run_name=${RUN_NAME}"
else
echo "[sync] disabled"
fi
echo "[deallocate] disabled"

ensure_python_venv_support
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python_dist_version_is() {
  local dist_name="$1"
  local expected="$2"
  python - "$dist_name" "$expected" <<'PY' >/dev/null 2>&1
from importlib import metadata
import sys

dist_name = sys.argv[1]
expected = sys.argv[2]
try:
    version = metadata.version(dist_name)
except metadata.PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if version == expected else 1)
PY
}

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  python -m pip install -q -U pip
  if ! python_dist_version_is "torch" "${TORCH_EXPECTED_VERSION}"; then
    python -m pip install -q --index-url "${PYTORCH_INDEX_URL}" "${TORCH_INSTALL_SPEC}"
  fi
  if ! python_dist_version_is "triton" "${TRITON_EXPECTED_VERSION}"; then
    python -m pip install -q "${TRITON_INSTALL_SPEC}"
  fi
  python -m pip install -q "${NUMPY_SPEC}" "${DATASETS_SPEC}" "${TRANSFORMERS_SPEC}" \
    "${TOKENIZERS_SPEC}" "${TQDM_SPEC}" "${SENTENCEPIECE_SPEC}"
  python -m pip install -q --no-deps "${FLA_CORE_SPEC}" "${FLASH_LINEAR_ATTENTION_SPEC}" "${TILELANG_SPEC}"
  python -m pip install -q --no-deps \
    "muon-optimizer @ git+https://github.com/KellerJordan/Muon.git@${MUON_EXPECTED_COMMIT}"
fi
python - "${MUON_EXPECTED_COMMIT}" <<'PY'
import json
from importlib import metadata
import sys

expected = sys.argv[1]
dist = metadata.distribution("muon-optimizer")
direct_url = dist.read_text("direct_url.json")
if not direct_url:
    raise SystemExit("[muon] installed distribution has no direct_url.json commit provenance")
payload = json.loads(direct_url)
actual = payload.get("vcs_info", {}).get("commit_id")
if actual != expected:
    raise SystemExit(f"[muon] expected commit {expected}, got {actual}")
print(f"[muon] verified commit {actual}")
PY
patch_fla_tilelang_guard

if [[ "${ATTENTION_PREFLIGHT_BACKENDS}" == *fa3* || "${ATTENTION_BACKEND}" == "fa3" ]]; then
  [[ -n "${FA3_HOPPER_PATH}" ]] || {
    echo "[fa3] ERROR: FA3_HOPPER_PATH is required for the pinned FA3 production/preflight path." >&2
    exit 2
  }
  actual_fa3_commit="$(git -C "${FA3_HOPPER_PATH}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${actual_fa3_commit}" != "${FA3_EXPECTED_COMMIT}" ]]; then
    echo "[fa3] ERROR: expected commit ${FA3_EXPECTED_COMMIT}, got ${actual_fa3_commit:-unavailable}." >&2
    exit 2
  fi
fi

ensure_confidential_gpu_ready

python - <<'PY'
import torch
print("[gpu] cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Stop before training.")
print("[gpu] bf16_supported:", torch.cuda.is_bfloat16_supported() if hasattr(torch.cuda, "is_bfloat16_supported") else torch.cuda.is_bf16_supported())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"[gpu] {index}: {props.name}, {props.total_memory / 1e9:.2f} GB")
PY

case "${ATTENTION_BACKEND}" in
  fa3|flex) ;;
  *)
    echo "[attention] ERROR: production launcher accepts only fa3 or flex; got ${ATTENTION_BACKEND}" >&2
    exit 2
    ;;
esac

echo "[contracts] running quantization, cache, checkpoint, and static contract tests"
python -m unittest discover -s tests -p 'test_*.py' -v
if [[ -n "${BITNET_EXTRA_CONTRACT_TEST_DIR:-}" ]]; then
  echo "[contracts] running extra contract tests from ${BITNET_EXTRA_CONTRACT_TEST_DIR}"
  python -m unittest discover -s "${BITNET_EXTRA_CONTRACT_TEST_DIR}" -p 'test_*.py' -v
fi

echo "[attention] running strict numerical, boundary, GQA backward, and sparsity preflight"
python attention_preflight.py \
  --backends "${ATTENTION_PREFLIGHT_BACKENDS}" \
  --lengths "${ATTENTION_PREFLIGHT_LENGTHS}" \
  --window "${LOCAL_ATTENTION_WINDOW}" \
  --dtype "${DTYPE}" \
  --temperature "${ATTENTION_TEMPERATURE_INIT}" \
  --softcap "${ATTENTION_SOFTCAP}" \
  --scaling-lengths "${ATTENTION_SCALING_LENGTHS}" \
  --run-scaling \
  --json-output "${OUTPUT_DIR}/logs/attention_preflight.json"

if [[ "${RUN_INFERENCE_CACHE_PREFLIGHT}" == "1" ]]; then
  echo "[inference] running cached local/global parity preflight"
  python inference_cache_preflight.py \
    --backends "${ATTENTION_PREFLIGHT_BACKENDS}" \
    --dtype "${DTYPE}" \
    --temperature-mode "${ATTENTION_TEMPERATURE_MODE}" \
    --softcap "${ATTENTION_SOFTCAP}" \
    --json-output "${OUTPUT_DIR}/logs/inference_cache_preflight.json"
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[preflight] all requested deterministic gates passed; PREFLIGHT_ONLY=1, training not started"
  exit 0
fi

python - <<PYEOF
import json
import hashlib
import os
import numpy as np

cosmo = "${COSMO_BIN}"
val = "${VAL_BIN}"
fineweb = "${FINEWEB_BIN}"
total = int("${TOTAL_TOKENS}")
if fineweb and not os.path.exists(fineweb):
    raise SystemExit(f"FINEWEB_BIN was set but does not exist: {fineweb}")
if os.path.samefile(cosmo, val):
    raise SystemExit("COSMO_BIN and VAL_BIN must be disjoint files")
cosmo_n = np.memmap(cosmo, dtype=np.uint16, mode="r").shape[0]
val_n = np.memmap(val, dtype=np.uint16, mode="r").shape[0]
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

cosmo_sha256 = sha256_file(cosmo)
val_sha256 = sha256_file(val)
if val_n < 2_000_001:
    raise SystemExit(f"VAL_BIN has {val_n:,} tokens; need at least 2,000,001")
shards_train = [
    {
        "path": cosmo,
        "source": "cosmopedia_v2_mincols",
        "split": "train",
        "num_tokens": cosmo_n,
        "dtype": "uint16",
        "dataset_name": "cosmopedia",
        "dataset_config": "default",
        "source_split": "train",
        "sha256": cosmo_sha256,
    }
]
blend = "cosmopedia-only"
source_mix = {}
if fineweb and os.path.exists(fineweb):
    fineweb_n = np.memmap(fineweb, dtype=np.uint16, mode="r").shape[0]
    fineweb_sha256 = sha256_file(fineweb)
    shards_train.append(
        {
            "path": fineweb,
            "source": "fineweb_edu",
            "split": "train",
            "num_tokens": fineweb_n,
            "dtype": "uint16",
            "dataset_name": "fineweb-edu",
            "dataset_config": "default",
            "source_split": "train",
            "sha256": fineweb_sha256,
        }
    )
    fineweb_fraction = float("${FINEWEB_FRACTION}")
    if not 0.0 < fineweb_fraction < 1.0:
        raise SystemExit("FINEWEB_FRACTION must be strictly between 0 and 1")
    if abs(fineweb_fraction * 1000 - round(fineweb_fraction * 1000)) > 1e-9:
        raise SystemExit("FINEWEB_FRACTION must be representable in 0.001 increments")
    source_mix = {
        "cosmopedia_v2_mincols": 1.0 - fineweb_fraction,
        "fineweb_edu": fineweb_fraction,
    }
    blend = f"deterministic-weighted-mix:{1.0-fineweb_fraction:.6f}/{fineweb_fraction:.6f}"
manifest = {
    "config": {
        "seq_len": ${SEQ_LEN},
        "batch_size": ${BATCH_SIZE},
        "total_train_tokens": total,
        "cosmopedia_stories_prefix_tokens": 0,
        "validation_tokens": 2_000_000,
        "token_dtype": "uint16",
        "train_split": "train",
        "val_split": "val",
        "prefix_source": "",
        "source_mix": source_mix,
    },
    "shards": shards_train
    + [
        {
            "path": val,
            "source": "cosmopedia_v2_mincols_val",
            "split": "val",
            "num_tokens": val_n,
            "dtype": "uint16",
            "dataset_name": "cosmopedia",
            "dataset_config": "default",
            "source_split": "heldout_validation",
            "sha256": val_sha256,
        }
    ],
    "derived": {"total_train_tokens": total, "blend": blend},
}
with open("${MANIFEST}", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"[manifest] wrote {blend} manifest")
PYEOF

if [[ ! -L "${OUTPUT_DIR}/checkpoints" ]]; then
  ln -sfn "${CKPT_STORE}" "${OUTPUT_DIR}/checkpoints"
fi

start_blob_sync_loop

COMPILE_FLAG=(--compile --compile-mode "${COMPILE_MODE}")
[[ "${COMPILE}" != "1" ]] && COMPILE_FLAG=(--no-compile)
GRADIENT_CHECKPOINTING_FLAG=()
[[ "${GRADIENT_CHECKPOINTING}" == "1" ]] && GRADIENT_CHECKPOINTING_FLAG=(--gradient-checkpointing)
RESEARCH_MODE_FLAG=()
[[ "${RESEARCH_MODE}" != "1" ]] && RESEARCH_MODE_FLAG=(--disable-research-mode)
PROGRESS_BAR_FLAG=()
[[ "${NO_PROGRESS_BAR}" == "1" ]] && PROGRESS_BAR_FLAG=(--no-progress-bar)
ACTIVATION_QUANTIZATION_FLAG=()
[[ "${ENABLE_ACTIVATION_QUANTIZATION}" == "1" ]] && ACTIVATION_QUANTIZATION_FLAG=(--enable-activation-quantization)
SHARD_ALIGNED_ALPHA_FLAG=()
[[ "${REQUIRE_SHARD_ALIGNED_ALPHA_GROUPS}" == "1" ]] && SHARD_ALIGNED_ALPHA_FLAG=(--require-shard-aligned-alpha-groups)
TELEMETRY_FAIL_HARD_FLAG=()
[[ "${TELEMETRY_FAIL_HARD}" == "1" ]] && TELEMETRY_FAIL_HARD_FLAG=(--telemetry-fail-hard)
SPLIT_NORM_FLAG=()
[[ "${SPLIT_NORM_PARAMETERS}" == "1" ]] && SPLIT_NORM_FLAG=(--split-norm-parameters)
SUBLN_FLAG=()
[[ "${USE_SUBLN}" == "1" ]] && SUBLN_FLAG=(--use-subln)
EMBEDDING_PROJECTION_FLAG=()
[[ "${USE_EMBEDDING_PROJECTION}" == "1" ]] && EMBEDDING_PROJECTION_FLAG=(--use-embedding-projection)
RUN_ID_FLAG=(--run-id "${RUN_ID}")
DISABLE_CUDA_AMP_FLAG=()
[[ "${DISABLE_CUDA_AMP:-0}" == "1" ]] && DISABLE_CUDA_AMP_FLAG=(--disable-cuda-amp)

ARGS=(
  --manifest "${MANIFEST}"
  --data-root "${DATA_ROOT}"
  --device cuda
  --data-mode memmap
  --tokenizer-name "${TOKENIZER_NAME}"
  --tokenizer-revision "${TOKENIZER_REVISION}"
  --vocab-size "${VOCAB_SIZE}"
  --dim "${MODEL_DIM}"
  --n-layers "${N_LAYERS}"
  --n-heads "${N_HEADS}"
  --n-kv-heads "${N_KV_HEADS}"
  --ffn-dim "${FFN_DIM}"
  --expand "${EXPAND}"
  --dtype "${DTYPE}"
  "${DISABLE_CUDA_AMP_FLAG[@]}"
  --loss-scale-init "${LOSS_SCALE_INIT}"
  --min-loss-scale "${MIN_LOSS_SCALE}"
  "${ACTIVATION_QUANTIZATION_FLAG[@]}"
  --attention-backend "${ATTENTION_BACKEND}"
  --local-attention-window "${LOCAL_ATTENTION_WINDOW}"
  --global-attention-every "${GLOBAL_ATTENTION_EVERY}"
  --qk-norm-mode "${QK_NORM_MODE}"
  --ffn-activation "${FFN_ACTIVATION}"
  "${EMBEDDING_PROJECTION_FLAG[@]}"
  --attention-temperature-mode "${ATTENTION_TEMPERATURE_MODE}"
  --attention-temperature-min "${ATTENTION_TEMPERATURE_MIN}"
  --attention-temperature-max "${ATTENTION_TEMPERATURE_MAX}"
  --attention-temperature-init "${ATTENTION_TEMPERATURE_INIT}"
  --attention-softcap "${ATTENTION_SOFTCAP}"
  "${SHARD_ALIGNED_ALPHA_FLAG[@]}"
  --transformer-layer-indices "${TRANSFORMER_LAYER_INDICES}"
  --batch-size "${BATCH_SIZE}"
  --max-train-tokens "${TOTAL_TOKENS}"
  --quantization-decomposition-interval "${QUANTIZATION_DECOMPOSITION_INTERVAL}"
  --functional-quantization-interval "${FUNCTIONAL_QUANTIZATION_INTERVAL}"
  --functional-quantization-max-rows "${FUNCTIONAL_QUANTIZATION_MAX_ROWS}"
  --prefetch-batches "${PREFETCH_BATCHES}"
  --val-batches "${VAL_BATCHES}"
  --val-interval "${VAL_INTERVAL}"
  --fixed-val-slices "${FIXED_VAL_SLICES}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --bootstrap-log-steps "${BOOTSTRAP_LOG_STEPS}"
  --flip-interval "${FLIP_INTERVAL}"
  --grad-detail-interval "${GRAD_DETAIL_INTERVAL}"
  --optimizer-stats-interval "${OPTIMIZER_STATS_INTERVAL}"
  --system-interval "${SYSTEM_INTERVAL}"
  --quant-error-interval "${QUANT_ERROR_INTERVAL}"
  --activation-stats-interval "${ACTIVATION_STATS_INTERVAL}"
  --ssm-stats-interval "${SSM_STATS_INTERVAL}"
  --attention-stats-interval "${ATTENTION_STATS_INTERVAL}"
  --attention-stats-mode "${ATTENTION_STATS_MODE}"
  --attention-stats-sample-seq-len "${ATTENTION_STATS_SAMPLE_SEQ_LEN}"
  --attention-stats-query-tile-size "${ATTENTION_STATS_QUERY_TILE_SIZE}"
  --dynamics-interval "${DYNAMICS_INTERVAL}"
  --update-sample-size "${UPDATE_SAMPLE_SIZE}"
  "${TELEMETRY_FAIL_HARD_FLAG[@]}"
  --warmup-fraction "${WARMUP_FRACTION}"
  --lr-schedule "${LR_SCHEDULE}"
  --decay-fraction "${DECAY_FRACTION}"
  --final-lr-fraction "${FINAL_LR_FRACTION}"
  --phase1-fraction-of-total "${PHASE1_FRACTION_OF_TOTAL}"
  --phase2-fraction-of-total "${PHASE2_FRACTION_OF_TOTAL}"
  --phase-lr-start-fraction "${PHASE_LR_START_FRACTION}"
  --phase1-end-lr-fraction "${PHASE1_END_LR_FRACTION}"
  --phase2-peak-lr-fraction "${PHASE2_PEAK_LR_FRACTION}"
  --phase2-peak-ramp-fraction "${PHASE2_PEAK_RAMP_FRACTION}"
  --terminal-cooldown-fraction "${TERMINAL_COOLDOWN_FRACTION}"
  --linear-lr-halfway-fraction "${LINEAR_LR_HALFWAY_FRACTION}"
  --optimizer-kind "${OPTIMIZER_KIND}"
  --muon-weight-decay "${MUON_WEIGHT_DECAY}"
  --adamw-weight-decay "${ADAMW_WEIGHT_DECAY}"
  --residual-init-scale "${RESIDUAL_INIT_SCALE}"
  --muon-lr "${MUON_LR}"
  --adamw-lr "${ADAMW_LR}"
  --alpha-mode "${ALPHA_MODE}"
  --transformer-alpha-mode "${TRANSFORMER_ALPHA_MODE}"
  --alpha-group-size "${ALPHA_GROUP_SIZE}"
  --alpha-lr "${ALPHA_LR}"
  --norm-lr "${NORM_LR}"
  --attention-temperature-lr "${ATTENTION_TEMPERATURE_LR}"
  "${SPLIT_NORM_FLAG[@]}"
  --alpha-reg-lambda "${ALPHA_REG_LAMBDA}"
  --alpha-grad-clip "${ALPHA_GRAD_CLIP}"
  "${SUBLN_FLAG[@]}"
  --milestone-eval-batches "${MILESTONE_EVAL_BATCHES}"
  --milestone-sample-max-new-tokens "${MILESTONE_SAMPLE_MAX_NEW_TOKENS}"
  --milestone-sample-temperature "${MILESTONE_SAMPLE_TEMPERATURE}"
  --milestone-sample-top-k "${MILESTONE_SAMPLE_TOP_K}"
  --milestone-sample-top-p "${MILESTONE_SAMPLE_TOP_P}"
  --stdout-format silent
  --output-dir "${OUTPUT_DIR}"
  "${RUN_ID_FLAG[@]}"
  "${COMPILE_FLAG[@]}"
  "${GRADIENT_CHECKPOINTING_FLAG[@]}"
  "${POST_SSM_NORM_FLAG[@]}"
  "${RESEARCH_MODE_FLAG[@]}"
  "${PROGRESS_BAR_FLAG[@]}"
)

[[ -n "${RESUME}" ]] && ARGS+=(--resume "${RESUME}")
[[ -n "${TIME_LIMIT_HOURS}" ]] && ARGS+=(--time-limit-hours "${TIME_LIMIT_HOURS}")
[[ -n "${MAX_STEPS}" ]] && ARGS+=(--max-steps "${MAX_STEPS}")
[[ -n "${SCHEDULE_HORIZON_TOKENS}" ]] && ARGS+=(--schedule-horizon-tokens "${SCHEDULE_HORIZON_TOKENS}")
[[ -n "${ATTENTION_LOGIT_ABORT_THRESHOLD}" ]] && ARGS+=(--max-attention-logit "${ATTENTION_LOGIT_ABORT_THRESHOLD}")
[[ -n "${MAX_ACTIVATION_ABS}" ]] && ARGS+=(--max-activation-abs "${MAX_ACTIVATION_ABS}")
[[ -n "${MAX_ACTIVATION_MAX_P999_RATIO}" ]] && ARGS+=(--max-activation-max-p999-ratio "${MAX_ACTIVATION_MAX_P999_RATIO}")
[[ -n "${MAX_QUANTIZATION_RELATIVE_MSE}" ]] && ARGS+=(--max-quantization-relative-mse "${MAX_QUANTIZATION_RELATIVE_MSE}")
[[ -n "${MIN_TERNARY_ZERO_FRACTION}" ]] && ARGS+=(--min-ternary-zero-fraction "${MIN_TERNARY_ZERO_FRACTION}")
[[ -n "${MAX_TERNARY_ZERO_FRACTION}" ]] && ARGS+=(--max-ternary-zero-fraction "${MAX_TERNARY_ZERO_FRACTION}")
ARGS+=(--quantization-health-grace-fraction "${QUANTIZATION_HEALTH_GRACE_FRACTION}")
ARGS+=(--activation-baseline-samples "${ACTIVATION_BASELINE_SAMPLES}")
[[ -n "${MAX_ACTIVATION_RMS_BASELINE_RATIO}" ]] && ARGS+=(--max-activation-rms-baseline-ratio "${MAX_ACTIVATION_RMS_BASELINE_RATIO}")
ARGS+=(--telemetry-probe-source "${TELEMETRY_PROBE_SOURCE}")
ARGS+=(--activation-gate-consecutive-breaches "${ACTIVATION_GATE_CONSECUTIVE_BREACHES}")

python "train.py" "${ARGS[@]}"
if [[ -f "analyze_run.py" ]]; then
  python "analyze_run.py" --run-dir "${OUTPUT_DIR}"
else
  echo "[analysis] analyze_run.py not present; skipping post-run analysis"
fi
if [[ -n "${POST_RUN_EVALUATOR}" ]]; then
  if [[ ! -f "${POST_RUN_EVALUATOR}" ]]; then
    echo "[analysis] ERROR: POST_RUN_EVALUATOR not found: ${POST_RUN_EVALUATOR}" >&2
    exit 2
  fi
  echo "[analysis] running frozen post-run evaluator: ${POST_RUN_EVALUATOR}"
  python "${POST_RUN_EVALUATOR}" --run-dir "${OUTPUT_DIR}"
fi
TRAINING_COMPLETED=1
echo "[done] finished_at=$(date -Iseconds)"
