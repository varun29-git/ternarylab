#!/usr/bin/env bash
# Rendered as EC2 user data by the launch command.  This file intentionally
# contains no credentials; the launcher injects a short-lived STS session only
# into the instance's process environment.
set -euo pipefail

exec > >(tee -a /var/log/terminus-openevolve-bootstrap.log) 2>&1

on_bootstrap_error() {
    status=$?
    printf 'bootstrap failure (exit=%s); stopping instance\n' "$status" >&2
    shutdown -h now || true
    exit "$status"
}
trap on_bootstrap_error ERR

: "${TERMINUS_REPO_URL:?missing repository URL}"
: "${TERMINUS_COMMIT:?missing immutable baseline commit}"
: "${AWS_ACCESS_KEY_ID:?missing temporary AWS access key}"
: "${AWS_SECRET_ACCESS_KEY:?missing temporary AWS secret}"
: "${AWS_SESSION_TOKEN:?missing temporary AWS session token}"

ROOT=/opt/terminus-openevolve
BASELINE="$ROOT/baseline"
CANDIDATE="$ROOT/candidate"
RESULTS="$ROOT/results"
mkdir -p "$ROOT"

date -Is > "$ROOT/started_at.txt"
printf '%s\n' "$TERMINUS_COMMIT" > "$ROOT/baseline_commit.txt"
nvidia-smi -q > "$ROOT/h100_nvidia_smi.txt"
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total --format=csv,noheader > "$ROOT/h100_identity.csv"

# The baseline is immutable evidence.  The candidate program is independently
# copied and also made read-only: OpenEvolve produces all mutations under
# RESULTS, never in either copy of the implementation.
git clone --no-checkout "$TERMINUS_REPO_URL" "$BASELINE"
git -C "$BASELINE" checkout --detach "$TERMINUS_COMMIT"
test "$(git -C "$BASELINE" rev-parse HEAD)" = "$TERMINUS_COMMIT"
git -C "$BASELINE" status --porcelain | tee "$ROOT/baseline_git_status.txt"
chmod -R a-w "$BASELINE"

mkdir -p "$CANDIDATE/evolution/packed_ternary_hopper" "$CANDIDATE"
cp "$BASELINE/evolution/packed_ternary_hopper/initial_program.py" "$CANDIDATE/evolution/packed_ternary_hopper/"
cp "$BASELINE/evolution/packed_ternary_hopper/evaluator.py" "$CANDIDATE/evolution/packed_ternary_hopper/"
cp -a "$BASELINE/src" "$CANDIDATE/src"
cp -a "$BASELINE/reports" "$CANDIDATE/reports"
chmod -R a-w "$CANDIDATE"
mkdir -p "$RESULTS"

python3 -m venv --system-site-packages "$ROOT/venv"
source "$ROOT/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'openevolve==0.3.2' boto3
python - <<'PY'
import torch
import triton
assert torch.cuda.is_available(), "CUDA is not available"
major, minor = torch.cuda.get_device_capability()
assert major == 9, f"expected Hopper SM90, got {major}.{minor}"
print({"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
       "device": torch.cuda.get_device_name(), "capability": f"{major}.{minor}"})
PY

cat > "$ROOT/bedrock_openai_proxy.py" <<'PY'
"""Minimal loopback-only OpenAI Chat Completions facade over Bedrock Converse.

It holds only the temporary STS session inherited from bootstrap.  The agent
cannot call AWS directly; OpenEvolve sees only this local endpoint.
"""
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3

CLIENT = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL = "zai.glm-5"


def as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def send_json(self, status, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "unknown endpoint"}})
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system, messages = [], []
            for message in request.get("messages", []):
                text = as_text(message.get("content", ""))
                if message.get("role") == "system":
                    system.append({"text": text})
                else:
                    role = message.get("role", "user")
                    messages.append({"role": "assistant" if role == "assistant" else "user", "content": [{"text": text}]})
            kwargs = {
                "modelId": MODEL,
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": min(int(request.get("max_tokens", 4096)), 12000),
                    "temperature": float(request.get("temperature", 0.6)),
                },
            }
            if system:
                kwargs["system"] = system
            response = CLIENT.converse(**kwargs)
            content = "".join(block.get("text", "") for block in response["output"]["message"]["content"])
            self.send_json(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", MODEL),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": response.get("usage", {}),
            })
        except Exception as error:
            self.send_json(500, {"error": {"message": f"Bedrock proxy failure: {type(error).__name__}: {error}"}})


ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
PY

cat > "$RESULTS/config.yaml" <<'YAML'
max_iterations: 100000
checkpoint_interval: 10
random_seed: 42
llm:
  primary_model: "zai.glm-5"
  primary_model_weight: 1.0
  api_base: "http://127.0.0.1:8080/v1"
  api_key: "loopback-only"
  temperature: 0.6
  max_tokens: 12000
  timeout: 240
  retries: 3
  retry_delay: 8
prompt:
  system_message: |
    You are the optimization researcher for a production packed-ternary transformer on one NVIDIA H100 SXM (SM90, 80 GB).  You may propose novel algorithmic and Triton scheduling workarounds, but only by evolving the candidate program supplied to OpenEvolve.  The original implementation and reports are immutable evidence at /opt/terminus-openevolve/baseline and /opt/terminus-openevolve/candidate.

    Target: implement a strictly correct, faster H100 kernel for BF16 activation x 2-bit-packed ternary weights.  Each uint8 holds four adjacent K-major codes: 00=0, 01=+1, 10=-1, and 11 is forbidden.  One FP32 grouped scale applies to each 128 logical-K weights for an output column.  Inputs/outputs must remain BF16, accumulation FP32, and no dense-weight materialization is allowed.

    The evaluator is the sole authority.  It requires every correctness/tail case before timing and scores the geometric speedup over the best dense BF16 baseline across q/KV/FFN decode (M=1), prefill (M=128), and training-token shapes (M=512/2048), with current flagship dimensions 768/2048/12 heads/4 KV heads.  Favor transferable mechanisms: byte decode reuse, warp specialization, layouts, tiles, launch geometry, L2 locality, pipeline stages, shape dispatch, and register-pressure control.  Never special-case input values or weaken checks.  Do not edit baseline files, reports, evaluator, source tree, or network/process state.  Return only a complete candidate-program change.
database:
  population_size: 48
  archive_size: 24
  num_islands: 4
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.65
  similarity_threshold: 0.995
evaluator:
  timeout: 300
  cascade_evaluation: true
  cascade_thresholds: [1.0]
  parallel_evaluations: 1
  enable_artifacts: true
diff_based_evolution: true
max_code_length: 45000
YAML

python "$ROOT/bedrock_openai_proxy.py" > "$ROOT/bedrock_proxy.log" 2>&1 &
PROXY_PID=$!
trap 'kill "$PROXY_PID" 2>/dev/null || true' EXIT
sleep 2
curl --fail --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"zai.glm-5","messages":[{"role":"user","content":"Reply exactly READY"}],"max_tokens":16,"temperature":0}' \
  > "$ROOT/bedrock_proxy_probe.json"

# Record the same pre-search score that all candidates must beat.  This makes
# later claims traceable even if OpenEvolve is interrupted before iteration 1.
python "$CANDIDATE/evolution/packed_ternary_hopper/evaluator.py" \
  > "$RESULTS/baseline_metrics.json"

set +e
timeout --signal=TERM --kill-after=5m 8h \
  openevolve-run \
    "$CANDIDATE/evolution/packed_ternary_hopper/initial_program.py" \
    "$CANDIDATE/evolution/packed_ternary_hopper/evaluator.py" \
    --config "$RESULTS/config.yaml" \
    --output "$RESULTS/openevolve" \
    --iterations 100000 \
    --log-level INFO \
  > "$RESULTS/openevolve.log" 2>&1
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$RESULTS/openevolve_exit_status.txt"
date -Is > "$RESULTS/finished_at.txt"
find "$RESULTS" -maxdepth 4 -type f -printf '%p\n' | sort > "$RESULTS/artifact_manifest.txt"

# Stop the compute instance when the bounded search ends.  The encrypted root
# EBS volume remains available because launch uses DeleteOnTermination=false.
shutdown -h now
