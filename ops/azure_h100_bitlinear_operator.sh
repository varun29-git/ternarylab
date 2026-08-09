#!/usr/bin/env bash
# Local Azure CLI entrypoint.  It never prints Azure account details or reads
# credentials; Azure CLI performs its normal authenticated transport.
set -euo pipefail
set +x

: "${AZURE_RESOURCE_GROUP:?set the Azure resource group name}"
: "${AZURE_VM_NAME:?set the Azure VM name}"
: "${AZURE_VM_USER:?set the Azure VM admin username}"
: "${TERMINUS_REMOTE_REPOSITORY:?set the absolute repository path on the VM}"
: "${TERMINUS_REMOTE_RESULTS:?set a writable absolute results directory on the VM}"

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-559d79fb6994a58b8a15b4b93bf13ccc16edf247}"
AZURE_VM_START_TIMEOUT_S="${AZURE_VM_START_TIMEOUT_S:-1800}"
AZURE_VM_POLL_INTERVAL_S="${AZURE_VM_POLL_INTERVAL_S:-15}"

command -v az >/dev/null 2>&1 || { echo "Azure CLI is required" >&2; exit 1; }

power_state() {
  az vm get-instance-view \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --name "$AZURE_VM_NAME" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState/')].code | [0]" \
    --output tsv
}

current_power_state="$(power_state)"
if [[ "$current_power_state" != "PowerState/running" ]]; then
  az vm start --resource-group "$AZURE_RESOURCE_GROUP" --name "$AZURE_VM_NAME" --no-wait
  deadline=$((SECONDS + AZURE_VM_START_TIMEOUT_S))
  while true; do
    sleep "$AZURE_VM_POLL_INTERVAL_S"
    current_power_state="$(power_state)"
    if [[ "$current_power_state" == "PowerState/running" ]]; then
      break
    fi
    if (( SECONDS >= deadline )); then
      echo "Azure VM did not reach PowerState/running within ${AZURE_VM_START_TIMEOUT_S}s." >&2
      exit 1
    fi
  done
fi

remote_command="bash -s -- $(printf '%q' "$TERMINUS_REMOTE_REPOSITORY") $(printf '%q' "$TERMINUS_REMOTE_RESULTS") $(printf '%q' "$DEEPGEMM_COMMIT")"
az ssh vm --resource-group "$AZURE_RESOURCE_GROUP" --name "$AZURE_VM_NAME" --local-user "$AZURE_VM_USER" -- "$remote_command" < "$SCRIPT_DIRECTORY/azure_h100_bitlinear_remote_gate.sh"
