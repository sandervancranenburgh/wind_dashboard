#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/knmi_notification_listener.log"
VENV="/home/sandervancranenburgh/Documents/python_envs/env"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] KNMI notification listener start"

  if [ -z "${KNMI_API_KEY:-}" ]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ERROR: KNMI_API_KEY is not set"
    exit 2
  fi
  if [ -z "${KNMI_NOTIFICATION_API_KEY:-}" ]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ERROR: KNMI_NOTIFICATION_API_KEY is not set"
    exit 2
  fi
  if [ -z "${KNMI_NOTIFICATION_CLIENT_ID:-}" ]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ERROR: KNMI_NOTIFICATION_CLIENT_ID is not set"
    exit 2
  fi

  if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
  fi

  python3 scripts/knmi_notification_listener.py

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] KNMI notification listener end"
} >> "${LOG_FILE}" 2>&1
