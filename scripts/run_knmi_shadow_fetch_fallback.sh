#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/knmi_shadow_fetch_fallback.log"
VENV="/home/sandervancranenburgh/Documents/python_envs/env"
LATEST_COUNT="${KNMI_FALLBACK_LATEST_COUNT:-3}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] KNMI shadow fallback fetch start"

  if [ -z "${KNMI_API_KEY:-}" ]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ERROR: KNMI_API_KEY is not set"
    exit 2
  fi

  if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
  fi

  python3 scripts/knmi_extract_latest_to_db.py --latest-count "${LATEST_COUNT}"

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] KNMI shadow fallback fetch end"
} >> "${LOG_FILE}" 2>&1
