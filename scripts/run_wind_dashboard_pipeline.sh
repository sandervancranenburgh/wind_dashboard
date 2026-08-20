#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${WIND_FETCHER_PYTHON_BIN:-/home/sandervancranenburgh/Documents/python_envs/env/bin/python}"

cd "${REPO_ROOT}"

# Hold the existing updater lock across the complete fetch -> gate sequence.
# The marker prevents recursion after the lock runner re-executes this script.
if [ "${WIND_DASHBOARD_PIPELINE_LOCK_HELD:-0}" != "1" ]; then
    exec env WIND_DASHBOARD_PIPELINE_LOCK_HELD=1 \
        "${SCRIPT_DIR}/run_wind_updater_locked.sh" "${BASH_SOURCE[0]}"
fi

# Prediction/dashboard work is deliberately conditional on a completely
# successful observation + production HARMONIE fetch and database write.
"${PYTHON_BIN}" source_fetch.py data

exec "${PYTHON_BIN}" next_day_wind_model/update_model_and_predict.py \
    --db data/wind_data_all_sites.db \
    --window-hours 72 \
    --skip-training \
    --skip-data-refresh-check \
    --web-out-dir docs \
    --current-day-interval-minutes 6 \
    --plot-update-interval-minutes 6 \
    --harmonie-update-interval-minutes 60 \
    --git-auto-push-pages \
    --git-remote origin \
    --git-branch main \
    --companion-app-base-url https://portal-cityailab.tbm.tudelft.nl
