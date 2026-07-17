#!/usr/bin/env bash
set -u

lock_file="${WIND_UPDATER_LOCK_FILE:-/tmp/wind_pipeline_heavy_updater.lock}"
exec 9>"${lock_file}"

if ! flock -n 9; then
    printf '%s execution_mode=skipped_lock_held lock=%s\n' "$(date --iso-8601=seconds)" "${lock_file}" >&2
    exit 0
fi

if [ "$#" -eq 0 ]; then
    printf 'usage: %s COMMAND [ARG ...]\n' "$0" >&2
    exit 64
fi

exec "$@"
