#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_DB="${HOME:?HOME is not set}/Documents/wind_fetcher_data/data/wind_data_all_sites.db"
readonly BACKUP_DIR="${HOME}/Documents/wind_fetcher_data/backups"
readonly LOG_FILE="${BACKUP_DIR}/backup.log"
readonly LOCK_FILE="${BACKUP_DIR}/.backup_database.lock"

DRY_RUN=0

usage() {
  printf 'Usage: %s [--dry-run]\n' "$(basename "$0")"
}

case "${1:-}" in
  "")
    ;;
  --dry-run)
    DRY_RUN=1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if (( $# > 1 )); then
  usage >&2
  exit 2
fi

for command_name in sqlite3 gzip date stat realpath flock; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'ERROR: required command not found: %s\n' "${command_name}" >&2
    exit 1
  fi
done

if [[ ! -f "${SOURCE_DB}" ]]; then
  printf 'ERROR: source database does not exist or is not a regular file: %s\n' "${SOURCE_DB}" >&2
  exit 1
fi
if [[ ! -r "${SOURCE_DB}" ]]; then
  printf 'ERROR: source database is not readable: %s\n' "${SOURCE_DB}" >&2
  exit 1
fi

readonly TODAY="$(date +%F)"
readonly TODAY_EPOCH="$(date -u --date="${TODAY}" +%s)"
readonly BACKUP_BASENAME="wind_data_${TODAY}.db"
readonly COMPRESSED_BASENAME="${BACKUP_BASENAME}.gz"
readonly BACKUP_FILE="${BACKUP_DIR}/${BACKUP_BASENAME}"
readonly COMPRESSED_FILE="${BACKUP_DIR}/${COMPRESSED_BASENAME}"

retention_category() {
  local age_days="$1"

  # Future-dated archives are retained defensively as daily backups.
  if (( age_days <= 2 )); then
    printf 'daily'
  elif (( age_days <= 6 )); then
    printf 'bridge'
  elif (( age_days <= 13 )); then
    printf 'weekly'
  elif (( age_days <= 20 )); then
    printf 'two-week'
  else
    printf 'delete'
  fi
}

validate_compressed_backup_path() {
  local candidate="$1"
  local candidate_parent
  local backup_parent
  local filename

  candidate_parent="$(realpath -m -- "$(dirname -- "${candidate}")")"
  backup_parent="$(realpath -m -- "${BACKUP_DIR}")"
  filename="$(basename -- "${candidate}")"

  [[ "${candidate_parent}" == "${backup_parent}" ]] &&
    [[ "${filename}" =~ ^wind_data_[0-9]{4}-[0-9]{2}-[0-9]{2}\.db\.gz$ ]]
}

remove_compressed_backup() {
  local candidate="$1"

  if ! validate_compressed_backup_path "${candidate}"; then
    printf 'ERROR: refusing to delete unsafe backup path: %s\n' "${candidate}" >&2
    return 1
  fi
  if [[ -L "${candidate}" || ! -f "${candidate}" ]]; then
    printf 'ERROR: refusing to delete a symlink or non-regular backup: %s\n' "${candidate}" >&2
    return 1
  fi

  if (( DRY_RUN )); then
    printf 'DRY RUN: would delete expired backup %s\n' "${candidate}"
  else
    rm -- "${candidate}"
    printf 'Deleted expired backup %s\n' "${candidate}"
  fi
}

apply_retention() {
  local candidate
  local filename
  local backup_date
  local backup_epoch
  local age_days
  local category
  local action

  [[ -d "${BACKUP_DIR}" ]] || return 0

  # Keep daily backups through age 2, bridge copies through age 6, the weekly
  # recovery window through age 13, and the two-week window through age 20.
  # Only valid, exactly named archives aged 21 days or more are deleted.
  shopt -s nullglob
  for candidate in "${BACKUP_DIR}"/wind_data_????-??-??.db.gz; do
    filename="$(basename -- "${candidate}")"

    if [[ ! "${filename}" =~ ^wind_data_([0-9]{4}-[0-9]{2}-[0-9]{2})\.db\.gz$ ]]; then
      continue
    fi
    backup_date="${BASH_REMATCH[1]}"
    if ! backup_epoch="$(date -u --date="${backup_date}" +%s 2>/dev/null)"; then
      continue
    fi

    age_days=$(((TODAY_EPOCH - backup_epoch) / 86400))
    category="$(retention_category "${age_days}")"
    action="KEEP"
    if [[ "${category}" == "delete" ]]; then
      action="DELETE"
    fi

    if (( DRY_RUN )); then
      printf 'DRY RUN: filename=%s age_days=%s category=%s action=%s\n' "${filename}" "${age_days}" "${category}" "${action}"
    fi

    if [[ "${action}" == "DELETE" ]]; then
      remove_compressed_backup "${candidate}"
    fi
  done
  shopt -u nullglob
}

if (( DRY_RUN )); then
  if [[ -e "${BACKUP_FILE}" || -L "${BACKUP_FILE}" ||
        -e "${COMPRESSED_FILE}" || -L "${COMPRESSED_FILE}" ]]; then
    printf 'DRY RUN: no files will be created, changed, or deleted.\n'
    apply_retention
    printf 'ERROR: today'\''s backup already exists; it would not be overwritten: %s\n' \
      "${COMPRESSED_FILE}" >&2
    exit 1
  fi

  printf 'DRY RUN: no files will be created, changed, or deleted.\n'
  if [[ ! -d "${BACKUP_DIR}" ]]; then
    printf 'DRY RUN: would create backup directory %s\n' "${BACKUP_DIR}"
  fi
  printf 'DRY RUN: would snapshot %s to %s using sqlite3 .backup\n' \
    "${SOURCE_DB}" "${BACKUP_FILE}"
  printf 'DRY RUN: would require PRAGMA quick_check to return exactly "ok"\n'
  printf 'DRY RUN: would compress the snapshot to %s\n' "${COMPRESSED_FILE}"
  apply_retention
  printf 'DRY RUN: would append one SUCCESS or FAILURE line to %s\n' "${LOG_FILE}"
  exit 0
fi

mkdir -p -- "${BACKUP_DIR}"

START_EPOCH="$(date +%s)"
OUTCOME="FAILURE"
CLEANUP_CURRENT=0

finish() {
  local exit_status=$?
  local end_epoch
  local duration
  local compressed_size=0
  local timestamp

  trap - EXIT
  set +e

  if (( exit_status != 0 && CLEANUP_CURRENT )); then
    # A failed SQLite snapshot is the sole uncompressed file this script removes.
    # Its exact directory and date-specific name are fixed above.
    if [[ -f "${BACKUP_FILE}" && ! -L "${BACKUP_FILE}" &&
          "$(realpath -m -- "$(dirname -- "${BACKUP_FILE}")")" == "$(realpath -m -- "${BACKUP_DIR}")" &&
          "$(basename -- "${BACKUP_FILE}")" =~ ^wind_data_[0-9]{4}-[0-9]{2}-[0-9]{2}\.db$ ]]; then
      rm -- "${BACKUP_FILE}"
    fi
    if [[ -f "${COMPRESSED_FILE}" && ! -L "${COMPRESSED_FILE}" ]]; then
      remove_compressed_backup "${COMPRESSED_FILE}"
    fi
  fi

  if [[ -f "${COMPRESSED_FILE}" && ! -L "${COMPRESSED_FILE}" ]]; then
    compressed_size="$(stat --format='%s' -- "${COMPRESSED_FILE}" 2>/dev/null || printf '0')"
  fi
  end_epoch="$(date +%s)"
  duration=$((end_epoch - START_EPOCH))
  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  if ! printf '%s status=%s filename=%s compressed_size_bytes=%s duration_seconds=%s\n' \
    "${timestamp}" "${OUTCOME}" "${COMPRESSED_BASENAME}" \
    "${compressed_size}" "${duration}" >> "${LOG_FILE}"; then
    printf 'ERROR: could not append backup result to %s\n' "${LOG_FILE}" >&2
    exit_status=1
  fi

  exit "${exit_status}"
}
trap finish EXIT

# Prevent a manual run from racing the nightly cron job.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'ERROR: another database backup is already running.\n' >&2
  exit 1
fi

if [[ -e "${BACKUP_FILE}" || -L "${BACKUP_FILE}" ||
      -e "${COMPRESSED_FILE}" || -L "${COMPRESSED_FILE}" ]]; then
  printf 'ERROR: today'\''s backup already exists; refusing to overwrite it: %s\n' \
    "${COMPRESSED_FILE}" >&2
  exit 1
fi

CLEANUP_CURRENT=1
printf 'Creating SQLite snapshot %s\n' "${BACKUP_FILE}"
if ! sqlite3 "${SOURCE_DB}" ".backup '${BACKUP_FILE}'"; then
  printf 'ERROR: sqlite3 .backup failed for %s\n' "${BACKUP_FILE}" >&2
  exit 1
fi

if ! QUICK_CHECK_RESULT="$(sqlite3 "${BACKUP_FILE}" 'PRAGMA quick_check;')"; then
  printf 'ERROR: PRAGMA quick_check could not be completed for %s\n' "${BACKUP_FILE}" >&2
  exit 1
fi
if [[ "${QUICK_CHECK_RESULT}" != "ok" ]]; then
  printf 'ERROR: PRAGMA quick_check failed for %s (result: %q)\n' \
    "${BACKUP_FILE}" "${QUICK_CHECK_RESULT}" >&2
  exit 1
fi

printf 'Compressing verified snapshot to %s\n' "${COMPRESSED_FILE}"
if ! gzip -- "${BACKUP_FILE}"; then
  printf 'ERROR: gzip failed for %s\n' "${BACKUP_FILE}" >&2
  exit 1
fi
CLEANUP_CURRENT=0

apply_retention

OUTCOME="SUCCESS"
printf 'Backup completed successfully: %s\n' "${COMPRESSED_FILE}"
