#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  setup_rider_portal_preview.sh [OPTIONS] [PRODUCTION_REPOSITORY]
  setup_rider_portal_preview.sh --undo [--dry-run]

Configure the current Git worktree to use Rider Portal data and artefacts from
the production repository, or remove links previously created by this helper.

Options:
  --dry-run      Report planned changes without modifying the worktree.
  --force        Allow existing non-symlink files or directories to be replaced.
  --link-secret  Also link data/.wind_dashboard_secret from production.
  --undo         Remove setup-managed links and restore tracked files with git restore.
  --help         Show this help message.

Without an explicit production path, the helper first looks for exactly one
other worktree on the main branch, then falls back to:
  ~/Documents/repos/wind_fetcher2
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

expand_home_path() {
  local value="$1"
  case "$value" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${value:2}" ;;
    *) printf '%s\n' "$value" ;;
  esac
}

dry_run=false
force=false
link_secret=false
undo=false
production_argument=""

while (( $# > 0 )); do
  case "$1" in
    --dry-run) dry_run=true ;;
    --force) force=true ;;
    --link-secret) link_secret=true ;;
    --undo) undo=true ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while (( $# > 0 )); do
        [[ -z "$production_argument" ]] || die "Only one production repository path may be provided."
        production_argument="$1"
        shift
      done
      break
      ;;
    -*) die "Unknown option: $1 (use --help for usage)" ;;
    *)
      [[ -z "$production_argument" ]] || die "Only one production repository path may be provided."
      production_argument="$1"
      ;;
  esac
  shift
done

if ! current_git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  die "Run this script from inside a Git worktree."
fi
worktree_root="$(cd "$current_git_root" && pwd -P)"
state_path="$worktree_root/.rider_portal_preview_state"

all_managed_paths=(
  "data/wind_data_all_sites.db"
  "data/.wind_dashboard_secret"
  "data/rider_activity_uploads"
  "data/rider_activity_analysis"
  "next_day_wind_model/artifacts/current_day_plot_archive"
)

is_allowed_managed_path() {
  local candidate="$1"
  local allowed
  for allowed in "${all_managed_paths[@]}"; do
    [[ "$candidate" == "$allowed" ]] && return 0
  done
  return 1
}

ensure_safe_parent() {
  local relative_parent="$1"
  local current="$worktree_root"
  local component
  local -a components=()

  IFS='/' read -r -a components <<< "$relative_parent"
  for component in "${components[@]}"; do
    current="$current/$component"
    [[ ! -L "$current" ]] || die "Refusing to use symlinked preview parent directory: $current"
    [[ ! -e "$current" || -d "$current" ]] || die "Preview parent path is not a directory: $current"
  done
}

declare -A recorded_links=()

load_state() {
  local relative_path
  local expected_target

  [[ ! -L "$state_path" ]] || die "Refusing to use symlinked preview state file: $state_path"
  [[ -f "$state_path" ]] || return 0

  while IFS=$'\t' read -r relative_path expected_target; do
    [[ -n "$relative_path" && -n "$expected_target" ]] || continue
    if is_allowed_managed_path "$relative_path"; then
      recorded_links["$relative_path"]="$expected_target"
    else
      printf 'Warning: ignoring unknown path in preview state: %s\n' "$relative_path" >&2
    fi
  done < "$state_path"
}

write_state() {
  local relative_path
  local state_tmp
  local entry_count=0

  for relative_path in "${all_managed_paths[@]}"; do
    if [[ -n "${recorded_links[$relative_path]+present}" ]]; then
      entry_count=$((entry_count + 1))
    fi
  done

  if (( entry_count == 0 )); then
    rm -f -- "$state_path"
    return 0
  fi

  state_tmp="$(mktemp "$worktree_root/.rider_portal_preview_state.tmp.XXXXXX")"
  chmod 600 "$state_tmp"
  for relative_path in "${all_managed_paths[@]}"; do
    if [[ -n "${recorded_links[$relative_path]+present}" ]]; then
      printf '%s\t%s\n' "$relative_path" "${recorded_links[$relative_path]}" >> "$state_tmp"
    fi
  done
  mv -f -- "$state_tmp" "$state_path"
}

is_tracked_path() {
  git -C "$worktree_root" ls-files --error-unmatch -- "$1" >/dev/null 2>&1
}

perform_undo() {
  local relative_path
  local expected_target
  local destination_path
  local relative_parent
  local current_target
  local removed_count=0
  local restored_count=0
  local already_count=0
  local skipped_count=0
  local managed_count=0

  [[ "$force" != true ]] || die "--force cannot be combined with --undo."
  [[ "$link_secret" != true ]] || die "--link-secret cannot be combined with --undo."
  [[ -z "$production_argument" ]] || die "A production repository path is not used with --undo."

  load_state
  for relative_path in "${all_managed_paths[@]}"; do
    [[ -z "${recorded_links[$relative_path]+present}" ]] || managed_count=$((managed_count + 1))
  done

  printf 'Preview worktree: %s\n\n' "$worktree_root"
  if (( managed_count == 0 )); then
    printf 'Undo complete: no setup-managed Rider Portal links were found.\n'
    return 0
  fi

  for relative_path in "${all_managed_paths[@]}"; do
    [[ -n "${recorded_links[$relative_path]+present}" ]] || continue
    expected_target="${recorded_links[$relative_path]}"
    destination_path="$worktree_root/$relative_path"
    relative_parent="${relative_path%/*}"
    ensure_safe_parent "$relative_parent"

    if [[ -L "$destination_path" ]]; then
      current_target="$(readlink -- "$destination_path")"
      if [[ "$current_target" != "$expected_target" ]]; then
        printf 'Leaving changed symlink untouched: %s -> %s\n' "$relative_path" "$current_target"
        skipped_count=$((skipped_count + 1))
        continue
      fi

      printf 'Will remove setup-managed symlink: %s -> %s\n' "$relative_path" "$expected_target"
      if [[ "$dry_run" == true ]]; then
        removed_count=$((removed_count + 1))
        if is_tracked_path "$relative_path"; then
          restored_count=$((restored_count + 1))
          printf 'Would restore tracked path with git restore: %s\n' "$relative_path"
        fi
        continue
      fi

      rm -f -- "$destination_path"
      removed_count=$((removed_count + 1))
      if is_tracked_path "$relative_path"; then
        git -C "$worktree_root" restore --worktree -- "$relative_path"
        restored_count=$((restored_count + 1))
        printf 'Restored tracked path with git restore: %s\n' "$relative_path"
      fi
      unset 'recorded_links[$relative_path]'
      write_state
    elif [[ ! -e "$destination_path" ]]; then
      printf 'Already absent: %s\n' "$relative_path"
      if [[ "$dry_run" == true ]]; then
        already_count=$((already_count + 1))
        if is_tracked_path "$relative_path"; then
          restored_count=$((restored_count + 1))
          printf 'Would restore missing tracked path with git restore: %s\n' "$relative_path"
        fi
        continue
      fi

      if is_tracked_path "$relative_path"; then
        git -C "$worktree_root" restore --worktree -- "$relative_path"
        restored_count=$((restored_count + 1))
        printf 'Restored missing tracked path with git restore: %s\n' "$relative_path"
      fi
      already_count=$((already_count + 1))
      unset 'recorded_links[$relative_path]'
      write_state
    elif is_tracked_path "$relative_path" && git -C "$worktree_root" diff --quiet -- "$relative_path"; then
      printf 'Already restored: %s\n' "$relative_path"
      already_count=$((already_count + 1))
      if [[ "$dry_run" != true ]]; then
        unset 'recorded_links[$relative_path]'
        write_state
      fi
    else
      printf 'Leaving non-symlink path untouched: %s\n' "$relative_path"
      skipped_count=$((skipped_count + 1))
    fi
  done

  printf '\n'
  if [[ "$dry_run" == true ]]; then
    printf 'Dry-run undo summary (no changes made):\n'
  else
    printf 'Undo summary:\n'
  fi
  printf '  Setup-managed symlinks %s: %d\n' \
    "$([[ "$dry_run" == true ]] && printf 'to remove' || printf 'removed')" "$removed_count"
  printf '  Tracked paths %s with git restore: %d\n' \
    "$([[ "$dry_run" == true ]] && printf 'to restore' || printf 'restored')" "$restored_count"
  printf '  Already absent/restored: %d\n' "$already_count"
  printf '  Changed or non-symlink paths left untouched: %d\n' "$skipped_count"
}

if [[ "$undo" == true ]]; then
  perform_undo
  exit 0
fi

detect_main_worktree() {
  local line
  local candidate=""
  local candidate_root
  local -a main_worktrees=()

  while IFS= read -r line; do
    case "$line" in
      "worktree "*) candidate="${line#worktree }" ;;
      "branch refs/heads/main")
        if [[ -n "$candidate" && -d "$candidate" ]]; then
          candidate_root="$(cd "$candidate" && pwd -P)"
          [[ "$candidate_root" == "$worktree_root" ]] || main_worktrees+=("$candidate_root")
        fi
        ;;
    esac
  done < <(git -C "$worktree_root" worktree list --porcelain)

  if (( ${#main_worktrees[@]} == 1 )); then
    printf '%s\n' "${main_worktrees[0]}"
    return 0
  fi
  return 1
}

production_source="explicit"
if [[ -z "$production_argument" ]]; then
  if detected_production="$(detect_main_worktree)"; then
    production_argument="$detected_production"
    production_source="auto-detected main worktree"
  else
    production_argument="$HOME/Documents/repos/wind_fetcher2"
    production_source="default fallback"
  fi
fi
production_argument="$(expand_home_path "$production_argument")"

[[ -d "$production_argument" ]] || die "Production repository does not exist: $production_argument"
if ! production_git_root="$(git -C "$production_argument" rev-parse --show-toplevel 2>/dev/null)"; then
  die "Production path is not inside a Git worktree: $production_argument"
fi
production_root="$(cd "$production_git_root" && pwd -P)"
production_argument_root="$(cd "$production_argument" && pwd -P)"
[[ "$production_argument_root" == "$production_root" ]] || die "Production path must be the repository root: $production_argument"

[[ "$worktree_root" != "$production_root" ]] || die "Refusing to configure the production repository itself: $production_root"
case "$worktree_root/" in
  "$production_root/"*) die "Refusing to configure a worktree located inside the production repository." ;;
esac
case "$production_root/" in
  "$worktree_root/"*) die "Refusing to use a production repository located inside the preview worktree." ;;
esac

selected_paths=(
  "data/wind_data_all_sites.db"
  "data/rider_activity_uploads"
  "data/rider_activity_analysis"
  "next_day_wind_model/artifacts/current_day_plot_archive"
)
selected_types=("file" "directory" "directory" "directory")
if [[ "$link_secret" == true ]]; then
  selected_paths+=("data/.wind_dashboard_secret")
  selected_types+=("file")
fi

missing_required=false
for index in "${!selected_paths[@]}"; do
  source_path="$production_root/${selected_paths[$index]}"
  case "${selected_types[$index]}" in
    file)
      if [[ ! -f "$source_path" ]]; then
        printf 'Error: required production file is missing: %s\n' "$source_path" >&2
        missing_required=true
      fi
      ;;
    directory)
      if [[ ! -d "$source_path" ]]; then
        printf 'Error: required production directory is missing: %s\n' "$source_path" >&2
        missing_required=true
      fi
      ;;
  esac
done
[[ "$missing_required" != true ]] || exit 1

load_state
actions=()
already_count=0
create_count=0
relink_count=0
force_required_count=0

printf 'Preview worktree: %s\n' "$worktree_root"
printf 'Production repository (%s): %s\n' "$production_source" "$production_root"
if [[ "$link_secret" != true ]]; then
  printf 'Flask secret linking skipped (use --link-secret to enable it).\n'
  printf 'Using a different Flask secret is harmless; you will simply need to log in again.\n'
fi
printf '\n'

for index in "${!selected_paths[@]}"; do
  relative_path="${selected_paths[$index]}"
  source_path="$production_root/$relative_path"
  destination_path="$worktree_root/$relative_path"
  relative_parent="${relative_path%/*}"
  ensure_safe_parent "$relative_parent"

  if [[ -L "$destination_path" ]]; then
    resolved_destination="$(readlink -f -- "$destination_path" 2>/dev/null || true)"
    resolved_source="$(readlink -f -- "$source_path")"
    if [[ "$resolved_destination" == "$resolved_source" ]]; then
      actions[$index]="already"
      already_count=$((already_count + 1))
      printf 'Already configured: %s -> %s\n' "$relative_path" "$source_path"
    else
      actions[$index]="relink"
      relink_count=$((relink_count + 1))
      printf 'Will replace incorrect symlink: %s -> %s (expected %s)\n' \
        "$relative_path" "$(readlink -- "$destination_path")" "$source_path"
    fi
  elif [[ -e "$destination_path" ]]; then
    actions[$index]="replace"
    force_required_count=$((force_required_count + 1))
    existing_kind="filesystem entry"
    [[ ! -d "$destination_path" ]] || existing_kind="directory"
    [[ ! -f "$destination_path" ]] || existing_kind="file"
    printf 'Will replace existing %s: %s with symlink to %s\n' \
      "$existing_kind" "$relative_path" "$source_path"
  else
    actions[$index]="create"
    create_count=$((create_count + 1))
    printf 'Will create symlink: %s -> %s\n' "$relative_path" "$source_path"
  fi
done

printf '\n'
if [[ "$dry_run" == true ]]; then
  printf 'Dry-run summary (no changes made):\n'
  printf '  Already configured: %d\n' "$already_count"
  printf '  Symlinks to create: %d\n' "$create_count"
  printf '  Incorrect symlinks to replace: %d\n' "$relink_count"
  printf '  Existing non-symlinks requiring --force: %d\n' "$force_required_count"
  exit 0
fi

if (( force_required_count > 0 )) && [[ "$force" != true ]]; then
  printf 'Refusing to replace %d existing non-symlink path(s) without --force.\n' \
    "$force_required_count" >&2
  printf 'No changes were made. Run with --dry-run to review or --force to proceed.\n' >&2
  exit 1
fi

changed_count=0
for index in "${!selected_paths[@]}"; do
  relative_path="${selected_paths[$index]}"
  source_path="$production_root/$relative_path"
  destination_path="$worktree_root/$relative_path"
  relative_parent="${relative_path%/*}"

  case "${actions[$index]}" in
    already)
      if [[ -n "${recorded_links[$relative_path]+present}" ]]; then
        recorded_links["$relative_path"]="$source_path"
      fi
      continue
      ;;
    create)
      mkdir -p -- "$worktree_root/$relative_parent"
      ;;
    relink)
      rm -f -- "$destination_path"
      ;;
    replace)
      if [[ -d "$destination_path" && ! -L "$destination_path" ]]; then
        rm -rf -- "$destination_path"
      else
        rm -f -- "$destination_path"
      fi
      ;;
  esac

  mkdir -p -- "$worktree_root/$relative_parent"
  ln -s -- "$source_path" "$destination_path"
  recorded_links["$relative_path"]="$source_path"
  write_state
  changed_count=$((changed_count + 1))
  printf 'Configured: %s -> %s\n' "$relative_path" "$source_path"
done
write_state

printf '\n'
if (( changed_count == 0 )); then
  printf 'Already configured: all %d selected Rider Portal preview links are correct.\n' "${#selected_paths[@]}"
else
  printf 'Configuration complete: %d link(s) updated; %d already configured.\n' \
    "$changed_count" "$already_count"
fi
