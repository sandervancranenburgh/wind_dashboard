"""Run the superlocal pipeline for registry-enabled sites with isolated paths.

Dry-run is the default. This module is intentionally not wired into any wrapper,
service, timer, or cron job during Phase 3A.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.site_paths import DEFAULT_ARTIFACT_ROOT, DEFAULT_WEB_ROOT, resolve_site_paths
from next_day_wind_model.site_registry import load_site_registry


UPDATER = REPO_ROOT / "next_day_wind_model" / "update_model_and_predict.py"


def enabled_model_site_ids() -> tuple[str, ...]:
    return tuple(site.site_id for site in load_site_registry().enabled_sites("superlocal_model"))


def build_site_command(
    site_id: str,
    *,
    db_path: Path,
    artifact_root: Path,
    web_root: Path,
    extra_args: list[str] | None = None,
) -> list[str]:
    registry = load_site_registry()
    site = registry.site(site_id)
    if not site.superlocal_model.enabled:
        raise ValueError(f"Site {site_id!r} is disabled for the superlocal model")
    paths = resolve_site_paths(
        site.site_id,
        artifact_root=artifact_root,
        web_root=web_root,
        registry=registry,
    )
    command = [
        sys.executable,
        str(UPDATER),
        "--db",
        str(db_path),
        "--site",
        site.site_id,
        "--model",
        "HARMONIE",
        "--out-dir",
        str(paths.artifact_dir),
        "--model-artifact-dir",
        str(paths.artifact_dir),
    ]
    if paths.web_dir is not None:
        command.extend(["--web-out-dir", str(paths.web_dir)])
    command.extend(extra_args or [])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--site", action="append", dest="sites", help="Enabled site to process; repeatable")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--web-root", type=Path, default=DEFAULT_WEB_ROOT)
    parser.add_argument("--execute", action="store_true", help="Execute commands; otherwise print the plan")
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the updater")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    enabled = enabled_model_site_ids()
    selected = tuple(args.sites or enabled)
    disabled = sorted(set(selected) - set(enabled))
    if disabled:
        raise SystemExit(f"Refusing disabled or unknown model sites: {', '.join(disabled)}")
    extra_args = list(args.pipeline_args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    for site_id in selected:
        command = build_site_command(
            site_id,
            db_path=args.db,
            artifact_root=args.artifact_root,
            web_root=args.web_root,
            extra_args=extra_args,
        )
        print(shlex.join(command))
        if args.execute:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
