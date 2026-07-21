"""Reproducibly profile Rider Portal import and landing-request latency.

Anonymous profiling never opens the portal database. Authenticated profiling
can run the landing-page backfill, so --data-dir is deliberately restricted to
a temporary database copy under the platform temporary directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROFILE_SECRET = "rider-portal-local-profile-only"


def db_filename() -> str:
    return "wind_data_all_sites.db"


def _temporary_data_dir(value: str) -> Path:
    path = Path(value).resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if path != temporary_root and temporary_root not in path.parents:
        raise argparse.ArgumentTypeError(
            "--data-dir must be a copied database directory under the temporary directory"
        )
    if not (path / db_filename()).is_file():
        raise argparse.ArgumentTypeError(f"{path / db_filename()} does not exist")
    return path


def _install_timer(target: Any, name: str, timings: dict[str, float]) -> None:
    original = getattr(target, name)

    def timed(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started

    setattr(target, name, timed)


def _profile_child(data_dir: Path | None, user_id: int | None) -> dict[str, Any]:
    import_started = time.perf_counter()
    import db_store
    from next_day_wind_model.web_dashboard import app as portal

    import_elapsed = time.perf_counter() - import_started
    if data_dir is not None:
        portal.DATA_DIR = data_dir
    portal.app.config.update(TESTING=True, SECRET_KEY=PROFILE_SECRET)

    stage_timings: dict[str, float] = {}
    for target, name in (
        (portal, "_connect_db"),
        (db_store, "backfill_surf_experience_measured_summaries"),
        (db_store, "refresh_surf_experience_measured_wind"),
        (db_store, "get_measured_wind_for_session"),
        (db_store, "get_forecast_temperature_for_session"),
        (db_store, "list_surf_experiences"),
        (portal, "render_template"),
    ):
        _install_timer(target, name, stage_timings)

    client = portal.app.test_client()
    if user_id is not None:
        with client.session_transaction() as current_session:
            current_session["user_id"] = user_id

    requests = []
    for label in ("first", "second"):
        stage_timings.clear()
        started = time.perf_counter()
        response = client.get("/", follow_redirects=True)
        requests.append(
            {
                "label": label,
                "elapsed_s": time.perf_counter() - started,
                "status": response.status_code,
                "bytes": len(response.data),
                "stages_s": dict(stage_timings),
            }
        )
    return {"app_import_s": import_elapsed, "requests": requests}


def _run_parent(args: argparse.Namespace) -> int:
    baseline_started = time.perf_counter()
    subprocess.run([sys.executable, "-c", "pass"], check=True)
    baseline_elapsed = time.perf_counter() - baseline_started

    command = [
        sys.executable,
        "-m",
        "next_day_wind_model.web_dashboard.profile_rider_portal",
        "--child",
    ]
    if args.data_dir is not None:
        command.extend(("--data-dir", str(args.data_dir), "--user-id", str(args.user_id)))
    environment = os.environ.copy()
    environment["WIND_DASHBOARD_SECRET_KEY"] = PROFILE_SECRET
    child_started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    child_elapsed = time.perf_counter() - child_started
    result = json.loads(completed.stdout)
    result.update(
        {
            "python_process_s": baseline_elapsed,
            "profile_process_s": child_elapsed,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=_temporary_data_dir,
        help="temporary directory containing a copied portal database",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        help="user id for authenticated landing-page profiling",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.data_dir is None) != (args.user_id is None):
        raise SystemExit("--data-dir and --user-id must be supplied together")
    if args.child:
        print(json.dumps(_profile_child(args.data_dir, args.user_id), sort_keys=True))
        return 0
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
