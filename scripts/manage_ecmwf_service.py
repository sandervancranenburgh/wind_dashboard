#!/usr/bin/env python3
"""Render and manage the production user-level ECMWF collector timer."""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "main"
SERVICE_NAME = "wind-fetcher2-ecmwf.service"
TIMER_NAME = "wind-fetcher2-ecmwf.timer"
DEVELOPMENT_TIMER_NAME = "wind-fetcher2-ecmwf-shadow-dev.timer"
TEMPLATE_DIR = REPO_ROOT / "ops" / "systemd"
DEFAULT_PYTHON = REPO_ROOT / ".venv-ecmwf" / "bin" / "python"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "ecmwf_archive"
DEFAULT_CONFIG = REPO_ROOT / "config" / "ecmwf_production.json"


class ServiceSafetyError(RuntimeError):
    pass


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_primary_checkout() -> None:
    branch = _git_output("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise ServiceSafetyError(
            f"refusing branch {branch!r}; expected production branch {EXPECTED_BRANCH!r}"
        )
    if not (REPO_ROOT / ".git").is_dir():
        raise ServiceSafetyError(
            "refusing a linked worktree; the primary production checkout is required"
        )


def validate_runtime_paths(
    python_path: Path,
    data_dir: Path,
    config_path: Path,
) -> tuple[Path, Path, Path]:
    # Preserve the venv interpreter path instead of resolving its symlink.
    python_path = python_path.expanduser().absolute()
    data_dir = data_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    environment_root = (REPO_ROOT / ".venv-ecmwf").resolve()
    expected_data = DEFAULT_DATA_DIR.resolve()
    expected_config = DEFAULT_CONFIG.resolve()
    if not python_path.is_file() or not python_path.is_relative_to(environment_root):
        raise ServiceSafetyError(
            f"production Python must exist inside {environment_root}: {python_path}"
        )
    if data_dir != expected_data:
        raise ServiceSafetyError(f"production archive must be exactly {expected_data}")
    if not config_path.is_file() or config_path != expected_config:
        raise ServiceSafetyError(f"production config must be exactly {expected_config}")
    for path in (REPO_ROOT.resolve(), python_path, data_dir, config_path):
        if any(character.isspace() for character in str(path)):
            raise ServiceSafetyError("systemd runtime paths must not contain whitespace")
    return python_path, data_dir, config_path


def render_units(
    python_path: Path = DEFAULT_PYTHON,
    data_dir: Path = DEFAULT_DATA_DIR,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, str]:
    replacements = {
        "@REPO_ROOT@": str(REPO_ROOT.resolve()),
        "@PYTHON@": str(python_path),
        "@DATA_DIR@": str(data_dir),
        "@CONFIG@": str(config_path.expanduser().resolve()),
    }
    service = (TEMPLATE_DIR / f"{SERVICE_NAME}.in").read_text(encoding="utf-8")
    for source, target in replacements.items():
        service = service.replace(source, target)
    if "@" in service:
        raise ServiceSafetyError("unresolved placeholder in service template")
    timer = (TEMPLATE_DIR / TIMER_NAME).read_text(encoding="utf-8")
    return {SERVICE_NAME: service, TIMER_NAME: timer}


def write_units(destination: Path, units: dict[str, str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in units.items():
        target = destination / name
        partial = destination / f".{name}.{os.getpid()}.part"
        partial.write_text(content, encoding="utf-8")
        partial.chmod(0o600)
        partial.replace(target)


def systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("systemctl", "--user", *args),
        check=check,
        capture_output=True,
        text=True,
    )


def validate_no_development_collector() -> None:
    enabled = systemctl("is-enabled", DEVELOPMENT_TIMER_NAME, check=False)
    active = systemctl("is-active", DEVELOPMENT_TIMER_NAME, check=False)
    if enabled.stdout.strip() == "enabled" or active.stdout.strip() == "active":
        raise ServiceSafetyError(
            f"disable and remove {DEVELOPMENT_TIMER_NAME} before production installation"
        )


def validate_linger_enabled() -> None:
    result = subprocess.run(
        ("loginctl", "show-user", getpass.getuser(), "--property=Linger", "--value"),
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip().lower() != "yes":
        raise ServiceSafetyError(
            "Linger is not enabled; run `sudo loginctl enable-linger "
            f"{getpass.getuser()}` before installing the production timer"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage only the uniquely named production ECMWF user timer."
    )
    parser.add_argument(
        "action",
        choices=(
            "render", "install", "start", "stop", "restart", "disable",
            "status", "logs", "uninstall",
        ),
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lines", type=int, default=100)
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=REPO_ROOT / "data" / "ecmwf_archive" / "systemd-rendered",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_primary_checkout()
        python_path, data_dir, config_path = validate_runtime_paths(
            args.python, args.data_dir, args.config
        )
        units = render_units(python_path, data_dir, config_path)
        if args.action == "render":
            destination = args.render_dir.expanduser().resolve()
            if not destination.is_relative_to(data_dir):
                raise ServiceSafetyError(
                    f"render directory must remain inside {data_dir}"
                )
            write_units(destination, units)
            print(f"Rendered units in {destination}")
            return 0

        user_unit_dir = Path.home() / ".config" / "systemd" / "user"
        if args.action in {"install", "start", "restart"}:
            validate_no_development_collector()
            validate_linger_enabled()
        if args.action == "install":
            data_dir.mkdir(parents=True, exist_ok=True)
            write_units(user_unit_dir, units)
            systemctl("daemon-reload")
            systemctl("enable", "--now", TIMER_NAME)
            systemctl("start", SERVICE_NAME)
            print(f"Installed and started {TIMER_NAME}")
            return 0
        if args.action == "start":
            systemctl("enable", "--now", TIMER_NAME)
            systemctl("start", SERVICE_NAME)
            return 0
        if args.action == "stop":
            systemctl("stop", TIMER_NAME, SERVICE_NAME)
            return 0
        if args.action == "restart":
            systemctl("restart", TIMER_NAME)
            systemctl("start", SERVICE_NAME)
            return 0
        if args.action == "disable":
            systemctl("disable", "--now", TIMER_NAME)
            systemctl("stop", SERVICE_NAME, check=False)
            return 0
        if args.action == "status":
            timer_status = systemctl("status", TIMER_NAME, "--no-pager", check=False)
            print(timer_status.stdout, end="")
            service_status = systemctl(
                "show", SERVICE_NAME,
                "--property=ActiveState,SubState,Result,ExecMainStatus",
                "--no-pager", check=False,
            )
            print(service_status.stdout, end="")
            return timer_status.returncode
        if args.action == "logs":
            return subprocess.run(
                (
                    "journalctl", "--user", "--unit", SERVICE_NAME,
                    "--lines", str(max(1, args.lines)), "--no-pager",
                ),
                check=False,
            ).returncode
        if args.action == "uninstall":
            systemctl("disable", "--now", TIMER_NAME, check=False)
            systemctl("stop", SERVICE_NAME, check=False)
            for name in (SERVICE_NAME, TIMER_NAME):
                (user_unit_dir / name).unlink(missing_ok=True)
            systemctl("daemon-reload")
            systemctl("reset-failed", SERVICE_NAME, check=False)
            print(f"Removed {SERVICE_NAME} and {TIMER_NAME}")
            return 0
        raise AssertionError(args.action)
    except (OSError, ServiceSafetyError, subprocess.CalledProcessError) as exc:
        print(f"Service action failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
