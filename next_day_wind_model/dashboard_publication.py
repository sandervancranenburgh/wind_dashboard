"""Publish a bounded public dashboard snapshot without committing runtime files.

Only explicitly named public website outputs are packaged. Rider data, daily
archives, databases and operations documents are never publication inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from next_day_wind_model.daily_archive import atomic_write
from next_day_wind_model.site_registry import load_site_registry

PUBLIC_FILES = frozenset("""
index.html evaluation.html dashboard_refresh.js site.webmanifest
metadata_update.json current_day_interactive_data.json next_day_interactive_data.json
day_after_tomorrow_interactive_data.json day_after_tomorrow_metadata.json
current_day_predictions.png current_day_predictions_mobile.png current_day_predictions.csv
next_day_predictions.png next_day_predictions_mobile.png next_day_predictions.csv
daily_mae_history.png daily_mae_history_mobile.png daily_mae_history.csv
model_gate_eval_history.png model_gate_eval_history.csv model_gate_direction_spider.png
model_gate_speed_by_direction.csv current_day_direction_spider.png current_day_speed_by_direction.csv
current_day_model_gate_eval_history.png current_day_model_gate_eval_history.csv
current_day_model_gate_eval_details.csv day_after_tomorrow_predictions.png
day_after_tomorrow_predictions_mobile.png day_after_tomorrow_predictions.csv
day_after_tomorrow_direction_spider.png day_after_tomorrow_model_gate_eval_details.csv
day_after_tomorrow_model_gate_eval_history.csv day_after_tomorrow_model_gate_eval_history.png
day_after_tomorrow_realized_scores.csv day_after_tomorrow_speed_by_direction.csv
site-assets/apple-touch-icon.png site-assets/favicon-16x16.png site-assets/favicon-32x32.png
site-assets/favicon.ico site-assets/icon-192x192.png site-assets/icon-512x512.png
""".split())
REQUIRED_FILES = frozenset({"index.html", "current_day_predictions.png", "current_day_predictions.csv",
                            "next_day_predictions.png", "next_day_predictions.csv", "metadata_update.json"})
MANIFEST = "publication-manifest.json"
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024


def default_bundle_path(repo_root: Path) -> Path:
    configured = os.environ.get("WIND_DASHBOARD_BUNDLE_PATH")
    return Path(configured) if configured else repo_root / "next_day_wind_model/publication/dashboard.zip"


def allowed_files() -> set[str]:
    registry = load_site_registry()
    result = set(PUBLIC_FILES)
    for site in registry.sites:
        relative = registry.web_output_relative_path(site.site_id)
        if relative is not None and relative != Path("."):
            result.update((relative / name).as_posix() for name in PUBLIC_FILES)
    return result


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if value and key in {"href", "src", "data-src"}:
                self.links.append(value)


def validate_files(files: dict[str, bytes]) -> None:
    missing = REQUIRED_FILES - files.keys()
    if missing:
        raise ValueError(f"dashboard snapshot lacks required files: {sorted(missing)}")
    if set(files) - allowed_files():
        raise ValueError("dashboard snapshot contains a non-public filename")
    for name, content in files.items():
        if not content:
            raise ValueError(f"dashboard snapshot contains an empty file: {name}")
        if not name.endswith(".html"):
            continue
        parser = _Links()
        parser.feed(content.decode("utf-8"))
        for value in parser.links:
            url = urlsplit(value)
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            # Do not allow a missing published image, download, script or page.
            path = unquote(url.path)
            key = (PurePosixPath(name).parent / path).as_posix()
            if key.startswith("./"):
                key = key[2:]
            if key not in files:
                raise ValueError(f"{name} references an unpublished local file: {path}")


def build_bundle(web_root: Path, destination: Path) -> dict:
    web_root = web_root.resolve(strict=True)
    if destination.resolve().is_relative_to(web_root):
        raise ValueError("publication bundle must be outside the public web directory")
    files: dict[str, bytes] = {}
    for name in sorted(allowed_files()):
        path = web_root / name
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(web_root) or not path.is_file():
            raise ValueError(f"public asset must be a regular in-tree file: {name}")
        files[name] = path.read_bytes()
    validate_files(files)
    if sum(map(len, files.values())) > MAX_EXPANDED_BYTES:
        raise ValueError("dashboard snapshot exceeds the expanded size limit")
    manifest = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                  for name, content in files.items()},
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr(MANIFEST, json.dumps(manifest, sort_keys=True))
    content = buffer.getvalue()
    if len(content) > MAX_BUNDLE_BYTES:
        raise ValueError("dashboard snapshot exceeds the compressed size limit")
    atomic_write(destination, content)
    return manifest


def extract_bundle(bundle: Path, destination: Path) -> None:
    """Validate the entire download before creating any deployable files."""
    if bundle.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("compressed publication bundle exceeds the size limit")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("publication destination must be empty")
    with zipfile.ZipFile(bundle) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError("duplicate filenames or missing publication manifest")
        if set(names) - allowed_files() - {MANIFEST}:
            raise ValueError("unrecognized public filename in downloaded bundle")
        if sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES:
            raise ValueError("expanded publication bundle exceeds the size limit")
        for entry in entries:
            if (entry.external_attr >> 16) & 0o170000 == 0o120000 or entry.is_dir():
                raise ValueError("publication bundle cannot contain links or directory entries")
        manifest = json.loads(archive.read(MANIFEST))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported publication manifest")
        files = {name: archive.read(name) for name in names if name != MANIFEST}
    if set(manifest["files"]) != set(files):
        raise ValueError("publication manifest and bundle filenames differ")
    for name, content in files.items():
        if manifest["files"][name] != {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}:
            raise ValueError(f"publication checksum mismatch: {name}")
    validate_files(files)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        atomic_write(destination / name, content)


def dispatch_publication(repo_root: Path, web_root: Path, remote: str, branch: str) -> dict:
    result = {"enabled": True, "pushed": False, "dispatched": False, "publication_mode": "artifact"}
    try:
        remote_url = subprocess.run(
            ["git", "-C", str(repo_root), "remote", "get-url", remote],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?", remote_url)
        if not match:
            raise ValueError("artifact publication requires a github.com repository remote")
        repository = match.group(1)
        token = os.environ.get("WIND_GITHUB_ACTIONS_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            raise ValueError("artifact publication requires an Actions-write token")
        build_bundle(web_root, default_bundle_path(repo_root))
        request = Request(
            f"https://api.github.com/repos/{repository}/actions/workflows/deploy-dashboard.yml/dispatches",
            data=json.dumps({"ref": branch}).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2026-03-10", "Content-Type": "application/json",
                     "User-Agent": "wind-dashboard-publication"},
        )
        run_details = {}
        with urlopen(request, timeout=60) as response:
            if response.status == 200:
                details = json.loads(response.read(1024 * 1024))
                run_details = {"workflow_run_id": details["workflow_run_id"],
                               "workflow_run_url": details["html_url"]}
            elif response.status != 204:
                raise ValueError("GitHub did not accept the workflow dispatch")
        return {**result, "dispatched": True, "reason": "artifact_dispatched", "repository": repository, **run_details}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        # Never print request headers or error bodies containing credentials.
        return {**result, "reason": "artifact_publication_failed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["build", "extract"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--web-root", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.operation == "build" and args.web_root:
        build_bundle(args.web_root, args.bundle)
    elif args.operation == "extract" and args.destination:
        extract_bundle(args.bundle, args.destination)
    else:
        parser.error("build requires --web-root; extract requires --destination")


if __name__ == "__main__":
    main()
