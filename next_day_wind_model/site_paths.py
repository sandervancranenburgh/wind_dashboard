"""Site-isolated runtime paths and checksummed model-artifact manifests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from next_day_wind_model.site_registry import SiteRegistry, load_site_registry


ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
DEFAULT_ARTIFACT_ROOT = Path("next_day_wind_model/artifacts")
DEFAULT_WEB_ROOT = Path("docs")


@dataclass(frozen=True)
class SitePaths:
    site_id: str
    artifact_dir: Path
    web_dir: Path | None


def publication_target(
    site_id: str,
    *,
    web_root: Path = DEFAULT_WEB_ROOT,
    registry: SiteRegistry | None = None,
) -> Path | None:
    """Return the only registry-authorized web destination for a site."""
    resolved_registry = registry or load_site_registry()
    web_relative = resolved_registry.web_output_relative_path(site_id)
    return None if web_relative is None else Path(web_root) / web_relative


def resolve_site_paths(
    site_id: str,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    web_root: Path = DEFAULT_WEB_ROOT,
    registry: SiteRegistry | None = None,
) -> SitePaths:
    resolved_registry = registry or load_site_registry()
    site = resolved_registry.site(site_id)
    return SitePaths(
        site_id=site.site_id,
        artifact_dir=Path(artifact_root) / site.site_id,
        web_dir=publication_target(site.site_id, web_root=web_root, registry=resolved_registry),
    )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file_entries(root: Path, files: Iterable[Path]) -> dict[str, dict[str, object]]:
    root_resolved = root.resolve()
    entries: dict[str, dict[str, object]] = {}
    for path in files:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"artifact file is outside artifact root: {path}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        key = relative.as_posix()
        entries[key] = {
            "sha256": sha256_file(resolved),
            "size_bytes": int(resolved.stat().st_size),
        }
    return dict(sorted(entries.items()))


def build_artifact_manifest(
    *,
    artifact_dir: Path,
    site_id: str,
    forecast_model: str,
    status: str,
    production_eligible: bool,
    files: Iterable[Path],
    training_data_start_utc: str | None = None,
    training_data_end_utc: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized_status = str(status).strip().lower()
    if normalized_status not in {"champion", "experimental", "unavailable"}:
        raise ValueError("artifact status must be champion, experimental, or unavailable")
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "site_id": str(site_id).strip(),
        "forecast_model": str(forecast_model).strip(),
        "status": normalized_status,
        "production_eligible": bool(production_eligible),
        "training_data_start_utc": training_data_start_utc,
        "training_data_end_utc": training_data_end_utc,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": _relative_file_entries(Path(artifact_dir), files),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def write_artifact_manifest(artifact_dir: Path, manifest: Mapping[str, object]) -> Path:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / ARTIFACT_MANIFEST_NAME
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_artifact_manifest(artifact_dir: Path) -> dict[str, object]:
    path = Path(artifact_dir) / ARTIFACT_MANIFEST_NAME
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("artifact manifest root must be an object")
    return value


def validate_artifact_manifest(
    artifact_dir: Path,
    *,
    expected_site_id: str,
    expected_forecast_model: str,
    verify_checksums: bool = True,
) -> dict[str, object]:
    artifact_dir = Path(artifact_dir)
    manifest = load_artifact_manifest(artifact_dir)
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported artifact manifest schema_version")
    if manifest.get("site_id") != expected_site_id:
        raise ValueError(
            f"artifact site mismatch: expected {expected_site_id!r}, got {manifest.get('site_id')!r}"
        )
    if manifest.get("forecast_model") != expected_forecast_model:
        raise ValueError(
            "artifact forecast-model mismatch: "
            f"expected {expected_forecast_model!r}, got {manifest.get('forecast_model')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("artifact manifest files must be an object")
    root = artifact_dir.resolve()
    for relative, metadata in files.items():
        path = (artifact_dir / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact manifest path escapes artifact directory: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_checksums:
            expected = metadata.get("sha256") if isinstance(metadata, dict) else None
            actual = sha256_file(path)
            if expected != actual:
                raise ValueError(f"artifact checksum mismatch: {relative}")
    return manifest


def validate_checkpoint_identity(
    checkpoint: Mapping[str, object],
    *,
    expected_site_id: str,
    expected_forecast_model: str,
    allow_legacy_identity: bool = False,
) -> None:
    schema_version = checkpoint.get("artifact_schema_version")
    site_id = checkpoint.get("site_id")
    forecast_model = checkpoint.get("forecast_model")
    identity_values = (schema_version, site_id, forecast_model)
    if any(value is None for value in identity_values):
        if allow_legacy_identity and all(value is None for value in identity_values):
            return
        raise ValueError("model checkpoint lacks artifact_schema_version, site_id, or forecast_model identity")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported model checkpoint artifact_schema_version: {schema_version!r}")
    if site_id != expected_site_id:
        raise ValueError(f"model checkpoint site mismatch: expected {expected_site_id!r}, got {site_id!r}")
    if forecast_model != expected_forecast_model:
        raise ValueError(
            "model checkpoint forecast-model mismatch: "
            f"expected {expected_forecast_model!r}, got {forecast_model!r}"
        )
