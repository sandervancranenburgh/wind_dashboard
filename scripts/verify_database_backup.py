#!/usr/bin/env python3
"""Restore and validate a compressed wind database backup away from production."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PRODUCTION_DB = Path.home() / "Documents/wind_fetcher_data/data/wind_data_all_sites.db"
REQUIRED_TABLES = {
    "forecasts",
    "harmonie_knmi_features",
    "knmi_forecasts_shadow",
    "observations",
    "prediction_log",
    "surf_experiences",
    "user_profiles",
    "users",
}
MINIMUM_FREE_SPACE_MULTIPLIER = 12


@dataclass(frozen=True)
class VerificationResult:
    report: dict[str, Any]
    restored_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive(path: Path, production_db: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError("backup archive must be a regular, non-symlink file")
    archive = candidate.resolve(strict=True)
    production = production_db.expanduser().resolve(strict=False)
    if archive == production:
        raise ValueError("backup archive must not be the production database")
    if not archive.is_file():
        raise ValueError("backup archive must be a regular, non-symlink file")
    if not archive.name.endswith(".db.gz"):
        raise ValueError("backup archive name must end with .db.gz")
    return archive


def _preflight_space(archive: Path, destination: Path) -> dict[str, int]:
    free_bytes = int(shutil.disk_usage(destination).free)
    conservative_required = int(archive.stat().st_size * MINIMUM_FREE_SPACE_MULTIPLIER)
    if free_bytes < conservative_required:
        raise OSError(
            "insufficient free space for isolated restore: "
            f"available={free_bytes} required={conservative_required}"
        )
    return {
        "available_bytes_before_restore": free_bytes,
        "conservative_required_bytes": conservative_required,
    }


def _restore(archive: Path, destination: Path) -> Path:
    restored = destination / archive.name.removesuffix(".gz")
    with gzip.open(archive, "rb") as source, restored.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    return restored


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _group_counts(conn: sqlite3.Connection, table: str, group_column: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        f'SELECT "{group_column}", COUNT(*) FROM "{table}" GROUP BY "{group_column}" '
        f'ORDER BY "{group_column}"'
    ).fetchall()
    return [{group_column: row[0], "rows": int(row[1])} for row in rows]


def _timestamp_bounds(conn: sqlite3.Connection, table: str, candidates: Iterable[str]) -> dict[str, Any]:
    columns = _table_columns(conn, table)
    timestamp_column = next((candidate for candidate in candidates if candidate in columns), None)
    if timestamp_column is None:
        return {}
    minimum, maximum = conn.execute(
        f'SELECT MIN("{timestamp_column}"), MAX("{timestamp_column}") FROM "{table}"'
    ).fetchone()
    return {
        "timestamp_column": timestamp_column,
        "minimum": minimum,
        "maximum": maximum,
    }


def _database_report(restored: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{restored.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        integrity_rows = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
        if integrity_rows != ["ok"]:
            raise sqlite3.DatabaseError(f"integrity_check failed: {integrity_rows[:10]}")

        objects = conn.execute(
            "SELECT type, name, COALESCE(sql, '') FROM sqlite_master "
            "WHERE type IN ('table', 'index') ORDER BY type, name"
        ).fetchall()
        tables = {str(row[1]) for row in objects if row[0] == "table"}
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise sqlite3.DatabaseError(f"required tables are missing: {', '.join(missing)}")
        schema_text = "\n".join(f"{row[0]}|{row[1]}|{row[2]}" for row in objects)

        table_report: dict[str, Any] = {}
        for table in sorted(REQUIRED_TABLES):
            row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            columns = _table_columns(conn, table)
            item: dict[str, Any] = {"rows": row_count}
            if "site" in columns:
                item["by_site"] = _group_counts(conn, table, "site")
            elif "spot" in columns:
                item["by_spot"] = _group_counts(conn, table, "spot")
            item.update(
                _timestamp_bounds(
                    conn,
                    table,
                    (
                        "iso_time",
                        "run_ts",
                        "issued_iso",
                        "date",
                        "created_ts",
                    ),
                )
            )
            if "payload" in columns:
                invalid_json = int(
                    conn.execute(
                        f'SELECT COUNT(*) FROM "{table}" '
                        "WHERE payload IS NOT NULL AND json_valid(payload)=0"
                    ).fetchone()[0]
                )
                if invalid_json:
                    raise sqlite3.DatabaseError(f"{table} contains {invalid_json} invalid JSON payloads")
                item["invalid_json_payloads"] = 0
            table_report[table] = item

        return {
            "integrity_check": "ok",
            "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
            "table_count": len(tables),
            "index_count": sum(1 for row in objects if row[0] == "index"),
            "tables": table_report,
        }
    finally:
        conn.close()


def verify_backup(
    archive_path: Path,
    *,
    work_parent: Path,
    production_db: Path = DEFAULT_PRODUCTION_DB,
) -> VerificationResult:
    archive = _safe_archive(archive_path, production_db)
    work_root = work_parent.expanduser().resolve()
    if not work_root.is_dir():
        raise ValueError(f"restore work directory does not exist: {work_root}")
    space = _preflight_space(archive, work_root)
    restore_dir = Path(tempfile.mkdtemp(prefix="wind-db-restore-", dir=work_root))
    try:
        restored = _restore(archive, restore_dir)
        database = _database_report(restored)
        report = {
            "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "archive": str(archive),
            "archive_size_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "restored_size_bytes": int(restored.stat().st_size),
            "work_parent": str(work_root),
            "space": space,
            "database": database,
        }
        return VerificationResult(report=report, restored_path=restored)
    except Exception:
        shutil.rmtree(restore_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore a .db.gz backup to isolated temporary storage and validate it read-only."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--work-parent", type=Path, default=Path("/tmp"))
    parser.add_argument("--production-db", type=Path, default=DEFAULT_PRODUCTION_DB)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result: VerificationResult | None = None
    try:
        result = verify_backup(
            args.archive,
            work_parent=args.work_parent,
            production_db=args.production_db,
        )
        text = json.dumps(result.report, indent=2, sort_keys=True) + "\n"
        if args.report is not None:
            report_path = args.report.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0
    finally:
        if result is not None:
            shutil.rmtree(result.restored_path.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
