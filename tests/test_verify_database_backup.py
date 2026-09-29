from __future__ import annotations

import gzip
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.verify_database_backup import REQUIRED_TABLES, verify_backup


class BackupVerificationTests(unittest.TestCase):
    def _archive(self, root: Path) -> Path:
        database = root / "fixture.db"
        conn = sqlite3.connect(database)
        try:
            for table in sorted(REQUIRED_TABLES):
                if table == "observations":
                    conn.execute(
                        "CREATE TABLE observations(site TEXT, iso_time TEXT, payload TEXT)"
                    )
                    conn.execute(
                        "INSERT INTO observations VALUES ('oostvoorne', '2026-01-01T00:00:00Z', json('{}'))"
                    )
                elif table == "forecasts":
                    conn.execute(
                        "CREATE TABLE forecasts(site TEXT, run_ts TEXT, payload TEXT)"
                    )
                    conn.execute(
                        "INSERT INTO forecasts VALUES ('valkenburgsemeer', '2026-01-01T00:00:00Z', json('{}'))"
                    )
                elif table == "surf_experiences":
                    conn.execute("CREATE TABLE surf_experiences(spot TEXT, date TEXT)")
                else:
                    conn.execute(f'CREATE TABLE "{table}"(site TEXT, run_ts TEXT)')
            conn.execute("CREATE INDEX idx_observations_site ON observations(site)")
            conn.commit()
        finally:
            conn.close()
        archive = root / "fixture.db.gz"
        with database.open("rb") as source, gzip.open(archive, "wb") as target:
            target.write(source.read())
        return archive

    def test_valid_backup_is_restored_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            result = verify_backup(
                archive,
                work_parent=root,
                production_db=root / "production.db",
            )
            try:
                self.assertEqual(result.report["database"]["integrity_check"], "ok")
                self.assertEqual(
                    result.report["database"]["tables"]["observations"]["by_site"],
                    [{"site": "oostvoorne", "rows": 1}],
                )
                self.assertTrue(result.restored_path.is_file())
            finally:
                import shutil

                shutil.rmtree(result.restored_path.parent)

    def test_production_database_cannot_be_used_as_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            with self.assertRaisesRegex(ValueError, "production"):
                verify_backup(archive, work_parent=root, production_db=archive)

    def test_corrupt_gzip_fails_without_touching_production(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "broken.db.gz"
            archive.write_bytes(b"not a gzip archive")
            with self.assertRaises((gzip.BadGzipFile, EOFError)):
                verify_backup(
                    archive,
                    work_parent=root,
                    production_db=root / "production.db",
                )


if __name__ == "__main__":
    unittest.main()
