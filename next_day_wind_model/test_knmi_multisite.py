from __future__ import annotations

import io
import sqlite3
import tarfile
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from next_day_wind_model.knmi_harmonie import (
    ExtractionResult,
    RawTarCleanupResult,
    SitePoint,
    extract_tar_features_for_sites,
)
from scripts.knmi_extract_latest_to_db import (
    DEFAULT_SITE_POINTS,
    parse_args,
    process_knmi_file_to_db_for_sites,
    site_points_from_args,
)


class KnmiMultiSiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tar_path = self.root / "HARM43_V1_P1_2026092910.tar"
        with tarfile.open(self.tar_path, "w") as archive:
            content = b"fake-grib"
            info = tarfile.TarInfo("HARM43_V1_P1_202609291000_00000_GB")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_cli_requires_explicit_selection_and_supports_repeated_sites(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([])
        parsed = parse_args([
            "--site", "oostvoorne", "--site", "valkenburgsemeer",
            "--site", "oostvoorne",
        ])
        args = Namespace(
            all_sites=parsed.all_sites, sites=parsed.sites,
            site_lat=parsed.site_lat, site_lon=parsed.site_lon,
        )
        self.assertEqual(
            tuple(point.site for point in site_points_from_args(args)),
            ("oostvoorne", "valkenburgsemeer"),
        )
        all_args = Namespace(all_sites=True, sites=None, site_lat=None, site_lon=None)
        self.assertEqual(
            tuple(point.site for point in site_points_from_args(all_args)),
            tuple(DEFAULT_SITE_POINTS),
        )

    def test_custom_coordinates_require_one_site(self) -> None:
        args = Namespace(
            all_sites=True, sites=None, site_lat=51.9, site_lon=4.1
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            site_points_from_args(args)

    def test_tar_member_is_read_once_for_both_sites(self) -> None:
        sites = (
            SitePoint("valkenburgsemeer", 52.168, 4.437),
            SitePoint("oostvoorne", 51.928597, 4.074097),
        )
        calls: list[str] = []

        def fake_extract(path: Path, site: SitePoint, **kwargs: object) -> dict[str, object]:
            calls.append(site.site)
            return self._row(site.site, site.lat, site.lon, 0)

        with patch("next_day_wind_model.knmi_harmonie.extract_one_grib", side_effect=fake_extract):
            results = extract_tar_features_for_sites(self.tar_path, sites)
        self.assertEqual(calls, ["valkenburgsemeer", "oostvoorne"])
        self.assertEqual(set(results), {"valkenburgsemeer", "oostvoorne"})

    def test_multisite_write_is_idempotent_and_does_not_touch_forecasts(self) -> None:
        db_path = self.root / "test.db"
        cleanup = RawTarCleanupResult((), (), (), ())
        extracted = self._extracted()
        with patch("scripts.knmi_extract_latest_to_db.select_tar", return_value=(self.tar_path, self.tar_path.name, "2026-09-29T10:00:00+00:00", 9)), patch(
            "scripts.knmi_extract_latest_to_db.extract_tar_features_for_sites", return_value=extracted
        ), patch("scripts.knmi_extract_latest_to_db.cleanup_raw_harmonie_tars", return_value=cleanup):
            for _ in range(2):
                process_knmi_file_to_db_for_sites(
                    filename=self.tar_path.name,
                    db_path=db_path,
                    sites=("valkenburgsemeer", "oostvoorne"),
                    skip_archive_diagnostic=True,
                )
        conn = sqlite3.connect(db_path)
        try:
            feature_counts = dict(conn.execute(
                "SELECT site, COUNT(*) FROM harmonie_knmi_features GROUP BY site"
            ))
            shadow_counts = dict(conn.execute(
                "SELECT site, COUNT(*) FROM knmi_forecasts_shadow GROUP BY site"
            ))
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()
        self.assertEqual(feature_counts, {"oostvoorne": 2, "valkenburgsemeer": 2})
        self.assertEqual(shadow_counts, feature_counts)
        self.assertNotIn("forecasts", tables)

    def test_second_site_failure_rolls_back_both_and_retains_tar(self) -> None:
        db_path = self.root / "rollback.db"
        cleanup = RawTarCleanupResult((), (), (), ())
        extracted = self._extracted()
        from scripts import knmi_extract_latest_to_db as worker
        real_upsert = worker.upsert_harmonie_knmi_features

        def fail_second(conn: sqlite3.Connection, frame: pd.DataFrame, **kwargs: object) -> int:
            if frame.iloc[0]["site"] == "oostvoorne":
                raise RuntimeError("simulated second-site failure")
            return real_upsert(conn, frame, **kwargs)

        with patch("scripts.knmi_extract_latest_to_db.select_tar", return_value=(self.tar_path, self.tar_path.name, "2026-09-29T10:00:00+00:00", 9)), patch(
            "scripts.knmi_extract_latest_to_db.extract_tar_features_for_sites", return_value=extracted
        ), patch("scripts.knmi_extract_latest_to_db.upsert_harmonie_knmi_features", side_effect=fail_second), patch(
            "scripts.knmi_extract_latest_to_db.cleanup_raw_harmonie_tars", return_value=cleanup
        ) as cleanup_mock:
            with self.assertRaisesRegex(RuntimeError, "second-site"):
                process_knmi_file_to_db_for_sites(
                    filename=self.tar_path.name,
                    db_path=db_path,
                    sites=("valkenburgsemeer", "oostvoorne"),
                    skip_archive_diagnostic=True,
                )
        cleanup_mock.assert_not_called()
        self.assertTrue(self.tar_path.exists())
        conn = sqlite3.connect(db_path)
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='harmonie_knmi_features'"
            ).fetchone()
            count = 0 if table is None else conn.execute("SELECT COUNT(*) FROM harmonie_knmi_features").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 0)

    def _extracted(self) -> dict[str, ExtractionResult]:
        return {
            site: ExtractionResult(
                pd.DataFrame([
                    self._row(site, point.lat, point.lon, horizon)
                    for horizon in (0, 1)
                ]),
                (),
            )
            for site, point in DEFAULT_SITE_POINTS.items()
        }

    @staticmethod
    def _row(site: str, lat: float, lon: float, horizon: int) -> dict[str, object]:
        target_hour = 10 + horizon
        return {
            "source": "knmi_harmonie_p1",
            "dataset": "harmonie_arome_cy43_p1",
            "run_ts": "2026-09-29T10:00:00+00:00",
            "fetched_ts": "2026-09-29T10:05:00Z",
            "target_ts": f"2026-09-29T{target_hour:02d}:00:00+00:00",
            "horizon_hr": horizon,
            "site": site,
            "site_lat": lat,
            "site_lon": lon,
            "grid_lat": lat,
            "grid_lon": lon,
            "wind_speed_10m_mps": 6.0,
            "wind_speed_10m_knots": 11.663,
            "wind_gust_10m_mps": 8.0,
            "wind_gust_10m_knots": 15.551,
            "wind_dir_10m": 250.0,
        }


if __name__ == "__main__":
    unittest.main()
