from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from db_store import init_db, upsert_forecasts
from next_day_wind_model.knmi_harmonie import (
    WEATHER_PARAMETERS,
    add_hourly_weather_features,
    create_harmonie_knmi_features_table,
    upsert_harmonie_knmi_features,
    extract_weather_parameters,
    SitePoint,
)
from next_day_wind_model.weather_conditions import (
    WEATHER_DESCRIPTIONS,
    WMO_CATEGORY,
    build_weather_timeline,
    conventional_degree_label,
    derive_weather_code,
    hourly_accumulation_increments,
    is_daylight,
    weather_icon_path,
)
from scripts.backfill_harmonie_weather import guarded_development_db


class WeatherClassifierTests(unittest.TestCase):
    def test_cloud_boundaries(self) -> None:
        self.assertEqual(derive_weather_code(cloud_cover_pct=19.999), 0)
        self.assertEqual(derive_weather_code(cloud_cover_pct=20), 1)
        self.assertEqual(derive_weather_code(cloud_cover_pct=49.999), 1)
        self.assertEqual(derive_weather_code(cloud_cover_pct=50), 2)
        self.assertEqual(derive_weather_code(cloud_cover_pct=79.999), 2)
        self.assertEqual(derive_weather_code(cloud_cover_pct=80), 3)

    def test_precipitation_boundaries(self) -> None:
        cases = [
            (0.01, 51), (0.499, 51), (0.5, 53), (0.999, 53),
            (1.0, 55), (1.299, 55), (1.3, 61), (2.5, 63), (7.6, 65),
        ]
        for amount, expected in cases:
            with self.subTest(amount=amount):
                self.assertEqual(
                    derive_weather_code(precipitation_mm=amount, cloud_cover_pct=0), expected
                )

    def test_snow_fog_and_precedence(self) -> None:
        self.assertEqual(derive_weather_code(snowfall_cm=0.01), 71)
        self.assertEqual(derive_weather_code(snowfall_cm=0.2), 73)
        self.assertEqual(derive_weather_code(snowfall_cm=0.8), 75)
        self.assertEqual(derive_weather_code(snow_water_equivalent_mm=1.0), 73)
        self.assertEqual(derive_weather_code(visibility_m=1000, cloud_cover_pct=0), 45)
        self.assertEqual(
            derive_weather_code(precipitation_mm=0.2, visibility_m=100, cloud_cover_pct=100), 51
        )
        self.assertEqual(
            derive_weather_code(precipitation_mm=9, snowfall_cm=0.3, thunderstorm=True), 95
        )

    def test_explicit_optional_branches(self) -> None:
        self.assertEqual(
            derive_weather_code(precipitation_mm=2, freezing_precipitation=True), 66
        )
        self.assertEqual(
            derive_weather_code(precipitation_mm=2, convective_precipitation=True), 81
        )
        self.assertEqual(
            derive_weather_code(snowfall_cm=1, convective_precipitation=True, snow_showers=True), 86
        )
        self.assertEqual(derive_weather_code(thunderstorm=True, heavy_hail=True), 99)
        self.assertIsNone(derive_weather_code())

    def test_accumulation_resets(self) -> None:
        actual = hourly_accumulation_increments([0.0, 0.5, 0.495, 0.2, np.nan, 0.3])
        np.testing.assert_allclose(actual[[0, 1, 2]], [0.0, 0.5, 0.0])
        self.assertTrue(np.isnan(actual[3]))
        self.assertTrue(np.isnan(actual[4]))
        self.assertTrue(np.isnan(actual[5]))

    def test_conventional_rounding(self) -> None:
        self.assertEqual(conventional_degree_label(12.5), "13°")
        self.assertEqual(conventional_degree_label(-2.5), "-3°")
        self.assertEqual(conventional_degree_label(None), "—")


class WeatherSolarAndIconTests(unittest.TestCase):
    def test_solar_day_night_and_dst_instants(self) -> None:
        self.assertTrue(is_daylight("2026-06-21T12:00:00Z", 52.168, 4.437))
        self.assertFalse(is_daylight("2026-06-21T00:00:00Z", 52.168, 4.437))
        # The calculation consumes instants, so equivalent DST representations agree.
        self.assertEqual(
            is_daylight("2026-03-29T18:30:00+02:00", 52.168, 4.437),
            is_daylight("2026-03-29T16:30:00Z", 52.168, 4.437),
        )

    def test_icon_map_is_complete_and_assets_are_rgba(self) -> None:
        self.assertTrue(set(WEATHER_DESCRIPTIONS).issubset(WMO_CATEGORY))
        checked: set[Path] = set()
        for code in WEATHER_DESCRIPTIONS:
            for daylight in (False, True):
                path = weather_icon_path(code, daylight)
                self.assertIsNotNone(path)
                self.assertTrue(path.exists())
                checked.add(path)
        self.assertEqual(len(checked), 13)
        for path in checked:
            with Image.open(path) as image:
                self.assertEqual(image.size, (128, 128))
                self.assertEqual(image.mode, "RGBA")


class WeatherStorageTests(unittest.TestCase):
    def test_backfill_refuses_production_paths(self) -> None:
        with self.assertRaises(ValueError):
            guarded_development_db("/tmp/wind_data_all_sites.db")
        with self.assertRaises(ValueError):
            guarded_development_db("/var/tmp/weather_icons.sqlite")
        self.assertEqual(
            guarded_development_db("/tmp/weather_icons.sqlite"),
            Path("/tmp/weather_icons.sqlite"),
        )

    def test_forecast_migration_is_additive_and_upsert_derives_code(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE forecasts (
                site TEXT NOT NULL, model TEXT NOT NULL, run_ts INTEGER NOT NULL,
                run_iso TEXT, fetched_ts INTEGER NOT NULL, fetched_iso TEXT,
                target_ts INTEGER NOT NULL, target_iso TEXT, horizon_hr INTEGER,
                wind_speed REAL, wind_gust REAL, wind_dir REAL, payload TEXT,
                PRIMARY KEY (site, model, run_ts, target_ts)
            )
            """
        )
        init_db(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(forecasts)")}
        self.assertIn("weather_code", columns)
        legacy = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'forecasts_legacy%'"
        ).fetchall()
        self.assertEqual(legacy, [])
        upsert_forecasts(
            conn, "valkenburgsemeer", "HARMONIE",
            [{"timestamp": 3_600_000, "WindForecastAvr": 5, "Clouds": 90, "Rain": 0}],
            0, 1,
        )
        self.assertEqual(conn.execute("SELECT weather_code FROM forecasts").fetchone()[0], 3)
        conn.close()

    def test_knmi_migration_and_hourly_derivation(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """CREATE TABLE harmonie_knmi_features (
            source TEXT NOT NULL, dataset TEXT NOT NULL, run_ts TEXT NOT NULL,
            fetched_ts TEXT NOT NULL, target_ts TEXT NOT NULL, horizon_hr INTEGER NOT NULL,
            site TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(source, dataset, run_ts, target_ts, site))"""
        )
        create_harmonie_knmi_features_table(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(harmonie_knmi_features)")}
        self.assertIn("temperature_2m_c", columns)
        self.assertIn("weather_code", columns)
        self.assertEqual(set(WEATHER_PARAMETERS), {
            "temperature_2m_c", "visibility_m", "relative_humidity_2m_pct",
            "total_precip_accum_mm", "total_cloud_cover_pct", "low_cloud_cover_pct",
            "medium_cloud_cover_pct", "high_cloud_cover_pct", "rain_accum_mm",
            "snow_accum_mm_we", "cloud_base_m", "graupel_accum_mm_we",
        })
        frame = pd.DataFrame(
            {
                "run_ts": ["2026-01-01T00:00:00Z"] * 3,
                "horizon_hr": [0, 1, 2],
                "total_precip_accum_mm": [0.0, 0.6, 1.8],
                "rain_accum_mm": [0.0, 0.6, 1.8],
                "snow_accum_mm_we": [0.0, 0.0, 0.0],
                "graupel_accum_mm_we": [0.0, 0.0, 0.0],
                "total_cloud_cover_pct": [10, 90, 90],
                "visibility_m": [10_000, 10_000, 10_000],
            }
        )
        derived = add_hourly_weather_features(frame)
        self.assertEqual(derived["weather_code"].tolist(), [0, 53, 55])
        conn.close()

    def test_p1_parameter_extraction_requests_accumulated_fields(self) -> None:
        class Dataset:
            def close(self) -> None:
                pass

        calls: list[tuple[int, int, int | None]] = []

        def fake_open(_path, parameter, level, *, time_range_indicator=None):
            calls.append((parameter, level, time_range_indicator))
            return Dataset()

        with mock.patch(
            "next_day_wind_model.knmi_harmonie.open_grib_parameter", side_effect=fake_open
        ), mock.patch(
            "next_day_wind_model.knmi_harmonie.nearest_value", return_value=(1.0, 52.0, 4.0)
        ):
            result = extract_weather_parameters(Path("fixture.grib"), SitePoint("site", 52, 4))
        self.assertEqual(set(result), set(WEATHER_PARAMETERS))
        self.assertIn((61, 0, 4), calls)
        self.assertIn((181, 0, 4), calls)
        self.assertIn((184, 0, 4), calls)
        self.assertIn((201, 0, 4), calls)

    def test_complete_as_of_p1_run_and_atomic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "weather.sqlite"
            conn = sqlite3.connect(db_path)
            create_harmonie_knmi_features_table(conn)
            targets = pd.date_range("2026-09-25T06:00:00Z", periods=3, freq="1h")
            records = []
            for i, target in enumerate(targets):
                records.append(
                    {
                        "source": "knmi_harmonie_p1", "dataset": "p1",
                        "run_ts": "2026-09-25T00:00:00+00:00",
                        "fetched_ts": "2026-09-25T01:00:00+00:00",
                        "target_ts": target.isoformat(), "horizon_hr": i + 6,
                        "site": "valkenburgsemeer", "temperature_2m_c": None if i == 1 else 10 + i,
                        "weather_code": 2,
                    }
                )
            upsert_harmonie_knmi_features(conn, pd.DataFrame(records))
            conn.close()
            timeline = build_weather_timeline(
                db_path, site="valkenburgsemeer", target_times=targets,
                fallback_temperature_c=[20, 21, 22], fallback_weather_code=[0, 0, 0],
                available_at="2026-09-25T02:00:00Z",
            )
            self.assertEqual(timeline["weather_source"].tolist(), ["knmi_p1", "windsurfice", "knmi_p1"])
            self.assertEqual(timeline["forecast_temperature_c"].tolist(), [10, 21, 12])
            before_fetch = build_weather_timeline(
                db_path, site="valkenburgsemeer", target_times=targets,
                fallback_temperature_c=[20, 21, 22], fallback_weather_code=[0, 0, 0],
                available_at="2026-09-25T00:30:00Z",
            )
            self.assertEqual(before_fetch["weather_source"].tolist(), ["windsurfice"] * 3)


if __name__ == "__main__":
    unittest.main()
