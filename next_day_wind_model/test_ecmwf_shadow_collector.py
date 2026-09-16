from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from next_day_wind_model.ecmwf_open_data import gust_parameters_for_leads
from next_day_wind_model.ecmwf_shadow import EcmwfShadowCollector, ifs_oper_lead_hours
from next_day_wind_model.forecast_provider import (
    ForecastBatch,
    ForecastPoint,
    ForecastRun,
    ForecastSite,
    ForecastValue,
    ForecastValueBatch,
)
from next_day_wind_model.forecast_store import (
    archive_coverage,
    create_forecast_points_table,
    upsert_forecast_points,
    upsert_forecast_values,
)


RUN_TIME = datetime(2026, 9, 13, 0, tzinfo=timezone.utc)
RUN = ForecastRun("ECMWF", "IFS", RUN_TIME, "ecmwf_open_data", resolution="0.25 degrees")
SITE = ForecastSite("valkenburgsemeer", 52.168, 4.437)


def point(lead: int, *, site: ForecastSite = SITE) -> ForecastPoint:
    return ForecastPoint.from_uv(
        provider="ECMWF",
        model="IFS",
        run_time=RUN_TIME,
        valid_time=RUN_TIME + timedelta(hours=lead),
        lead_time_hours=lead,
        site=site,
        grid_latitude=52.25,
        grid_longitude=4.5,
        u10_mps=3.0,
        v10_mps=4.0,
        source="test",
        fetched_time=RUN_TIME + timedelta(hours=8),
        resolution="0.25 degrees",
        metadata={"u_units": "m s**-1", "v_units": "m s**-1"},
    )


def gust(lead: int, *, site: ForecastSite = SITE) -> ForecastValue:
    return ForecastValue(
        provider="ECMWF",
        model="IFS",
        run_time=RUN_TIME,
        valid_time=RUN_TIME + timedelta(hours=lead),
        lead_time_hours=lead,
        site=site,
        grid_latitude=52.25,
        grid_longitude=4.5,
        variable="wind_gust_10m",
        value=8.5,
        unit="m s**-1",
        source="test",
        fetched_time=RUN_TIME + timedelta(hours=8),
        resolution="0.25 degrees",
        metadata={"short_name": "10fg", "step_type": "max"},
    )


class FakeProvider:
    provider_name = "ECMWF"
    model_name = "IFS"

    def __init__(self, *, fail_gust_once: bool = False) -> None:
        self.wind_calls: list[tuple[int, ...]] = []
        self.gust_calls: list[tuple[int, ...]] = []
        self.wind_site_calls: list[tuple[str, ...]] = []
        self.gust_site_calls: list[tuple[str, ...]] = []
        self.fail_gust_once = fail_gust_once

    def latest_run(self, required_lead_hours: tuple[int, ...]) -> ForecastRun:
        return RUN

    def fetch_wind(
        self, run: ForecastRun, site: ForecastSite, lead_hours: tuple[int, ...], work_dir: Path
    ) -> ForecastBatch:
        return self.fetch_wind_sites(run, (site,), lead_hours, work_dir)

    def fetch_wind_sites(
        self, run: ForecastRun, sites: tuple[ForecastSite, ...], lead_hours: tuple[int, ...], work_dir: Path
    ) -> ForecastBatch:
        leads = tuple(lead_hours)
        self.wind_calls.append(leads)
        self.wind_site_calls.append(tuple(site.name for site in sites))
        work_dir.mkdir(parents=True, exist_ok=True)
        artifact = work_dir / f"wind-{len(self.wind_calls)}.grib2"
        artifact.write_bytes(b"wind")
        return ForecastBatch(
            run,
            tuple(point(lead, site=site) for site in sites for lead in leads),
            (artifact,),
        )

    def fetch_gust_values(
        self, run: ForecastRun, site: ForecastSite, lead_hours: tuple[int, ...], work_dir: Path
    ) -> ForecastValueBatch:
        return self.fetch_gust_values_sites(run, (site,), lead_hours, work_dir)

    def fetch_gust_values_sites(
        self, run: ForecastRun, sites: tuple[ForecastSite, ...], lead_hours: tuple[int, ...], work_dir: Path
    ) -> ForecastValueBatch:
        leads = tuple(lead_hours)
        self.gust_calls.append(leads)
        self.gust_site_calls.append(tuple(site.name for site in sites))
        if self.fail_gust_once:
            self.fail_gust_once = False
            raise OSError("simulated interrupted gust download")
        work_dir.mkdir(parents=True, exist_ok=True)
        artifact = work_dir / f"gust-{len(self.gust_calls)}.grib2"
        artifact.write_bytes(b"gust")
        return ForecastValueBatch(
            run,
            tuple(gust(lead, site=site) for site in sites for lead in leads),
            (artifact,),
        )


class LeadScheduleTests(unittest.TestCase):
    def test_first_48_hours_are_all_native_three_hour_steps(self) -> None:
        self.assertEqual(ifs_oper_lead_hours(48), tuple(range(0, 49, 3)))
        self.assertEqual(len(ifs_oper_lead_hours(48)), 17)

    def test_schedule_changes_to_six_hour_steps_after_144(self) -> None:
        steps = ifs_oper_lead_hours(162)
        self.assertEqual(steps[-6:], (138, 141, 144, 150, 156, 162))
        self.assertNotIn(147, steps)

    def test_five_day_schedule_uses_41_native_three_hour_steps(self) -> None:
        steps = ifs_oper_lead_hours(120)
        self.assertEqual(steps, tuple(range(0, 121, 3)))
        self.assertEqual(len(steps), 41)
        self.assertEqual(len(tuple(step for step in steps if step > 0)), 40)
        self.assertEqual(gust_parameters_for_leads(steps[1:]), ("10fg", "10fg3"))
        self.assertEqual(gust_parameters_for_leads((93, 120)), ("10fg3",))


class ShadowStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp.name) / "staging"
        self.conn = sqlite3.connect(":memory:")
        create_forecast_points_table(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def collector(self, provider: FakeProvider) -> EcmwfShadowCollector:
        return EcmwfShadowCollector(provider=provider, conn=self.conn, work_dir=self.work_dir)

    def test_already_complete_run_is_not_downloaded_twice(self) -> None:
        provider = FakeProvider()
        collector = self.collector(provider)

        first = collector.collect_run(RUN, SITE, (0, 3, 6), include_gust=True)
        second = collector.collect_run(RUN, SITE, (0, 3, 6), include_gust=True)

        self.assertTrue(first.after.complete)
        self.assertTrue(second.after.complete)
        self.assertFalse(second.attempted)
        self.assertEqual(provider.wind_calls, [(0, 3, 6)])
        self.assertEqual(provider.gust_calls, [(3, 6)])
        count = self.conn.execute("SELECT COUNT(*) FROM forecast_points").fetchone()[0]
        self.assertEqual(count, 3)

    def test_only_missing_leads_are_retrieved(self) -> None:
        upsert_forecast_points(self.conn, (point(0), point(3)))
        upsert_forecast_values(self.conn, (gust(3),))
        provider = FakeProvider()

        result = self.collector(provider).collect_run(RUN, SITE, (0, 3, 6), include_gust=True)

        self.assertTrue(result.after.complete)
        self.assertEqual(provider.wind_calls, [(6,)])
        self.assertEqual(provider.gust_calls, [(6,)])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM forecast_points").fetchone()[0],
            3,
        )

    def test_interrupted_gust_recovers_without_redownloading_uv(self) -> None:
        provider = FakeProvider(fail_gust_once=True)
        collector = self.collector(provider)

        with self.assertRaisesRegex(OSError, "interrupted gust"):
            collector.collect_run(RUN, SITE, (0, 3, 6), include_gust=True)

        partial = archive_coverage(
            self.conn,
            provider="ECMWF",
            model="IFS",
            run_time="2026-09-13T00:00:00Z",
            site=SITE.name,
            expected_wind_leads=(0, 3, 6),
            expected_gust_leads=(3, 6),
        )
        status = self.conn.execute("SELECT status FROM forecast_collection_runs").fetchone()[0]
        self.assertFalse(partial.complete)
        self.assertEqual(partial.missing_wind_leads, frozenset())
        self.assertEqual(status, "partial")

        recovered = collector.collect_run(RUN, SITE, (0, 3, 6), include_gust=True)

        self.assertTrue(recovered.after.complete)
        self.assertEqual(provider.wind_calls, [(0, 3, 6)])
        self.assertEqual(provider.gust_calls, [(3, 6), (3, 6)])
        status, attempts, error = self.conn.execute(
            "SELECT status, attempts, last_error FROM forecast_collection_runs"
        ).fetchone()
        self.assertEqual((status, attempts, error), ("complete", 2, None))

    def test_gust_metadata_is_merged_into_wind_point(self) -> None:
        upsert_forecast_points(self.conn, (point(3),))

        self.assertEqual(upsert_forecast_values(self.conn, (gust(3),)), 1)

        gust_value, metadata_json = self.conn.execute(
            "SELECT wind_gust_mps, metadata_json FROM forecast_points"
        ).fetchone()
        metadata = json.loads(metadata_json)
        self.assertEqual(gust_value, 8.5)
        self.assertEqual(metadata["gust_unit"], "m s**-1")
        self.assertEqual(metadata["gust_metadata"]["step_type"], "max")

    def test_scalar_value_cannot_precede_mandatory_uv(self) -> None:
        with self.assertRaisesRegex(ValueError, "before its U/V point"):
            upsert_forecast_values(self.conn, (gust(3),))

    def test_same_model_run_and_valid_time_are_stored_for_multiple_sites(self) -> None:
        second_site = ForecastSite("oostvoorne", 51.9278, 4.05502)

        upsert_forecast_points(
            self.conn,
            (point(3, site=SITE), point(3, site=second_site)),
        )
        upsert_forecast_points(self.conn, (point(3, site=SITE),))

        rows = self.conn.execute(
            "SELECT site, COUNT(*) FROM forecast_points GROUP BY site ORDER BY site"
        ).fetchall()
        self.assertEqual(rows, [("oostvoorne", 1), ("valkenburgsemeer", 1)])

    def test_one_retrieval_per_field_group_populates_two_sites(self) -> None:
        second_site = ForecastSite("oostvoorne", 51.9278, 4.05502)
        provider = FakeProvider()

        result = self.collector(provider).collect_run_sites(
            RUN,
            (SITE, second_site),
            (0, 3, 6),
            include_gust=True,
        )

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.retrieve_calls, 2)
        self.assertEqual(provider.wind_site_calls, [("valkenburgsemeer", "oostvoorne")])
        self.assertEqual(provider.gust_site_calls, [("valkenburgsemeer", "oostvoorne")])
        self.assertEqual(
            self.conn.execute(
                "SELECT site, COUNT(*) FROM forecast_points GROUP BY site ORDER BY site"
            ).fetchall(),
            [("oostvoorne", 3), ("valkenburgsemeer", 3)],
        )
        downloads = self.conn.execute(
            "SELECT site, sites_json, field_group FROM forecast_downloads ORDER BY field_group"
        ).fetchall()
        self.assertEqual(len(downloads), 2)
        self.assertTrue(all(row[0] == "__shared__" for row in downloads))
        self.assertTrue(
            all(json.loads(row[1]) == ["oostvoorne", "valkenburgsemeer"] for row in downloads)
        )

    def test_historical_vintages_with_same_valid_time_are_preserved(self) -> None:
        newer_run = RUN_TIME + timedelta(hours=3)
        older = point(3)
        newer = ForecastPoint.from_uv(
            provider="ECMWF",
            model="IFS",
            run_time=newer_run,
            valid_time=newer_run,
            lead_time_hours=0,
            site=SITE,
            grid_latitude=52.25,
            grid_longitude=4.5,
            u10_mps=6.0,
            v10_mps=1.0,
            source="test",
            fetched_time=newer_run + timedelta(hours=8),
            resolution="0.25 degrees",
        )

        upsert_forecast_points(self.conn, (older, newer))

        rows = self.conn.execute(
            "SELECT run_time, valid_time FROM forecast_points ORDER BY run_time"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], rows[1][1])


if __name__ == "__main__":
    unittest.main()
