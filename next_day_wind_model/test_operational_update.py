from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from next_day_wind_model.operational_update import (
    CachedArtifactStatus,
    ExecutionDecision,
    ForecastIdentity,
    OperationalSnapshot,
    compute_forecast_fingerprint,
    decide_execution_mode,
    launch_operational_update,
    load_latest_forecast_identity,
    state_path_for,
    validate_cached_prediction_artifacts,
    write_bytes_if_changed,
)


def _forecast_rows() -> list[dict]:
    return [
        {
            "timestamp": 1784275200000 + hour * 3_600_000,
            "WindForecastAvr": 5 + hour / 10,
            "WindForecastMax": 8 + hour / 10,
            "WindDirection": 180 + hour,
            "Temperature": 20,
            "Pressure": 1013,
            "Rain": 0,
            "RH": 60,
            "Clouds": 4,
            "low_cloud_cover": 2,
            "medium_cloud_cover": 1,
            "high_cloud_cover": 1,
            "cloud_base": 1200,
            "global_radiation": 250,
        }
        for hour in range(24)
    ]


def _snapshot(
    *,
    observation_max_ts: int = 200,
    forecast_hash: str = "forecast-a",
    model_hash: str | None = "model-a",
    cache_valid: bool = True,
    cache_hash: str | None = "cache-a",
) -> OperationalSnapshot:
    return OperationalSnapshot(
        site="valkenburgsemeer",
        model="HARMONIE",
        observation_max_ts=observation_max_ts,
        forecast=ForecastIdentity(forecast_hash, 10, 10, None, 24),
        model_fingerprint=model_hash,
        cached_artifacts=CachedArtifactStatus(cache_valid, cache_hash, "ok" if cache_valid else "missing"),
    )


def _state(**overrides) -> dict:
    state = {
        "forecast_fingerprint": "forecast-a",
        "model_fingerprint": "model-a",
        "cached_prediction_fingerprint": "cache-a",
        "observation_max_ts": 200,
    }
    state.update(overrides)
    return state


class ForecastFingerprintTests(unittest.TestCase):
    def test_identical_content_ignores_fetch_time_formatting_and_row_order(self) -> None:
        first = _forecast_rows()
        second = list(reversed([dict(row) for row in first]))
        for row in first:
            row["fetched_ts"] = 1
            row["transport"] = {"request": "a"}
        for row in second:
            row["fetched_ts"] = 999
            row["transport"] = {"request": "b"}
            row["WindForecastAvr"] = str(row["WindForecastAvr"])
        self.assertEqual(
            compute_forecast_fingerprint(first, site="Site", model="harmonie"),
            compute_forecast_fingerprint(second, site="site", model="HARMONIE"),
        )

    def test_changed_speed_changes_fingerprint(self) -> None:
        changed = _forecast_rows()
        changed[3]["WindForecastAvr"] += 0.01
        self.assertNotEqual(
            compute_forecast_fingerprint(_forecast_rows(), site="s", model="HARMONIE"),
            compute_forecast_fingerprint(changed, site="s", model="HARMONIE"),
        )

    def test_changed_direction_changes_fingerprint(self) -> None:
        changed = _forecast_rows()
        changed[3]["WindDirection"] += 1
        self.assertNotEqual(
            compute_forecast_fingerprint(_forecast_rows(), site="s", model="HARMONIE"),
            compute_forecast_fingerprint(changed, site="s", model="HARMONIE"),
        )

    def test_equivalent_timezone_timestamps_match(self) -> None:
        row_a = dict(_forecast_rows()[0], timestamp="2026-07-17T10:00:00+02:00")
        row_b = dict(_forecast_rows()[0], timestamp="2026-07-17T08:00:00Z")
        self.assertEqual(
            compute_forecast_fingerprint([row_a], site="s", model="m"),
            compute_forecast_fingerprint([row_b], site="s", model="m"),
        )

    def test_same_nominal_issue_with_changed_content_differs(self) -> None:
        changed = _forecast_rows()
        changed[0]["WindDirection"] += 5
        issue = "2026-07-17T06:00:00Z"
        self.assertNotEqual(
            compute_forecast_fingerprint(_forecast_rows(), site="s", model="m", source_run_ts=issue),
            compute_forecast_fingerprint(changed, site="s", model="m", source_run_ts=issue),
        )

    def test_authoritative_issue_is_part_of_identity(self) -> None:
        self.assertNotEqual(
            compute_forecast_fingerprint(
                _forecast_rows(), site="s", model="m", source_run_ts="2026-07-17T06:00:00Z"
            ),
            compute_forecast_fingerprint(
                _forecast_rows(), site="s", model="m", source_run_ts="2026-07-17T07:00:00Z"
            ),
        )

    def test_latest_identity_reads_only_latest_run_and_ignores_fallback_fetch_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "fixture.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE forecasts(
                    site TEXT, model TEXT, run_ts INTEGER, fetched_ts INTEGER,
                    target_ts INTEGER, wind_speed REAL, wind_gust REAL,
                    wind_dir REAL, payload TEXT,
                    PRIMARY KEY(site, model, run_ts, target_ts)
                )
                """
            )
            for run_ts, fetched_ts in [(1000, 1000), (2000, 2000)]:
                for row in _forecast_rows():
                    conn.execute(
                        "INSERT INTO forecasts VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            "s",
                            "HARMONIE",
                            run_ts,
                            fetched_ts,
                            row["timestamp"],
                            row["WindForecastAvr"],
                            row["WindForecastMax"],
                            row["WindDirection"],
                            json.dumps(row),
                        ),
                    )
            conn.commit()
            conn.close()
            identity = load_latest_forecast_identity(db_path, site="s", model="HARMONIE")
            self.assertEqual(identity.run_ts, 2000)
            self.assertIsNone(identity.source_run_ts)
            self.assertEqual(identity.row_count, 24)


class ExecutionModeTests(unittest.TestCase):
    def assertDecision(self, expected: str, snapshot: OperationalSnapshot, state: dict | None) -> None:
        self.assertEqual(decide_execution_mode(snapshot, state).mode, expected)

    def test_no_change(self) -> None:
        self.assertDecision("no_change", _snapshot(), _state())

    def test_new_observations_measured_only(self) -> None:
        self.assertDecision("measured_only", _snapshot(observation_max_ts=201), _state())

    def test_changed_forecast_full(self) -> None:
        self.assertDecision("forecast_changed", _snapshot(forecast_hash="forecast-b"), _state())

    def test_changed_forecast_and_observations_full(self) -> None:
        self.assertDecision(
            "forecast_changed",
            _snapshot(observation_max_ts=201, forecast_hash="forecast-b"),
            _state(),
        )

    def test_changed_model_full(self) -> None:
        self.assertDecision("model_changed", _snapshot(model_hash="model-b"), _state())

    def test_missing_state_full(self) -> None:
        self.assertDecision("recovery_missing_state", _snapshot(), None)

    def test_missing_cached_artifact_full(self) -> None:
        self.assertDecision(
            "recovery_missing_cache",
            _snapshot(cache_valid=False, cache_hash=None),
            _state(),
        )

    def test_changed_cached_prediction_full(self) -> None:
        self.assertDecision(
            "recovery_cached_prediction_changed",
            _snapshot(cache_hash="cache-b"),
            _state(),
        )

    def test_old_state_after_failed_run_retries(self) -> None:
        self.assertDecision("forecast_changed", _snapshot(forecast_hash="forecast-new"), _state())

    def test_failed_child_never_advances_state(self) -> None:
        snapshot = _snapshot(forecast_hash="forecast-new")
        argv = ["--skip-training", "--skip-data-refresh-check"]
        with (
            mock.patch("next_day_wind_model.operational_update._collect_snapshot", return_value=snapshot),
            mock.patch("next_day_wind_model.operational_update.load_success_state", return_value=_state()),
            mock.patch("next_day_wind_model.operational_update._run_child", return_value=9) as child,
            mock.patch("next_day_wind_model.operational_update.write_success_state") as write_state,
        ):
            result = launch_operational_update(Path("fake.py"), argv)
        self.assertEqual(result, 9)
        child.assert_called_once_with(Path("fake.py"), argv, measured_only=False)
        write_state.assert_not_called()

    def test_measured_mode_launches_only_measured_child_and_advances_state(self) -> None:
        snapshot = _snapshot(observation_max_ts=201)
        argv = ["--skip-training", "--skip-data-refresh-check"]
        with (
            mock.patch("next_day_wind_model.operational_update._collect_snapshot", return_value=snapshot),
            mock.patch("next_day_wind_model.operational_update.load_success_state", return_value=_state()),
            mock.patch("next_day_wind_model.operational_update._run_child", return_value=0) as child,
            mock.patch(
                "next_day_wind_model.operational_update.compute_model_fingerprint",
                return_value=("model-a", ()),
            ),
            mock.patch(
                "next_day_wind_model.operational_update.validate_cached_prediction_artifacts",
                return_value=CachedArtifactStatus(True, "cache-a", "ok"),
            ),
            mock.patch("next_day_wind_model.operational_update.write_success_state") as write_state,
        ):
            result = launch_operational_update(Path("fake.py"), argv)
        self.assertEqual(result, 0)
        child.assert_called_once_with(Path("fake.py"), argv, measured_only=True)
        self.assertEqual(write_state.call_args.kwargs["execution_mode"], "measured_only")

    def test_no_change_does_not_launch_child(self) -> None:
        snapshot = _snapshot()
        argv = ["--skip-training", "--skip-data-refresh-check"]
        with (
            mock.patch("next_day_wind_model.operational_update._collect_snapshot", return_value=snapshot),
            mock.patch("next_day_wind_model.operational_update.load_success_state", return_value=_state()),
            mock.patch("next_day_wind_model.operational_update._run_child") as child,
        ):
            result = launch_operational_update(Path("fake.py"), argv)
        self.assertEqual(result, 0)
        child.assert_not_called()


class ArtifactAndStateTests(unittest.TestCase):
    def test_write_if_changed_preserves_unchanged_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.txt"
            self.assertTrue(write_bytes_if_changed(path, b"same"))
            first_mtime = path.stat().st_mtime_ns
            self.assertFalse(write_bytes_if_changed(path, b"same"))
            self.assertEqual(path.stat().st_mtime_ns, first_mtime)

    def test_state_paths_are_site_scoped(self) -> None:
        root = Path("artifacts")
        self.assertNotEqual(
            state_path_for(root, site="site-a", model="HARMONIE"),
            state_path_for(root, site="site-b", model="HARMONIE"),
        )

    def test_cached_artifact_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 17, 8, tzinfo=timezone.utc)
            current_header = [
                "time_local",
                "is_forecast_grid",
                "forecast_wind_speed",
                "forecast_wind_min",
                "forecast_wind_max",
                "forecast_wind_dir_deg",
                "lstm_pred_wind_speed_full",
                "lstm_pred_wind_dir_deg_full",
                "lstm_pred_wind_speed",
                "lstm_pred_wind_dir_deg",
            ]
            with (root / "current_day_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=current_header)
                writer.writeheader()
                writer.writerow(
                    {
                        "time_local": "2026-07-17T10:00:00+02:00",
                        "is_forecast_grid": True,
                        **{column: 5 for column in current_header[2:]},
                    }
                )
            next_header = ["target_time_utc", *(
                "forecast_wind_speed",
                "forecast_wind_min",
                "forecast_wind_max",
                "lstm_pred_wind_speed",
                "forecast_wind_dir_deg",
                "lstm_pred_wind_dir_deg",
            )]
            with (root / "next_day_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=next_header)
                writer.writeheader()
                writer.writerow({"target_time_utc": "2026-07-18T00:00:00Z", **{c: 5 for c in next_header[1:]}})
            (root / "metadata_update.json").write_text("{}", encoding="utf-8")
            (root / "current_day_predictions.png").write_bytes(b"png")
            (root / "current_day_predictions_mobile.png").write_bytes(b"png")
            (root / "next_day_predictions.png").write_bytes(b"png")
            (root / "next_day_predictions_mobile.png").write_bytes(b"png")
            status = validate_cached_prediction_artifacts(
                root,
                local_timezone="Europe/Amsterdam",
                now_utc=now,
            )
            self.assertTrue(status.valid, status.reason)
            self.assertIsNotNone(status.fingerprint)

            web = root / "web"
            missing_web = validate_cached_prediction_artifacts(
                root,
                local_timezone="Europe/Amsterdam",
                web_out_dir=web,
                now_utc=now,
            )
            self.assertFalse(missing_web.valid)
            web.mkdir()
            for filename in (
                "index.html",
                "metadata_update.json",
                "current_day_predictions.csv",
                "next_day_predictions.csv",
                "current_day_predictions.png",
                "current_day_predictions_mobile.png",
                "next_day_predictions.png",
                "next_day_predictions_mobile.png",
                "current_day_interactive_data.json",
                "next_day_interactive_data.json",
            ):
                (web / filename).write_bytes(b"present")
            published = validate_cached_prediction_artifacts(
                root,
                local_timezone="Europe/Amsterdam",
                web_out_dir=web,
                now_utc=now,
            )
            self.assertTrue(published.valid, published.reason)


if __name__ == "__main__":
    unittest.main()
