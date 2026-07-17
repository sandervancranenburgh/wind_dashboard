from __future__ import annotations

import builtins
import contextlib
import csv
import json
import os
import runpy
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from next_day_wind_model.operational_update import (
    CachedArtifactStatus,
    ExecutionDecision,
    FORECAST_FINGERPRINT_VERSION,
    ForecastIdentity,
    GATE_CHILD_ENV,
    OperationalSnapshot,
    STATE_SCHEMA_VERSION,
    _child_command,
    _run_child,
    compute_model_fingerprint,
    compute_forecast_fingerprint,
    decide_execution_mode,
    launch_operational_update,
    load_latest_forecast_identity,
    load_success_state,
    state_path_for,
    validate_cached_prediction_artifacts,
    write_bytes_if_changed,
    write_success_state,
)


FORECAST_A = "a" * 64
FORECAST_B = "b" * 64
MODEL_A = "c" * 64
MODEL_B = "d" * 64
CACHE_A = "e" * 64
CACHE_B = "f" * 64


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
    forecast_hash: str | None = FORECAST_A,
    model_hash: str | None = MODEL_A,
    cache_valid: bool = True,
    cache_hash: str | None = CACHE_A,
) -> OperationalSnapshot:
    return OperationalSnapshot(
        site="valkenburgsemeer",
        model="HARMONIE",
        observation_max_ts=observation_max_ts,
        forecast=None if forecast_hash is None else ForecastIdentity(forecast_hash, 10, 10, None, 24),
        model_fingerprint=model_hash,
        cached_artifacts=CachedArtifactStatus(cache_valid, cache_hash, "ok" if cache_valid else "missing"),
    )


def _state(**overrides) -> dict:
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "fingerprint_version": FORECAST_FINGERPRINT_VERSION,
        "status": "success",
        "site": "valkenburgsemeer",
        "model": "HARMONIE",
        "forecast_fingerprint": FORECAST_A,
        "model_fingerprint": MODEL_A,
        "cached_prediction_fingerprint": CACHE_A,
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
        self.assertDecision("forecast_changed", _snapshot(forecast_hash=FORECAST_B), _state())

    def test_changed_forecast_and_observations_full(self) -> None:
        self.assertDecision(
            "forecast_changed",
            _snapshot(observation_max_ts=201, forecast_hash=FORECAST_B),
            _state(),
        )

    def test_changed_model_full(self) -> None:
        self.assertDecision("model_changed", _snapshot(model_hash=MODEL_B), _state())

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
            _snapshot(cache_hash=CACHE_B),
            _state(),
        )

    def test_old_state_after_failed_run_retries(self) -> None:
        self.assertDecision("forecast_changed", _snapshot(forecast_hash=FORECAST_B), _state())

    def test_failed_child_never_advances_state(self) -> None:
        snapshot = _snapshot(forecast_hash=FORECAST_B)
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
                return_value=(MODEL_A, ()),
            ),
            mock.patch(
                "next_day_wind_model.operational_update.validate_cached_prediction_artifacts",
                return_value=CachedArtifactStatus(True, CACHE_A, "ok"),
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


class StatePersistenceTests(unittest.TestCase):
    def _load_payload(self, payload: dict) -> dict | None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_success_state(path, site="valkenburgsemeer", model="HARMONIE")

    def test_valid_state_with_matching_versions_loads(self) -> None:
        self.assertEqual(self._load_payload(_state()), _state())

    def test_missing_state_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            self.assertIsNone(load_success_state(path, site="valkenburgsemeer", model="HARMONIE"))

    def test_corrupt_state_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("{not-json", encoding="utf-8")
            self.assertIsNone(load_success_state(path, site="valkenburgsemeer", model="HARMONIE"))

    def test_wrong_schema_version_returns_none(self) -> None:
        self.assertIsNone(self._load_payload(_state(schema_version=STATE_SCHEMA_VERSION + 1)))

    def test_fingerprint_version_is_mandatory_and_exact(self) -> None:
        cases = {
            "missing": None,
            "lower": FORECAST_FINGERPRINT_VERSION - 1,
            "higher": FORECAST_FINGERPRINT_VERSION + 1,
        }
        for label, version in cases.items():
            with self.subTest(label=label):
                payload = _state()
                if version is None:
                    payload.pop("fingerprint_version")
                else:
                    payload["fingerprint_version"] = version
                self.assertIsNone(self._load_payload(payload))

    def test_missing_or_invalid_forecast_fingerprint_returns_none(self) -> None:
        for label, fingerprint in (("missing", None), ("invalid", "not-a-sha256")):
            with self.subTest(label=label):
                payload = _state()
                if fingerprint is None:
                    payload.pop("forecast_fingerprint")
                else:
                    payload["forecast_fingerprint"] = fingerprint
                self.assertIsNone(self._load_payload(payload))

    def test_different_site_or_model_returns_none(self) -> None:
        for label, overrides in (
            ("site", {"site": "other-site"}),
            ("model", {"model": "OTHER"}),
        ):
            with self.subTest(label=label):
                self.assertIsNone(self._load_payload(_state(**overrides)))

    def test_success_state_writes_both_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            write_success_state(
                path,
                snapshot=_snapshot(),
                execution_mode="forecast_changed",
                successful_at_utc=datetime(2026, 7, 17, 8, tzinfo=timezone.utc),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], STATE_SCHEMA_VERSION)
        self.assertEqual(payload["fingerprint_version"], FORECAST_FINGERPRINT_VERSION)


class OperationalLauncherTests(unittest.TestCase):
    BASE_ARGS = ("--skip-training", "--skip-data-refresh-check")
    SCRIPT = Path("fake.py")

    def _argv(self, root: Path, extra: tuple[str, ...] = ()) -> list[str]:
        return [
            *self.BASE_ARGS,
            "--out-dir",
            str(root / "out"),
            "--web-out-dir",
            str(root / "web"),
            *extra,
        ]

    def _invoke(
        self,
        root: Path,
        *,
        snapshot: OperationalSnapshot | None = None,
        state: dict | None = None,
        patch_state: bool = True,
        child_return: int = 0,
        child_side_effect: BaseException | None = None,
        collect_error: Exception | None = None,
        post_model: str | None | object = ...,
        post_cache: CachedArtifactStatus | object = ...,
        extra_argv: tuple[str, ...] = (),
        operational_args: bool = True,
    ) -> dict:
        snapshot = _snapshot() if snapshot is None and collect_error is None else snapshot
        argv = self._argv(root, extra_argv) if operational_args else [
            "--out-dir",
            str(root / "out"),
            "--web-out-dir",
            str(root / "web"),
            *extra_argv,
        ]
        state_path = state_path_for(root / "out", site="valkenburgsemeer", model="HARMONIE")
        if post_model is ...:
            post_model = snapshot.model_fingerprint if snapshot and snapshot.model_fingerprint else MODEL_A
        if post_cache is ...:
            if snapshot and snapshot.cached_artifacts.valid:
                post_cache = snapshot.cached_artifacts
            else:
                post_cache = CachedArtifactStatus(True, CACHE_A, "ok")

        collect_context = mock.patch(
            "next_day_wind_model.operational_update._collect_snapshot",
            side_effect=collect_error,
        ) if collect_error is not None else mock.patch(
            "next_day_wind_model.operational_update._collect_snapshot",
            return_value=snapshot,
        )
        state_context = (
            mock.patch("next_day_wind_model.operational_update.load_success_state", return_value=state)
            if patch_state
            else contextlib.nullcontext()
        )
        child_context = (
            mock.patch("next_day_wind_model.operational_update._run_child", side_effect=child_side_effect)
            if child_side_effect is not None
            else mock.patch("next_day_wind_model.operational_update._run_child", return_value=child_return)
        )
        decisions: list[str] = []
        with (
            collect_context as collect,
            state_context,
            child_context as child,
            mock.patch(
                "next_day_wind_model.operational_update.compute_model_fingerprint",
                return_value=(post_model, ()),
            ) as model_fingerprint,
            mock.patch(
                "next_day_wind_model.operational_update.validate_cached_prediction_artifacts",
                return_value=post_cache,
            ) as cache_validation,
            mock.patch(
                "next_day_wind_model.operational_update._log_decision",
                side_effect=lambda decision, _snapshot: decisions.append(decision.mode),
            ),
        ):
            result = launch_operational_update(self.SCRIPT, argv)
        return {
            "result": result,
            "argv": argv,
            "state_path": state_path,
            "decisions": decisions,
            "collect": collect,
            "child": child,
            "model_fingerprint": model_fingerprint,
            "cache_validation": cache_validation,
        }

    def test_complete_decision_matrix_controls_child_and_state(self) -> None:
        cases = (
            ("no_change", _snapshot(), _state(), False, False, False),
            ("measured_only", _snapshot(observation_max_ts=201), _state(), True, True, True),
            ("forecast_changed", _snapshot(forecast_hash=FORECAST_B), _state(), True, False, True),
            ("model_changed", _snapshot(model_hash=MODEL_B), _state(), True, False, True),
            ("recovery_missing_state", _snapshot(), None, True, False, True),
            ("recovery_forecast_identity", _snapshot(forecast_hash=None), _state(), True, False, False),
            (
                "recovery_missing_cache",
                _snapshot(cache_valid=False, cache_hash=None),
                _state(),
                True,
                False,
                True,
            ),
            ("recovery_model_artifacts", _snapshot(model_hash=None), _state(), True, False, True),
            (
                "recovery_cached_prediction_changed",
                _snapshot(cache_hash=CACHE_B),
                _state(),
                True,
                False,
                True,
            ),
            (
                "recovery_observation_state",
                _snapshot(observation_max_ts=None),
                _state(),
                True,
                False,
                True,
            ),
            (
                "recovery_observation_regressed",
                _snapshot(observation_max_ts=199),
                _state(),
                True,
                False,
                True,
            ),
        )
        for mode, snapshot, state, launched, measured_only, advances in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                outcome = self._invoke(Path(directory), snapshot=snapshot, state=state)
                self.assertEqual(outcome["result"], 0)
                self.assertEqual(outcome["decisions"], [mode])
                if launched:
                    outcome["child"].assert_called_once_with(
                        self.SCRIPT,
                        outcome["argv"],
                        measured_only=measured_only,
                    )
                else:
                    outcome["child"].assert_not_called()
                    outcome["model_fingerprint"].assert_not_called()
                    outcome["cache_validation"].assert_not_called()
                self.assertEqual(outcome["state_path"].is_file(), advances)
                if advances:
                    payload = json.loads(outcome["state_path"].read_text(encoding="utf-8"))
                    self.assertEqual(payload["execution_mode"], mode)
                    self.assertEqual(payload["fingerprint_version"], FORECAST_FINGERPRINT_VERSION)

    def test_startup_state_validation_controls_expensive_launch(self) -> None:
        variants = (
            ("valid", _state(), "no_change", False),
            ("missing", None, "recovery_missing_state", True),
            ("corrupt", b"{broken", "recovery_missing_state", True),
            (
                "wrong_schema",
                _state(schema_version=STATE_SCHEMA_VERSION + 1),
                "recovery_missing_state",
                True,
            ),
            (
                "missing_fingerprint_version",
                {key: value for key, value in _state().items() if key != "fingerprint_version"},
                "recovery_missing_state",
                True,
            ),
            (
                "lower_fingerprint_version",
                _state(fingerprint_version=FORECAST_FINGERPRINT_VERSION - 1),
                "recovery_missing_state",
                True,
            ),
            (
                "higher_fingerprint_version",
                _state(fingerprint_version=FORECAST_FINGERPRINT_VERSION + 1),
                "recovery_missing_state",
                True,
            ),
            (
                "missing_forecast_fingerprint",
                {key: value for key, value in _state().items() if key != "forecast_fingerprint"},
                "recovery_missing_state",
                True,
            ),
            (
                "invalid_forecast_fingerprint",
                _state(forecast_fingerprint="invalid"),
                "recovery_missing_state",
                True,
            ),
            ("different_site", _state(site="other"), "recovery_missing_state", True),
            ("different_model", _state(model="OTHER"), "recovery_missing_state", True),
        )
        for label, persisted, mode, launched in variants:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state_path = state_path_for(root / "out", site="valkenburgsemeer", model="HARMONIE")
                if persisted is not None:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(persisted, bytes):
                        state_path.write_bytes(persisted)
                    else:
                        state_path.write_text(json.dumps(persisted), encoding="utf-8")
                before = state_path.read_bytes() if state_path.is_file() else None
                outcome = self._invoke(root, snapshot=_snapshot(), patch_state=False)
                self.assertEqual(outcome["decisions"], [mode])
                if launched:
                    outcome["child"].assert_called_once_with(
                        self.SCRIPT,
                        outcome["argv"],
                        measured_only=False,
                    )
                    self.assertNotEqual(state_path.read_bytes(), before)
                    self.assertEqual(
                        json.loads(state_path.read_text(encoding="utf-8"))["fingerprint_version"],
                        FORECAST_FINGERPRINT_VERSION,
                    )
                else:
                    outcome["child"].assert_not_called()
                    self.assertEqual(state_path.read_bytes(), before)

    def test_forecast_identity_paths(self) -> None:
        cases = (
            ("unchanged", _snapshot(), "no_change", False, False),
            ("changed", _snapshot(forecast_hash=FORECAST_B), "forecast_changed", True, True),
            ("missing", _snapshot(forecast_hash=None), "recovery_forecast_identity", True, False),
            ("invalid", _snapshot(forecast_hash="invalid"), "recovery_forecast_identity", True, False),
        )
        for label, snapshot, mode, launched, advances in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                outcome = self._invoke(Path(directory), snapshot=snapshot, state=_state())
                self.assertEqual(outcome["decisions"], [mode])
                self.assertEqual(outcome["child"].called, launched)
                if launched:
                    outcome["child"].assert_called_once_with(
                        self.SCRIPT,
                        outcome["argv"],
                        measured_only=False,
                    )
                self.assertEqual(outcome["state_path"].is_file(), advances)

    def test_observation_watermark_paths(self) -> None:
        cases = (
            ("unchanged", 200, "no_change", False, False),
            ("increased", 201, "measured_only", True, True),
            ("regressed", 199, "recovery_observation_regressed", True, False),
            ("missing_current", None, "recovery_observation_state", True, False),
            ("missing_saved", 200, "recovery_observation_state", True, False),
        )
        for label, current, mode, launched, measured_only in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                state = _state(observation_max_ts=None) if label == "missing_saved" else _state()
                outcome = self._invoke(
                    Path(directory),
                    snapshot=_snapshot(observation_max_ts=current),
                    state=state,
                )
                self.assertEqual(outcome["decisions"], [mode])
                if launched:
                    outcome["child"].assert_called_once_with(
                        self.SCRIPT,
                        outcome["argv"],
                        measured_only=measured_only,
                    )
                else:
                    outcome["child"].assert_not_called()

    def test_missing_incomplete_cache_and_model_artifacts_force_full_recovery(self) -> None:
        cases = (
            (
                "missing_cache",
                _snapshot(cache_valid=False, cache_hash=None),
                "recovery_missing_cache",
            ),
            (
                "incomplete_cache",
                OperationalSnapshot(
                    site="valkenburgsemeer",
                    model="HARMONIE",
                    observation_max_ts=200,
                    forecast=ForecastIdentity(FORECAST_A, 10, 10, None, 24),
                    model_fingerprint=MODEL_A,
                    cached_artifacts=CachedArtifactStatus(False, None, "invalid:partial cache"),
                ),
                "recovery_missing_cache",
            ),
            ("missing_model_artifacts", _snapshot(model_hash=None), "recovery_model_artifacts"),
        )
        for label, snapshot, mode in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                outcome = self._invoke(Path(directory), snapshot=snapshot, state=_state())
                self.assertEqual(outcome["decisions"], [mode])
                outcome["child"].assert_called_once_with(
                    self.SCRIPT,
                    outcome["argv"],
                    measured_only=False,
                )
                self.assertTrue(outcome["state_path"].is_file())

    def test_nonzero_child_exit_never_advances_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = state_path_for(root / "out", site="valkenburgsemeer", model="HARMONIE")
            state_path.parent.mkdir(parents=True)
            state_path.write_bytes(b"last-success")
            outcome = self._invoke(
                root,
                snapshot=_snapshot(forecast_hash=FORECAST_B),
                state=_state(),
                child_return=17,
            )
            self.assertEqual(outcome["result"], 17)
            self.assertEqual(state_path.read_bytes(), b"last-success")
            outcome["model_fingerprint"].assert_not_called()
            outcome["cache_validation"].assert_not_called()

    def test_child_keyboard_interrupt_never_advances_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = state_path_for(root / "out", site="valkenburgsemeer", model="HARMONIE")
            state_path.parent.mkdir(parents=True)
            state_path.write_bytes(b"last-success")
            with self.assertRaises(KeyboardInterrupt):
                self._invoke(
                    root,
                    snapshot=_snapshot(forecast_hash=FORECAST_B),
                    state=_state(),
                    child_side_effect=KeyboardInterrupt(),
                )
            self.assertEqual(state_path.read_bytes(), b"last-success")

    def test_unexpected_child_exception_never_advances_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = state_path_for(root / "out", site="valkenburgsemeer", model="HARMONIE")
            state_path.parent.mkdir(parents=True)
            state_path.write_bytes(b"last-success")
            with self.assertRaisesRegex(RuntimeError, "child exploded"):
                self._invoke(
                    root,
                    snapshot=_snapshot(forecast_hash=FORECAST_B),
                    state=_state(),
                    child_side_effect=RuntimeError("child exploded"),
                )
            self.assertEqual(state_path.read_bytes(), b"last-success")

    def test_partial_cache_generation_never_advances_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = state_path_for(root / "out", site="valkenburgsemeer", model="HARMONIE")
            state_path.parent.mkdir(parents=True)
            state_path.write_bytes(b"last-success")
            outcome = self._invoke(
                root,
                snapshot=_snapshot(forecast_hash=FORECAST_B),
                state=_state(),
                post_cache=CachedArtifactStatus(False, None, "missing:next_csv"),
            )
            self.assertEqual(outcome["result"], 0)
            self.assertEqual(state_path.read_bytes(), b"last-success")

    def test_gate_exception_runs_full_child_without_advancing_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome = self._invoke(
                Path(directory),
                collect_error=ValueError("bad gate input"),
                state=_state(),
            )
            self.assertEqual(outcome["decisions"], ["recovery_gate_error"])
            outcome["child"].assert_called_once_with(
                self.SCRIPT,
                outcome["argv"],
                measured_only=False,
            )
            self.assertFalse(outcome["state_path"].exists())

    def test_manual_bypass_runs_full_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome = self._invoke(
                Path(directory),
                snapshot=_snapshot(),
                state=_state(),
                operational_args=False,
            )
            self.assertEqual(outcome["decisions"], ["bypass_full"])
            outcome["child"].assert_called_once_with(
                self.SCRIPT,
                outcome["argv"],
                measured_only=False,
            )
            self.assertTrue(outcome["state_path"].is_file())

    def test_non_prediction_bypass_does_not_advance_state(self) -> None:
        for flag in ("--skip-prediction", "--plots-only", "--use-existing-artifacts"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                outcome = self._invoke(
                    Path(directory),
                    snapshot=_snapshot(),
                    state=_state(),
                    extra_argv=(flag,),
                )
                self.assertEqual(outcome["decisions"], ["bypass_full"])
                outcome["child"].assert_called_once_with(
                    self.SCRIPT,
                    outcome["argv"],
                    measured_only=False,
                )
                self.assertFalse(outcome["state_path"].exists())

    def test_internal_measured_flag_outside_child_fails_safe_to_full_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome = self._invoke(
                Path(directory),
                snapshot=_snapshot(),
                state=_state(),
                extra_argv=("--operational-measured-only",),
            )
            self.assertEqual(outcome["decisions"], ["bypass_full"])
            outcome["child"].assert_called_once_with(
                self.SCRIPT,
                outcome["argv"],
                measured_only=False,
            )
            self.assertFalse(outcome["state_path"].exists())


class LauncherHelperTests(unittest.TestCase):
    def test_child_command_adds_measured_only_argument_once(self) -> None:
        script = Path("update.py")
        base = ["--skip-training"]
        self.assertEqual(
            _child_command(script, base, measured_only=True),
            [sys.executable, str(script), *base, "--operational-measured-only"],
        )
        existing = [*base, "--operational-measured-only"]
        self.assertEqual(
            _child_command(script, existing, measured_only=True),
            [sys.executable, str(script), *existing],
        )
        self.assertEqual(
            _child_command(script, base, measured_only=False),
            [sys.executable, str(script), *base],
        )

    def test_run_child_sets_gate_environment_and_uses_expected_command(self) -> None:
        completed = mock.Mock(returncode=7)
        with mock.patch("next_day_wind_model.operational_update.subprocess.run", return_value=completed) as run:
            result = _run_child(Path("update.py"), ["--flag"], measured_only=True)
        self.assertEqual(result, 7)
        run.assert_called_once()
        command = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        self.assertEqual(
            command,
            [sys.executable, "update.py", "--flag", "--operational-measured-only"],
        )
        self.assertEqual(kwargs["env"][GATE_CHILD_ENV], "1")
        self.assertFalse(kwargs["check"])

    def test_missing_model_artifacts_are_reported_without_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fingerprint, missing = compute_model_fingerprint(Path(directory))
        self.assertIsNone(fingerprint)
        self.assertGreater(len(missing), 0)


class ImportPerformanceGuardTests(unittest.TestCase):
    def test_no_change_bootstrap_does_not_import_heavy_pipeline_modules(self) -> None:
        forbidden = ("torch", "data_pipeline", "intraday_model", "train_lstm")
        attempted: list[str] = []
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == module or name.endswith(f".{module}") for module in forbidden):
                attempted.append(name)
                raise AssertionError(f"heavy import attempted before gate exit: {name}")
            return original_import(name, globals, locals, fromlist, level)

        script = Path(__file__).with_name("update_model_and_predict.py")
        with (
            mock.patch(
                "next_day_wind_model.operational_update.launch_operational_update",
                return_value=0,
            ) as gate,
            mock.patch.object(builtins, "__import__", side_effect=guarded_import),
            mock.patch.object(sys, "argv", [str(script), *OperationalLauncherTests.BASE_ARGS]),
            mock.patch.dict(os.environ, {GATE_CHILD_ENV: "0"}),
            self.assertRaises(SystemExit) as exit_context,
        ):
            runpy.run_path(str(script), run_name="__main__")
        self.assertEqual(exit_context.exception.code, 0)
        gate.assert_called_once()
        self.assertEqual(attempted, [])


if __name__ == "__main__":
    unittest.main()
