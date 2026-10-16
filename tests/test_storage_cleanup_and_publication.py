from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock

# The operational updater retains script-style sibling imports. Make these
# tests independently runnable, as pytest's next_day_wind_model collection does.
MODEL_DIR = Path(__file__).resolve().parents[1] / "next_day_wind_model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from next_day_wind_model.daily_archive import update_daily_archive
from next_day_wind_model.dashboard_publication import (
    MANIFEST, REQUIRED_FILES, build_bundle, dispatch_publication, extract_bundle,
)
from scripts.plan_runtime_cleanup import plan_cleanup


NOW = datetime(2026, 10, 16, 20, 54, tzinfo=timezone.utc)


def table(day="2026-10-16", hours=("12:00", "22:54")):
    return ("time_local,actual_wind_speed,lstm_pred_wind_speed\n" + "".join(
        f"{day}T{hour}:00+02:00,5,6\n" for hour in hours)).encode()


def public_site(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        content = b"public output"
        if name == "index.html":
            content = b'<html><img src="current_day_predictions.png?v=1"><a href="current_day_predictions.csv">CSV</a></html>'
        (root / name).write_bytes(content)


class DailyArchiveTests(unittest.TestCase):
    def test_failed_archive_render_preserves_previous_daily_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, image = root / "current.csv", root / "current.png"
            source.write_bytes(table(hours=("12:00",)))
            image.write_bytes(b"previous daily image")
            first = update_daily_archive(out_dir=root, current_csv=source, current_png=image, now_utc=NOW)
            previous_csv = Path(first["csv"]).read_bytes()
            source.write_bytes(table())
            with self.assertRaises(RuntimeError):
                update_daily_archive(out_dir=root, current_csv=source, current_png=image, now_utc=NOW,
                                     render_png=lambda path: (_ for _ in ()).throw(RuntimeError("render failed")))
            self.assertEqual(Path(first["csv"]).read_bytes(), previous_csv)
            self.assertEqual(Path(first["png"]).read_bytes(), b"previous daily image")

    def test_latest_render_replaces_one_pair_and_keeps_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "current_day_predictions.csv"
            image = root / "current_day_predictions.png"
            source.write_bytes(table(hours=("12:00",)))
            image.write_bytes(b"noon")
            update_daily_archive(out_dir=root, current_csv=source, current_png=image, now_utc=NOW)
            source.write_bytes(table())
            image.write_bytes(b"22:54 final observed plot")
            result = update_daily_archive(out_dir=root, current_csv=source, current_png=image, now_utc=NOW)
            self.assertEqual(Path(result["csv"]).read_bytes(), table())
            self.assertEqual(Path(result["png"]).read_bytes(), image.read_bytes())
            self.assertEqual(len(list((root / "current_day_csv_archive").iterdir())), 1)
            self.assertEqual(len(list((root / "current_day_plot_archive").iterdir())), 1)
            source.write_bytes(table(hours=("12:00",)))
            image.write_bytes(b"stale")
            result = update_daily_archive(out_dir=root, current_csv=source, current_png=image, now_utc=NOW)
            self.assertEqual(result["status"], "preserved_more_complete_archive")
            self.assertIn(b"22:54 final", next((root / "current_day_plot_archive").iterdir()).read_bytes())

    def test_midnight_uses_table_date_and_sites_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for site in ["valkenburgsemeer", "oostvoorne"]:
                root = Path(tmp) / site
                root.mkdir()
                source, image = root / "current.csv", root / "current.png"
                source.write_bytes(table())
                image.write_bytes(site.encode())
                result = update_daily_archive(out_dir=root, current_csv=source, current_png=image,
                                              now_utc=datetime(2026, 10, 17, 0, 1, tzinfo=timezone.utc))
                self.assertEqual(result["date"], "2026-10-16")
                self.assertEqual(Path(result["png"]).read_bytes(), site.encode())

    def test_completed_cache_is_pruned_only_after_observed_daily_pair_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "intraday_prediction_cache/20261015"
            cache.mkdir(parents=True)
            old = cache / "20261015-120000_+0200_target_20261015.csv"
            old.write_bytes(table("2026-10-15", ("12:00",)))
            unknown = cache / "rider.csv"
            unknown.write_bytes(b"protected")
            incomplete = root / "intraday_prediction_cache/20261014"
            incomplete.mkdir()
            (incomplete / "latest.csv").write_bytes(table("2026-10-14"))
            source, image = root / "current.csv", root / "current.png"
            image.write_bytes(b"plot")
            source.write_bytes(table("2026-10-15"))
            update_daily_archive(out_dir=root, current_csv=source, current_png=image, now_utc=NOW)
            self.assertFalse(old.exists())
            self.assertEqual(unknown.read_bytes(), b"protected")
            self.assertTrue((incomplete / "latest.csv").exists())


class CleanupPlanTests(unittest.TestCase):
    def test_portal_plots_rider_files_and_unfinalized_csvs_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "current_day_plot_archive"
            archive.mkdir()
            names = ["20261014-120000_current_day_predictions.png", "20261014-225400_current_day_predictions.png",
                     "current_day_predictions_2026-10-14.png", "20261016-120000_current_day_predictions.png",
                     "rider.png"]
            for name in names:
                (archive / name).write_bytes(b"plot")
            snapshots = root / "current_day_snapshots"
            snapshots.mkdir()
            snapshot = snapshots / "20261014-120000_target_20261014.csv"
            snapshot.write_bytes(table("2026-10-14"))
            rider = root / "rider_activity_analysis/17"
            rider.mkdir(parents=True)
            (rider / "map.png").write_bytes(b"private plot")
            (rider / "runs.csv").write_bytes(b"private csv")
            plan = plan_cleanup([root], today=date(2026, 10, 16))
            self.assertEqual([Path(p["path"]).name for p in plan["candidates"]], [names[0]])
            self.assertEqual(plan["blocked"][0]["path"], str(snapshot))
            self.assertEqual(len(list(archive.iterdir())), len(names))
            self.assertEqual((rider / "map.png").read_bytes(), b"private plot")

    def test_csvs_are_candidates_only_when_daily_observations_cover_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, image = root / "current.csv", root / "current.png"
            source.write_bytes(table("2026-10-15", ("12:00",)))
            image.write_bytes(b"plot")
            update_daily_archive(out_dir=root, current_csv=source, current_png=image, now_utc=NOW)
            snapshots = root / "current_day_snapshots"
            snapshots.mkdir()
            covered = snapshots / "20261015-120000_target_20261015.csv"
            covered.write_bytes(table("2026-10-15", ("12:00",)))
            uncovered = snapshots / "20261015-225400_target_20261015.csv"
            uncovered.write_bytes(table("2026-10-15"))
            plan = plan_cleanup([root], today=date(2026, 10, 16))
            self.assertEqual([p["path"] for p in plan["candidates"]], [str(covered)])
            self.assertEqual([p["path"] for p in plan["blocked"]], [str(uncovered)])

    def test_day_ahead_evaluation_retains_the_earliest_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "dayahead_snapshots"
            directory.mkdir()
            for hour in ["070000", "080000", "090000"]:
                (directory / f"20261014-{hour}_target_20261015.csv").write_text("prediction")
            plan = plan_cleanup([root], today=date(2026, 10, 16))
            self.assertEqual(len(plan["candidates"]), 2)
            self.assertIn("070000", plan["protected"][0]["path"])

    def test_symlink_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "current_day_snapshots").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                plan_cleanup([root], today=date(2026, 10, 16))


class ArtifactPublicationTests(unittest.TestCase):
    def test_pipeline_reports_failed_dispatch_as_an_error(self):
        from next_day_wind_model import update_model_and_predict as updater
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "main\n", "")), \
                 patch("next_day_wind_model.dashboard_publication.dispatch_publication", return_value={"dispatched": False}):
                with self.assertRaisesRegex(RuntimeError, "artifact publication failed"):
                    updater.auto_push_dashboard_changes(root, root / "docs", "origin", "main", publication_mode="artifact")

    def test_round_trip_includes_linked_downloads_but_never_rider_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            web, bundle, deployed = root / "docs", root / "publication/dashboard.zip", root / "deployed"
            public_site(web)
            (web / "rider_private.csv").write_bytes(b"private")
            (web / "database_backup_and_restore.md").write_bytes(b"operations")
            build_bundle(web, bundle)
            extract_bundle(bundle, deployed)
            self.assertEqual((deployed / "current_day_predictions.csv").read_bytes(), b"public output")
            self.assertFalse((deployed / "rider_private.csv").exists())
            self.assertFalse((deployed / "database_backup_and_restore.md").exists())

    def test_missing_public_asset_keeps_previous_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            web, bundle = root / "docs", root / "dashboard.zip"
            public_site(web)
            build_bundle(web, bundle)
            previous = bundle.read_bytes()
            (web / "index.html").write_text('<img src="missing.png">')
            with self.assertRaises(ValueError):
                build_bundle(web, bundle)
            self.assertEqual(bundle.read_bytes(), previous)

    def test_symlink_asset_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            web = root / "docs"
            public_site(web)
            image = web / "current_day_predictions.png"
            image.unlink()
            image.symlink_to(root / "private.png")
            with self.assertRaises(ValueError):
                build_bundle(web, root / "dashboard.zip")

    def test_tampering_or_path_traversal_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            web, bundle = root / "docs", root / "dashboard.zip"
            public_site(web)
            build_bundle(web, bundle)
            with zipfile.ZipFile(bundle) as archive:
                entries = {name: archive.read(name) for name in archive.namelist()}
            for extra in ["current_day_predictions.csv", "../private.csv"]:
                with self.subTest(extra=extra):
                    bad = root / "bad.zip"
                    with zipfile.ZipFile(bad, "w") as archive:
                        for name, content in entries.items():
                            archive.writestr(name, b"tampered" if name == extra else content)
                        if extra.startswith(".."):
                            archive.writestr(extra, b"private")
                    with self.assertRaises(ValueError):
                        extract_bundle(bad, root / "deploy")
                    self.assertFalse((root / "deploy").exists())

    def test_repeated_publication_dispatches_without_creating_git_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout
            git("init", "-b", "main")
            (root / "code.py").write_text("# application source\n")
            git("add", "code.py")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "source")
            git("remote", "add", "origin", "https://github.com/test/wind.git")
            head, count = git("rev-parse", "HEAD"), git("count-objects", "-v")
            web = root / "docs"
            public_site(web)
            calls = []
            def fake_dispatch(request, **kwargs):
                calls.append(request)
                response = MagicMock()
                response.__enter__.return_value.status = 200
                response.__enter__.return_value.read.return_value = json.dumps({
                    "workflow_run_id": 123, "html_url": "https://github.com/test/wind/actions/runs/123"}).encode()
                return response
            with patch("next_day_wind_model.dashboard_publication.urlopen", side_effect=fake_dispatch), \
                 patch.dict(os.environ, {"WIND_GITHUB_ACTIONS_TOKEN": "fixture-token"}):
                for n in range(12):
                    (web / "current_day_predictions.png").write_bytes(f"plot update {n}".encode())
                    result = dispatch_publication(root, web, "origin", "main")
                    self.assertTrue(result["dispatched"])
                    self.assertEqual(result["workflow_run_id"], 123)
            self.assertEqual(len(calls), 12)
            self.assertEqual(calls[0].full_url, "https://api.github.com/repos/test/wind/actions/workflows/deploy-dashboard.yml/dispatches")
            self.assertEqual(json.loads(calls[0].data), {"ref": "main"})
            self.assertEqual(git("rev-parse", "HEAD"), head)
            self.assertEqual(git("count-objects", "-v"), count)
            self.assertEqual(len(list((root / "next_day_wind_model/publication").iterdir())), 1)

    def test_dispatch_failure_does_not_fall_back_to_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_site(root / "docs")
            calls = []
            def failure(command, **kwargs):
                calls.append(command)
                if command[0] == "git":
                    return subprocess.CompletedProcess(command, 0, "git@github.com:test/wind.git\n", "")
                raise subprocess.CalledProcessError(1, command, stderr="sensitive message")
            with patch("subprocess.run", side_effect=failure), \
                 patch.dict(os.environ, {"WIND_GITHUB_ACTIONS_TOKEN": "fixture-token"}), \
                 patch("next_day_wind_model.dashboard_publication.urlopen", side_effect=OSError("unreachable")):
                result = dispatch_publication(root, root / "docs", "origin", "main")
            self.assertFalse(result["dispatched"])
            self.assertEqual(result["reason"], "artifact_publication_failed")
            self.assertFalse(any("commit" in command or "push" in command for command in calls))


class BoundedSnapshotTests(unittest.TestCase):
    def test_refreshes_retain_earliest_hourly_branch_and_latest_table(self):
        import pandas as pd
        from next_day_wind_model import update_model_and_predict as updater
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({"time_local": pd.to_datetime(["2026-10-16T12:00:00+02:00"]),
                                  "lstm_pred_wind_speed": [5.]})
            for minute in range(0, 60, 6):
                frame["lstm_pred_wind_speed"] = minute + 5.
                updater.save_current_day_snapshot(root, frame, "Europe/Amsterdam",
                                                   f"2026-10-16T10:{minute:02}:00Z")
            history = updater.load_current_day_prediction_history(root, date(2026, 10, 16), "Europe/Amsterdam")
            latest = updater.load_latest_current_day_snapshot(root, date(2026, 10, 16), "Europe/Amsterdam")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["lstm_pred_wind_speed"].iloc[0], 5.)
            self.assertEqual(latest["lstm_pred_wind_speed"].iloc[0], 59.)
            self.assertEqual(len(list((root / "intraday_prediction_cache/20261016").iterdir())), 2)

    def test_legacy_earlier_issue_wins_over_the_new_cache(self):
        import pandas as pd
        from next_day_wind_model import update_model_and_predict as updater
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({"time_local": pd.to_datetime(["2026-10-16T12:00:00+02:00"]),
                                  "lstm_pred_wind_speed": [8.]})
            updater.save_current_day_snapshot(root, frame, "Europe/Amsterdam", "2026-10-16T10:06:00Z")
            directory = root / "current_day_snapshots"
            directory.mkdir()
            frame["issued_at_local"] = "2026-10-16T12:00:12+02:00"
            frame["lstm_pred_wind_speed"] = 5.
            frame.to_csv(directory / "20261016-120012_target_20261016.csv", index=False)
            history = updater.load_current_day_prediction_history(root, date(2026, 10, 16), "Europe/Amsterdam")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["lstm_pred_wind_speed"].iloc[0], 5.)

    def test_dst_repeated_local_hour_has_two_distinct_forecast_branches(self):
        import pandas as pd
        from next_day_wind_model import update_model_and_predict as updater
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({"time_local": pd.to_datetime(["2026-10-25T00:00:00Z"]),
                                  "lstm_pred_wind_speed": [5.]})
            updater.save_current_day_snapshot(root, frame, "Europe/Amsterdam", "2026-10-25T00:10:00Z")
            updater.save_current_day_snapshot(root, frame, "Europe/Amsterdam", "2026-10-25T01:10:00Z")
            history = updater.load_current_day_prediction_history(root, date(2026, 10, 25), "Europe/Amsterdam")
            self.assertEqual(len(history), 2)

    def test_day_ahead_refreshes_do_not_overwrite_the_earliest_fallback(self):
        import pandas as pd
        from next_day_wind_model import update_model_and_predict as updater
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({"target_time_utc": pd.to_datetime(["2026-10-17T10:00Z"]), "hour_utc": [10],
                                  "forecast_wind_speed": [5.], "lstm_pred_wind_speed": [6.],
                                  "forecast_wind_dir_deg": [180.], "lstm_pred_wind_dir_deg": [180.]})
            first = updater.save_dayahead_snapshot(root, frame, "Europe/Amsterdam", "2026-10-16T05:00Z")
            original = Path(first).read_bytes()
            frame["lstm_pred_wind_speed"] = 99.
            updater.save_dayahead_snapshot(root, frame, "Europe/Amsterdam", "2026-10-16T06:00Z")
            self.assertEqual(Path(first).read_bytes(), original)
            self.assertEqual(len(list((root / "dayahead_snapshots").iterdir())), 1)

    def test_archived_plot_shows_night_observations_and_dst_calendar_bounds(self):
        import pandas as pd
        from next_day_wind_model import update_model_and_predict as updater
        from next_day_wind_model.measured_update import FORECAST_COLUMNS
        from matplotlib import dates as mdates
        with tempfile.TemporaryDirectory() as tmp:
            for day, following, hours in [("2026-10-15", "2026-10-16", 24),
                                           ("2026-03-29", "2026-03-30", 23),
                                           ("2026-10-25", "2026-10-26", 25)]:
                times = pd.date_range(day, following, freq="h", inclusive="left", tz="Europe/Amsterdam")
                frame = pd.DataFrame({"time_local": times})
                for column in FORECAST_COLUMNS:
                    frame[column] = 180. if "dir" in column else 6.
                frame["actual_wind_speed"] = 5.
                frame["actual_wind_min"] = 4.
                frame["actual_wind_max"] = 8.
                frame["actual_wind_dir_deg"] = 180.
                frame["is_forecast_grid"] = True
                frame["is_actual_observation"] = True
                frame["is_future"] = False
                diagnostics = {}
                updater.save_current_day_plot(frame, Path(tmp) / f"{day}.png", "Europe/Amsterdam",
                    prediction_generated_at_utc=None, prediction_updated_at_utc=None,
                    model_trained_at_utc=None, full_local_day=True, render_diagnostics=diagnostics)
                left, right = map(mdates.num2date, diagnostics["x_limits"])
                self.assertEqual((right-left).total_seconds(), hours*3600)
                self.assertEqual(pd.Timestamp(left).tz_convert("Europe/Amsterdam").strftime("%Y-%m-%d %H:%M"), day + " 00:00")
                measured = diagnostics["series"]["measured_wind"]
                self.assertEqual(len(measured), len(frame))
                self.assertEqual(measured["time_local"].min(), times[0])
                self.assertEqual(measured["time_local"].max(), times[-1])


class PortalArchiveCompatibilityTests(unittest.TestCase):
    def test_site_specific_daily_archive_and_legacy_urls_both_work(self):
        from next_day_wind_model.web_dashboard import app as portal
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy"
            legacy.mkdir()
            name = "current_day_predictions_2026-10-15.png"
            (legacy / name).write_bytes(b"legacy portal plot")
            site_dir = root / "artifacts/valkenburgsemeer/current_day_plot_archive"
            site_dir.mkdir(parents=True)
            (site_dir / name).write_bytes(b"last site plot with observations")
            with patch.object(portal, "ARTIFACTS_DIR", root / "artifacts"), \
                 patch.object(portal, "CURRENT_DAY_PLOT_ARCHIVE_DIR", legacy):
                with portal.app.test_request_context():
                    archive = portal._current_day_archive_plot_for_submission("2026-10-15", "Valkenburgse meer")
                    missing = portal._current_day_archive_plot_for_submission("2026-10-15", "Oostvoornse meer")
                self.assertIn("/valkenburgsemeer/", archive["url"])
                self.assertIsNone(missing)
                client = portal.app.test_client()
                for url, content in [(archive["url"], b"last site plot with observations"),
                                     (f"/current-day-plot-archive/{name}", b"legacy portal plot")]:
                    with client.get(url) as response:
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.data, content)
                with client.get(f"/current-day-plot-archive/unknown/{name}") as response:
                    self.assertEqual(response.status_code, 404)

    def test_only_the_prebuilt_public_bundle_is_exposed(self):
        from next_day_wind_model.web_dashboard import app as portal
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_site(root / "web")
            bundle = root / "dashboard.zip"
            build_bundle(root / "web", bundle)
            with patch.dict(portal.app.config, {"DASHBOARD_BUNDLE_PATH": str(bundle)}):
                client = portal.app.test_client()
                with client.get("/forecast-publication/dashboard.zip") as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                        self.assertIn(MANIFEST, archive.namelist())
                with client.get("/forecast-publication/../../data/wind_data_all_sites.db") as response:
                    self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
