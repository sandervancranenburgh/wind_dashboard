from __future__ import annotations

import inspect
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from next_day_wind_model import day_after_tomorrow_experiment as d2


def sample(issue: str, available: int = 15) -> d2.Sample:
    cutoff = d2.utc(issue)
    targets = d2.calendar_targets(cutoff)
    mask = np.arange(15) < available
    frame = pd.DataFrame({"forecast_avg": np.where(mask, 8., np.nan),
                          "forecast_dir": np.where(mask, 359., np.nan),
                          "horizon_hr": (targets - cutoff).total_seconds() / 3600}, index=targets)
    return d2.Sample(cutoff, targets, np.ones((87, 11), np.float32), np.ones((72, 6), np.float32),
                     frame, np.full(15, 9., np.float32), np.full(15, 1., np.float32),
                     mask, mask, mask, np.ones((15, 4), np.float32), {})


class DirectionPerformanceTests(unittest.TestCase):
    def test_target_averaging_circular_sectors_and_matched_masks(self):
        # Opposite speed errors cancel when overlapping issue predictions are
        # averaged, as in the next-day gate; 359/1 degrees must remain north.
        frame = pd.DataFrame({
            "target_time_utc": ["2026-09-01T08:00Z"] * 2 + ["2026-09-01T09:00Z", "2026-09-01T10:00Z", "2026-09-01T11:00Z", "2026-09-01T12:00Z"],
            "scorable": [True, True, True, True, False, True],
            "actual": [10.] * 6,
            "harmonie": [12., 8., 13., 12., 100., 100.],
            "prediction": [11., 9., 11., 10., 100., np.nan],
            "harmonie_direction": [359., 1., 22.5, 90., 180., 270.],
        })
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            summary = d2.export_direction_performance(frame, output).set_index("sector")
            self.assertEqual(summary.index.tolist(), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
            self.assertEqual(summary.n_points.sum(), 3)
            self.assertEqual(summary.loc["N", "forecast_mae"], 0.)
            self.assertEqual(summary.loc["N", "champion_mae"], 0.)
            self.assertEqual(summary.loc["NE", "forecast_mae"], 3.)
            self.assertEqual(summary.loc["NE", "champion_mae"], 1.)
            self.assertTrue(pd.isna(summary.loc["S", "champion_mae"]))
            details = pd.read_csv(output / "day_after_tomorrow_direction_eval_details.csv")
            self.assertEqual(details.n_overlaps.sum(), 4)
            self.assertTrue((output / "dashboard/day_after_tomorrow_direction_spider.png").exists())
            # A sparse evaluation cannot retain a previous, misleading image.
            d2.export_direction_performance(frame.iloc[:2], output)
            self.assertFalse((output / "day_after_tomorrow_direction_spider.png").exists())
            self.assertFalse((output / "dashboard/day_after_tomorrow_direction_spider.png").exists())

    def test_next_day_renderer_defaults_still_match(self):
        signature = inspect.signature(d2.save_wind_direction_performance_spider_plot)
        self.assertEqual(signature.parameters["model_label"].default, "Super local champion model next-day")
        self.assertEqual(signature.parameters["title"].default, "MAE for next-day models by forecast wind direction")


class CalendarAndPurgingTests(unittest.TestCase):
    def test_calendar_day_and_dst(self):
        for issue, target in [("2026-10-24T07:00:00+02:00", "2026-10-26T08:00:00+01:00"),
                              ("2026-03-28T07:00:00+01:00", "2026-03-30T08:00:00+02:00"),
                              ("2026-12-31T22:00:00+01:00", "2027-01-02T08:00:00+01:00")]:
            times = d2.calendar_targets(d2.utc(issue))
            self.assertEqual(times[0].tz_convert(d2.TZ), pd.Timestamp(target))
            self.assertEqual(len(times), 15)
            self.assertEqual(times[-1].tz_convert(d2.TZ).hour, 22)

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            d2.utc("2026-10-01T09:00:00")

    def test_purge_targets_across_validation_and_fit_cutoffs(self):
        samples = [sample(f"2026-09-{day:02d}T07:00:00+02:00") for day in range(1, 22)]
        cutoff = d2.utc("2026-09-20T07:00:00+02:00")
        train, val = d2.purged_split(samples, cutoff)
        self.assertTrue(all(s.label_end <= cutoff for s in train + val))
        self.assertTrue(all(s.label_end <= min(v.issue for v in val) for s in train))
        self.assertFalse({s.day for s in train} & {s.day for s in val})
        self.assertNotIn(samples[-1], val)

    def test_direction_wrap_target(self):
        targets, mask = d2.target_array([sample("2026-09-01T07:00:00+02:00")], "direction")
        np.testing.assert_allclose(targets[mask], 2.)
        self.assertEqual(d2.metric([359.], [1.], circular=True)["mae"], 2.)


class VintageAndMaskTests(unittest.TestCase):
    def test_newest_partial_run_keeps_gaps_and_excludes_future_run(self):
        issue = d2.utc("2026-09-01T07:00:00+02:00")
        targets = d2.calendar_targets(issue)
        old = int((issue - pd.Timedelta(hours=4)).value // 1_000_000)
        new = int((issue - pd.Timedelta(hours=1)).value // 1_000_000)
        future = int((issue + pd.Timedelta(hours=1)).value // 1_000_000)
        stored_ms = np.array([int(ts.timestamp() * 1000) for ts in targets], dtype=np.int64)
        targets_ms = np.concatenate([stored_ms, stored_ms[:9], stored_ms])
        run_ts = np.r_[np.full(15, old), np.full(9, new), np.full(15, future)]
        values = np.r_[np.full(15, 5.), np.full(9, 8.), np.full(15, 20.)]
        lookup = SimpleNamespace(run_ts=run_ts, fetched_ts=run_ts, target_ts=targets_ms,
                                 float_columns={"forecast_avg": values},
                                 run_starts=np.array([0, 15, 24]), run_ends=np.array([15, 24, 39]),
                                 run_available_ts=np.array([old, new, future]))
        frame = d2.partial_run(lookup, targets, issue)
        np.testing.assert_allclose(frame.forecast_avg.iloc[:9], 8.)
        self.assertTrue(frame.forecast_avg.iloc[9:].isna().all())
        self.assertTrue((frame.run_ts.dropna() == new).all())

    def test_run_completion_excludes_partially_fetched_run(self):
        issue = d2.utc("2026-09-01T07:00:00+02:00")
        ms = int(issue.value // 1_000_000)
        targets = d2.calendar_targets(issue)
        lookup = SimpleNamespace(run_ts=np.array([ms - 1000]), fetched_ts=np.array([ms - 1000]),
                                 target_ts=np.array([int(targets[0].timestamp() * 1000)]),
                                 float_columns={"forecast_avg": np.array([10.])},
                                 run_starts=np.array([0]), run_ends=np.array([1]),
                                 run_available_ts=np.array([ms + 1000]))
        self.assertTrue(d2.partial_run(lookup, targets, issue).forecast_avg.isna().all())

    def test_mask_excludes_nan_labels_and_gradients(self):
        prediction = torch.tensor([1., 100.], requires_grad=True)
        loss = d2.masked_mse(prediction, torch.tensor([3., float("nan")]), torch.tensor([True, False]))
        self.assertEqual(float(loss.detach()), 4.)
        loss.backward()
        self.assertEqual(prediction.grad[1], 0.)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_target_padding_not_supervision(self):
        values, mask = d2.target_array([sample("2026-09-01T07:00:00+02:00", 9)], "speed")
        self.assertEqual(int(mask.sum()), 9)
        self.assertTrue((values[~mask] == 0).all())

    def test_target_date_bootstrap_counts_days_not_issues(self):
        frame = pd.DataFrame({"target_date": ["2026-09-03"] * 100 + ["2026-09-04"] * 100,
                              "prediction": [9.] * 200, "harmonie": [8.] * 200, "actual": [10.] * 200})
        result = d2.bootstrap(frame, "prediction", "harmonie", iterations=100)
        self.assertEqual(result["days"], 2)
        self.assertEqual(result["ci95"], [1., 1.])

    def test_ecmwf_metrics_compare_identical_available_hours(self):
        frame = pd.DataFrame({"target_date": ["2026-09-03"] * 3,
            "prediction": [1., 10., 10.], "harmonie": [2., 12., 12.],
            "ecmwf": [2., np.nan, np.nan], "actual": [1., 10., 10.],
            "prediction_direction": [0.] * 3, "harmonie_direction": [0.] * 3,
            "actual_direction": [0.] * 3, "scorable": [True] * 3, "complete_window": [False] * 3})
        summary = d2.summarize(frame)
        paired = summary["ecmwf_matched_subset"]
        self.assertEqual(summary["prediction"]["count"], 3)
        self.assertEqual({paired[key]["count"] for key in ["prediction", "harmonie", "ecmwf"]}, {1})
        self.assertEqual(paired["harmonie"]["mae"], 1.)
        self.assertEqual(paired["ecmwf"]["mae"], 1.)


class ECMWFTests(unittest.TestCase):
    def archive(self):
        archive = d2.ECMWFArchive(None)
        points = pd.DataFrame({"fetched_time": [d2.utc("2026-09-01T01:00:00Z")] * 3,
                               "u10_mps": [3., 0., -3.], "v10_mps": [0., 3., 0.],
                               "wind_gust_mps": [6., 9., 6.]},
                              index=pd.date_range("2026-09-03T06:00:00Z", periods=3, freq="3h"))
        archive.runs = [(d2.utc("2026-09-01T00:00:00Z"), d2.utc("2026-09-01T02:00:00Z"), points)]
        return archive

    def test_component_interpolation_units_direction_and_no_extrapolation(self):
        archive = self.archive()
        targets = pd.to_datetime(["2026-09-03T05:00:00Z", "2026-09-03T07:00:00Z", "2026-09-03T13:00:00Z"], utc=True)
        values, metadata = archive.features(d2.utc("2026-09-01T07:00:00Z"), targets)
        self.assertTrue(np.isnan(values[[0, 2]]).all())
        np.testing.assert_allclose(values[1], np.array([2., 1., np.sqrt(5), 7.]) * d2.MPS_TO_KNOT, rtol=1e-6)
        self.assertAlmostEqual(metadata["direction_deg"][1], 243.4349488, places=4)

    def test_completed_and_fetched_cutoffs(self):
        archive = self.archive()
        target = pd.to_datetime(["2026-09-03T06:00:00Z"], utc=True)
        values, _ = archive.features(d2.utc("2026-09-01T01:30:00Z"), target)
        self.assertTrue(np.isnan(values).all())
        archive.runs[0][2].loc[:, "fetched_time"] = d2.utc("2026-09-01T08:00:00Z")
        values, _ = archive.features(d2.utc("2026-09-01T07:00:00Z"), target)
        self.assertTrue(np.isnan(values).all())

    def test_missing_native_point_not_bridged(self):
        archive = self.archive()
        archive.runs[0] = (*archive.runs[0][:2], archive.runs[0][2].iloc[[0, 2]])
        values, _ = archive.features(d2.utc("2026-09-01T07:00:00Z"), pd.to_datetime(["2026-09-03T08:00:00Z"], utc=True))
        self.assertTrue(np.isnan(values).all())


class SafetyAndCompatibilityTests(unittest.TestCase):
    def test_numeric_cache_round_trip_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.npz"
            original = sample("2026-09-01T07:00:00+02:00", 9)
            d2.save_samples([original], path)
            restored = d2.load_samples(path)[0]
            pd.testing.assert_frame_equal(original.frame, restored.frame)
            np.testing.assert_array_equal(original.forecast_mask, restored.forecast_mask)
            self.assertEqual(original.issue, restored.issue)

    def test_output_rejects_source_runtime_and_main_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                d2.validate_output(root / "data" / "results", root / "data" / "live.sqlite", root / "ec" / "archive.sqlite", None)

    def test_snapshot_captures_wal_and_source_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source.sqlite", Path(tmp) / "snapshot.sqlite"
            writer = sqlite3.connect(source)
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("CREATE TABLE example (value INTEGER)")
            writer.execute("INSERT INTO example VALUES (42)")
            writer.commit()
            d2.snapshot(source, dest)
            with d2.read_only(dest) as conn:
                self.assertEqual(conn.execute("SELECT value FROM example").fetchone()[0], 42)
            with d2.read_only(source) as conn:
                with self.assertRaises(sqlite3.OperationalError):
                    conn.execute("INSERT INTO example VALUES (99)")
            writer.close()

    def test_snapshot_retains_only_needed_weather_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source.sqlite", Path(tmp) / "snapshot.sqlite"
            with sqlite3.connect(source) as conn:
                conn.executescript("CREATE TABLE forecasts(value); CREATE TABLE observations(value); CREATE TABLE users(secret);")
                conn.execute("INSERT INTO users VALUES ('private')")
            d2.snapshot(source, dest, tables=("forecasts", "observations"))
            with d2.read_only(dest) as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertEqual(tables, {"forecasts", "observations"})
            with d2.read_only(source) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM users").fetchone()[0], 1)

    def test_existing_next_day_defaults_unchanged(self):
        self.assertEqual(dp_config := d2.dp.DatasetConfig().target_hours, 24)
        self.assertEqual(inspect.signature(d2.save_prediction_plot).parameters["experiment_label"].default, None)
        model = d2.TargetAwareNextDayLSTM(10, dp_config, 72)
        self.assertEqual(tuple(model(torch.zeros(2, 96, 10)).shape), (2, 24))

    def test_training_and_scalers_ignore_future_samples(self):
        torch.set_num_threads(1)
        samples = [sample(f"2026-09-{day:02d}T07:00:00+02:00") for day in range(1, 18)]
        cutoff = d2.utc("2026-09-16T07:00:00+02:00")
        first = d2.fit_model(samples, cutoff, "speed", False, 1, 16, 42)
        for s in samples:
            if s.label_end > cutoff:
                s.speed_x[:] = 1e6
                s.actual[:] = 1e6
        second = d2.fit_model(samples, cutoff, "speed", False, 1, 16, 42)
        np.testing.assert_array_equal(first.x_mean, second.x_mean)
        self.assertEqual(first.y_mean, second.y_mean)
        self.assertLessEqual(d2.utc(first.info["max_label_end_utc"]), cutoff)
        np.testing.assert_array_equal(d2.predict(first, samples[:1]), d2.predict(second, samples[:1]))

    def test_checkpoint_round_trip_without_unsafe_deserialization(self):
        torch.set_num_threads(1)
        samples = [sample(f"2026-09-{day:02d}T07:00:00+02:00") for day in range(1, 18)]
        fit = d2.fit_model(samples, d2.utc("2026-09-16T07:00:00+02:00"), "speed", False, 1, 16, 42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            d2.save_fit(fit, path)
            restored = d2.load_fit(path)
        np.testing.assert_array_equal(d2.predict(fit, samples[:1]), d2.predict(restored, samples[:1]))

    def test_speed_and_direction_use_existing_next_day_inverse(self):
        torch.set_num_threads(1)
        samples = [sample(f"2026-09-{day:02d}T07:00:00+02:00") for day in range(1, 18)]
        cutoff = d2.utc("2026-09-16T07:00:00+02:00")
        for kind in ["speed", "direction"]:
            fit = d2.fit_model(samples, cutoff, kind, False, 1, 16, 42)
            subset = samples[:2]
            x = (d2.input_array(subset, kind, False) - fit.x_mean) / fit.x_std
            if kind == "speed":
                baseline = np.stack([s.frame.forecast_avg.to_numpy(np.float32) for s in subset])
                expected = d2._predict_speed_batch(fit.model, x, baseline, fit.y_mean, fit.y_std,
                    "constrained_logratio", d2.EPS, None, None, torch.device("cpu"))
            else:
                baseline = np.stack([s.frame.forecast_dir.to_numpy(np.float32) for s in subset])
                expected = d2._predict_direction_batch(fit.model, x, baseline, fit.y_mean, fit.y_std, torch.device("cpu"))
            np.testing.assert_array_equal(d2.predict(fit, subset, calibrate=False), expected)

    def test_experimental_render_keeps_evening_gaps_and_no_promotion_label(self):
        original = sample("2026-09-01T07:00:00+02:00", 9)
        inference = {"target_times": original.targets.astype(str).to_numpy(),
                     "forecast_next24": original.frame.forecast_avg.to_numpy(np.float32),
                     "forecast_min_next24": original.frame.forecast_avg.to_numpy(np.float32),
                     "forecast_max_next24": original.frame.forecast_avg.to_numpy(np.float32),
                     "forecast_dir_next24": original.frame.forecast_dir.to_numpy(np.float32)}
        values = np.where(original.forecast_mask, 9., np.nan).astype(np.float32)
        direction = np.where(original.forecast_mask, 1., np.nan).astype(np.float32)
        table = d2.build_prediction_table(inference, values, direction, d2.TZ)
        ec = pd.DataFrame({"time_utc": pd.date_range(original.targets[0] - pd.Timedelta(hours=2), periods=7, freq="3h"),
                           "wind_speed_knots": np.full(7, 8.)})
        with tempfile.TemporaryDirectory() as tmp:
            for mobile in [False, True]:
                diagnostic = {}
                path = Path(tmp) / f"plot_{mobile}.png"
                d2.save_prediction_plot(table, path, d2.TZ, original.issue, original.issue.isoformat(),
                    original.issue.isoformat(), mobile=mobile, ecmwf_speed_series=ec,
                    experiment_label="Experimental D+2 · coverage 9/15 hours", render_diagnostics=diagnostic)
                self.assertTrue(path.is_file())
                self.assertEqual(diagnostic["direction_arrow_count"], 18)
                self.assertNotIn("promoted", diagnostic["plot_meta_text"])
                self.assertIn("Forecast issued", diagnostic["plot_meta_text"])
                self.assertLess(diagnostic["ecmwf_x_data"][0], 0)
        self.assertTrue(table.lstm_pred_wind_speed.iloc[9:].isna().all())


if __name__ == "__main__":
    unittest.main()
