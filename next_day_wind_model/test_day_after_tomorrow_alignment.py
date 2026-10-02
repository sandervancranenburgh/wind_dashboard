"""Regression checks for shared masked calibration and website presentation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from next_day_wind_model import update_model_and_predict as updater
from next_day_wind_model import day_after_tomorrow_core as core
from next_day_wind_model.test_day_after_tomorrow_experiment import sample


def calibration_data():
    rng = np.random.default_rng(13)
    forecast = rng.uniform(1., 3., (40, 15)).astype(np.float32)
    pred = forecast + 1.
    actual = forecast + .4
    times = np.stack([pd.date_range(f"2026-07-{i % 20 + 1:02d}T06:00Z", periods=15, freq="h").astype(str) for i in range(40)])
    context = {"anchor_dir_deg": np.arange(40) * 9., "target_month": np.full(40, 7),
        "target_forecast_dir_deg": np.tile(np.arange(15) * 24., (40, 1)), "target_times_utc": times,
        "target_horizon_hr": np.tile(np.arange(15) + 40., (40, 1))}
    return pred, forecast, actual, context


class CalibrationTests(unittest.TestCase):
    def test_direction_bootstrap_uses_circular_errors(self):
        frame = pd.DataFrame({"target_date": ["2026-09-01", "2026-09-02"],
            "actual_direction": [0., 0.], "harmonie_direction": [358., 358.], "prediction_direction": [359., 1.]})
        result = core.bootstrap(frame, "prediction_direction", "harmonie_direction", actual_column="actual_direction", circular=True)
        self.assertEqual(result["estimate"], 1.)
        self.assertEqual(result["ci95"], [1., 1.])

    def test_complete_window_mask_preserves_next_day_selection(self):
        pred, forecast, actual, context = calibration_data()
        original = updater.fit_speed_regime_calibration(pred, forecast, actual, context)
        masked = updater.fit_speed_regime_calibration(pred, forecast, actual, context, target_mask=np.ones_like(pred, bool))
        self.assertIsNotNone(original)
        self.assertEqual(original, masked)
        np.testing.assert_array_equal(updater.apply_speed_regime_calibration(pred, forecast, original, context),
            updater.apply_speed_regime_calibration(pred, forecast, masked, context, target_mask=np.ones_like(pred, bool)))

    def test_unavailable_values_cannot_influence_fitting_or_window_signals(self):
        pred, forecast, actual, context = calibration_data()
        mask = np.ones_like(pred, bool)
        mask[:, 7:] = False
        poisoned = [np.where(mask, values, 1e8) for values in (pred, forecast, actual)]
        absent = [np.where(mask, values, np.nan) for values in (pred, forecast, actual)]
        a = updater.fit_speed_regime_calibration(*poisoned, context, target_mask=mask)
        b = updater.fit_speed_regime_calibration(*absent, context, target_mask=mask)
        self.assertIsNotNone(a)
        self.assertEqual(a, b)
        signal = updater._extract_speed_regime_signal(poisoned[0], poisoned[1], "pred_max", mask)
        np.testing.assert_array_equal(signal, pred[:, :7].max(axis=1))
        self.assertEqual(signal.shape, (40,))

    def test_selector_evaluates_all_three_methods_with_windows_intact(self):
        pred, forecast, actual, context = calibration_data()
        mask = np.ones_like(pred, bool)
        mask[:, 8:] = False
        with patch.object(updater, "_fit_threshold_speed_calibration", return_value={"type": "threshold_v1", "calibrated_mae": .3}) as threshold, \
             patch.object(updater, "_fit_contextual_speed_calibration", return_value={"type": "contextual_linear_v2", "calibrated_mae": .2}) as contextual, \
             patch.object(updater, "_fit_target_hour_speed_calibration", return_value={"type": "target_hour_ridge_v1", "calibrated_mae": .4}) as ridge:
            diagnostics = {}
            selected = updater.fit_speed_regime_calibration(pred, forecast, actual, context, target_mask=mask, diagnostics=diagnostics)
            self.assertEqual(selected["type"], "contextual_linear_v2")
            for method in [threshold, contextual, ridge]:
                self.assertEqual(method.call_count, 1)
                self.assertEqual(method.call_args.args[0].shape, (40, 15))
            self.assertEqual(set(diagnostics["candidates"]), {"threshold_v1", "contextual_linear_v2", "target_hour_ridge_v1"})

    def test_d2_calibration_context_has_anchor_and_target_features(self):
        s = sample("2026-09-30T22:00:00+02:00")
        s.speed_x[71, 2:4] = [1., 0.]
        context = core.calibration_context([s])
        self.assertAlmostEqual(context["anchor_dir_deg"][0], 90.)
        self.assertEqual(context["target_month"].tolist(), [10])
        self.assertEqual(context["target_times_utc"].shape, (1, 15))


class PresentationTests(unittest.TestCase):
    def test_operational_plot_matches_next_day_template(self):
        s = sample("2026-10-02T15:00:00+02:00")
        values = np.full(15, 5., np.float32)
        table = updater.build_prediction_table({"target_times": s.targets.astype(str).to_numpy(),
            "forecast_next24": values, "forecast_min_next24": values, "forecast_max_next24": values,
            "forecast_dir_next24": values}, values, values)
        with tempfile.TemporaryDirectory() as temp:
            for mobile in [False, True]:
                diagnostics = []
                for experimental in [False, True]:
                    diagnostic = {}
                    updater.save_prediction_plot(table, Path(temp) / f"{mobile}_{experimental}.png", core.TZ,
                        s.issue, s.issue.isoformat(), s.issue.isoformat(), mobile=mobile,
                        experiment_label="Experimental D+2" if experimental else None,
                        operational_forecast=experimental, render_diagnostics=diagnostic)
                    diagnostics.append(diagnostic)
                for key in ["axis_roles", "axes_positions", "x_limits", "y_limits", "y_ticks", "x_tick_labels", "figure_size_inches", "legend_labels", "plot_meta_text"]:
                    self.assertEqual(diagnostics[0][key], diagnostics[1][key], key)

    def test_evaluation_order_and_shared_forecast_markup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "plot.png"
            image.write_bytes(b"image fixture")
            csv = root / "forecast.csv"
            pd.DataFrame({"target_time_local": ["2026-10-03T08:00:00+02:00"]}).to_csv(csv, index=False)
            d2_assets = {f"day_after_tomorrow_{name}.png": image for name in ["predictions", "predictions_mobile", "direction_spider", "model_gate_eval_history"]}
            state = {"status": "available", "issue_time_utc": "2026-10-02T15:00:00Z", "target_date": "2026-10-04", "available_hours": 15}
            published = updater.publish_web_dashboard(web_out_dir=root / "web", local_tz=core.TZ, web_refresh_seconds=360,
                next_day_png=image, next_day_png_mobile=image, next_day_csv=csv,
                current_day_png=image, current_day_png_mobile=image, current_day_csv=root / "absent.csv",
                daily_mae_png=image, daily_mae_png_mobile=image, daily_mae_csv=None,
                gate_eval_png=image, gate_eval_csv=None, direction_spider_png=image, direction_spider_csv=None,
                current_day_direction_spider_png=image, current_day_direction_spider_csv=None,
                current_day_gate_eval_png=image, spot_name="Valkenburgse Meer", day_after_tomorrow_assets=d2_assets, day_after_tomorrow_state=state)
            evaluation = (root / "web/evaluation.html").read_text()
            titles = ["Current-day performance by wind direction", "Next-day performance by wind direction",
                "Day-after-tomorrow performance by wind direction", "Current-day model-gate evaluation history",
                "Next-day model-gate evaluation history", "Day-after-tomorrow model-gate evaluation history", "Realised forecast MAE history"]
            positions = [evaluation.index(title) for title in titles]
            self.assertEqual(positions, sorted(positions))
            index = (root / "web/index.html").read_text()
            self.assertIn('data-plot-id="day-after-tomorrow-interactive-plot"', index)
            self.assertIn('id="day-after-tomorrow-fallback"', index)
            self.assertNotIn("day_after_tomorrow_interactive.js", index)
            self.assertNotIn("Show full forecast plot", index)
            self.assertIn("HARMONIE available for 15 of 15 forecast hours", index)
            self.assertIn("current_day_model_gate_eval_history.png", published)

    def test_current_day_gate_uses_actual_aligned_rows_and_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "20261002-052151_intraday_model_gate_eval_speed.csv"
            pd.DataFrame({"anchor_time_utc": ["2026-09-01T07:00Z", "2026-09-01T08:00Z"],
                "target_time_utc": ["2026-09-01T09:00Z"] * 2, "actual_value": [5., 5.],
                "harmonie_value": [7., 9.], "challenger_prediction_value": [6., 7.], "champion_prediction_value": [5., 6.]}).to_csv(source, index=False)
            gate = {"enabled": True, "intraday_model_id_champion": "actual-champion", "intraday_model_id_challenger": "actual-challenger"}
            with patch.object(updater, "save_model_gate_eval_history_plot") as render:
                result = updater.current_day_gate_assets(root, source, gate)
                updater.current_day_gate_assets(root, source, gate)
            details = pd.read_csv(result["current_day_gate_eval_details_csv"])
            self.assertEqual(details.forecast_wind_speed.tolist(), [8.])
            self.assertEqual(details.champion_wind_speed.tolist(), [5.5])
            history = pd.read_csv(result["current_day_gate_eval_csv"])
            self.assertEqual(len(history), 1)
            self.assertEqual(history.speed_model_id_champion.iloc[0], "actual-champion")
            self.assertEqual(history.speed_eval_rows.iloc[0], 2)
            self.assertEqual(render.call_args.kwargs["horizon_label"], "Current-day")


if __name__ == "__main__":
    unittest.main()
