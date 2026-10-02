"""Calibration policy, bounded affine fitting and frozen replay regressions."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from next_day_wind_model import d2_calibration_study as study
from next_day_wind_model import day_after_tomorrow_core as core
from next_day_wind_model import day_after_tomorrow as d2
from next_day_wind_model import operational_update as operational
from next_day_wind_model.test_day_after_tomorrow_operational import samples


class AffineTests(unittest.TestCase):
    def test_masks_and_standardisation_exclude_unavailable_targets(self):
        pred = np.arange(80, dtype=float).reshape(8, 10) / 10
        actual = pred + 1.5
        mask = np.ones(pred.shape, bool); mask[:, 6:] = False
        a = study.fit_affine(pred, actual, mask, .1)
        b = study.fit_affine(np.where(mask, pred, 1e9), np.where(mask, actual, -1e9), mask, .1)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['pred_mean'], pred[mask].mean())
        self.assertLess(np.mean(np.abs(study.apply_affine(pred, a)[mask] - actual[mask])), .2)

    def test_regularisation_shrinks_toward_identity(self):
        pred = np.arange(1., 21.)
        actual = 1.7 * pred + 2.
        weak = study.fit_affine(pred, actual, np.ones(20, bool), .1)
        strong = study.fit_affine(pred, actual, np.ones(20, bool), 100.)
        self.assertLess(abs(strong['offset']), abs(weak['offset']))
        self.assertLess(abs(strong['coefficient']), abs(weak['coefficient']))
        np.testing.assert_array_equal(study.apply_affine(pred, None), pred)

    def test_non_negative_slope_and_output(self):
        pred = np.linspace(0, 30, 60)
        fit = study.fit_affine(pred, 30 - 3 * pred, np.ones(60, bool), .1)
        self.assertGreaterEqual(fit['slope'], 0.)
        values = study.apply_affine(pred, fit)
        self.assertTrue((values >= 0).all())
        self.assertTrue((np.diff(values) >= -1e-12).all())

    def test_partition_purges_target_observations_before_selection_issue(self):
        fitting, selection = study.calibration_partitions(samples())
        self.assertFalse({s.day for s in fitting} & {s.day for s in selection})
        self.assertTrue(all(s.label_end <= min(v.issue for v in selection) for s in fitting))
        self.assertEqual(len({s.day for s in selection}), 12)

    def test_insufficient_data_never_fits(self):
        with patch.object(study, 'fit_affine') as fit:
            result = study.select_affine(None, samples()[:20])
        self.assertEqual(result['status'], 'insufficient_data')
        fit.assert_not_called()

    def test_selection_uses_fixed_grid_and_no_correction_is_candidate(self):
        values = samples()
        with patch.object(core, 'predict', side_effect=lambda fit, rows, **kw: np.stack([s.actual for s in rows])):
            result = study.select_affine(None, values)
        self.assertEqual(result['status'], 'no_improving_candidate')
        self.assertIsNone(result['selected'])
        self.assertEqual([c['calibration']['ridge'] for c in result['candidates']], list(study.RIDGES))
        self.assertEqual(result['best_candidate']['calibration']['ridge'], 100.)

    def test_select_improving_affine_without_gate_observations(self):
        values = samples()
        with patch.object(core, 'predict', side_effect=lambda fit, rows, **kw: np.stack([s.actual - 1 for s in rows])):
            result = study.select_affine(None, values)
        self.assertEqual(result['status'], 'selected_experimental')
        self.assertLess(result['best_candidate']['selection_mae'], .99 * result['selection_uncalibrated_mae'])
        self.assertFalse(result['independent_final_test'])


class PolicyTests(unittest.TestCase):
    def test_none_bypasses_selector_and_survives_checkpoint(self):
        previous = torch.get_num_threads(); torch.set_num_threads(2)
        try:
            with patch.object(core, 'fit_speed_regime_calibration') as selector:
                fit = core.fit_model(samples(), core.utc('2026-08-20T05:00Z'), 'speed', False, 1, 16, 42,
                                     calibration_policy='none')
            selector.assert_not_called()
            self.assertIsNone(fit.calibration)
            self.assertEqual(fit.info['calibration_policy'], 'none')
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'candidate.pt'
                core.save_fit(fit, path)
                loaded = core.load_fit(path)
                self.assertIsNone(loaded.calibration)
                self.assertEqual(loaded.info['calibration_policy'], 'none')
                with patch.object(core, 'apply_speed_regime_calibration') as apply:
                    np.testing.assert_array_equal(core.predict(loaded, samples()[:2]), core.predict(loaded, samples()[:2], calibrate=False))
                    apply.assert_not_called()
        finally:
            torch.set_num_threads(previous)

    def test_core_default_preserves_historical_legacy_training(self):
        import inspect
        self.assertEqual(inspect.signature(core.fit_model).parameters['calibration_policy'].default, 'legacy')
        self.assertEqual(inspect.signature(d2.train).parameters['calibration_policy'].default, 'none')

    def test_policy_change_invalidates_operational_cache(self):
        settings = SimpleNamespace(enable_day_after_tomorrow=True, site=core.SITE, day_after_tomorrow_calibration='none')
        with tempfile.TemporaryDirectory() as temp, patch.object(operational, 'compute_model_fingerprint', return_value=('shared', ())):
            path = Path(temp)
            first = operational.operational_model_fingerprint(settings, path)
            settings.day_after_tomorrow_calibration = 'legacy'
            self.assertNotEqual(first, operational.operational_model_fingerprint(settings, path))


class FrozenTests(unittest.TestCase):
    def test_manifest_is_sealed_and_rejects_changed_models_or_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); artifact = path / 'weights'; artifact.write_bytes(b'original')
            manifest = {'artifacts': {'weights': study.sha256(artifact)}, 'required_target_dates': 20}
            study.write_manifest(path, manifest)
            self.assertEqual(study.validate_manifest(path), manifest)
            with self.assertRaises(FileExistsError):
                study.write_manifest(path, manifest)
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact was altered'):
                study.validate_manifest(path)
            artifact.write_bytes(b'original')
            sealed = path / 'frozen_manifest.json'; sealed.chmod(0o644)
            sealed.write_text(json.dumps({**manifest, 'required_target_dates': 10}))
            with self.assertRaisesRegex(ValueError, 'manifest was altered'):
                study.validate_manifest(path)

    def test_confirmation_starts_with_full_issue_window_after_freeze_and_dst(self):
        self.assertEqual(study.first_confirmation_date('2026-10-02T20:30Z').isoformat(), '2026-10-05')
        self.assertEqual(study.first_confirmation_date('2026-10-02T04:00Z').isoformat(), '2026-10-04')
        self.assertEqual(study.first_confirmation_date('2026-10-24T20:30Z').isoformat(), '2026-10-27')
        self.assertEqual(study.first_confirmation_date('2026-12-31T20:30Z').isoformat(), '2027-01-03')

    def test_insufficient_contexts_do_not_extend_fixed_period(self):
        result = study.assess_confirmation({}, {'d2': 59, 'next_day': 61}, {'minimum_issue_contexts_per_horizon': 60})
        self.assertEqual(result['status'], 'insufficient_contexts')
        self.assertFalse(result['period_extended'])

    def test_assessment_uses_gain_and_date_interval_without_promotion(self):
        def report(name, gain, interval):
            return {'matched_hours': {name: {'relative_improvement_vs_uncalibrated': gain,
                                            'vs_uncalibrated': {'ci95': interval}}}}
        reports = {'d2': report('simple', .02, [.01, .2]), 'next_day': report('legacy', .005, [-.1, .2])}
        result = study.assess_confirmation(reports, {'d2': 80, 'next_day': 80}, {'minimum_issue_contexts_per_horizon': 60})
        self.assertEqual(result['decisions']['d2']['outcome'], 'promising')
        self.assertEqual(result['decisions']['next_day']['outcome'], 'inconclusive')
        self.assertFalse(result['production_promotion'])
        self.assertEqual(result['operational_d2_calibration'], 'none')

    def test_next_day_replay_uses_issue_cutoff_and_completed_history(self):
        for stamp in ['2026-10-24T07:00:00+02:00', '2026-03-28T22:00:00+01:00']:
            issue = core.utc(stamp)
            day = issue.tz_convert(core.TZ).date() + pd.Timedelta(days=1)
            start = pd.Timestamp(str(day), tz=core.TZ)
            targets = pd.date_range(start, periods=24, freq='h').tz_convert('UTC')
            target = pd.DataFrame({'forecast_avg': 8., 'forecast_dir': 270., 'horizon_hr': np.arange(24),
                'run_ts': int(issue.timestamp()*1000)-1000, 'fetched_ts': int(issue.timestamp()*1000)-500}, index=targets)
            history = pd.DataFrame({'forecast_dir': [270.]})
            obs = pd.DataFrame({'actual_avg': 9.}, index=targets)
            scalers = {'x_mean': np.zeros(10), 'x_std': np.ones(10)}
            checkpoint = {'target_hours': 24, 'history_hours': 72, 'trained_at_utc': '2025-01-01T00:00Z'}
            with patch.object(core.dp, '_build_training_history_forecast_frame', return_value=history) as hist, \
                 patch.object(core.dp, '_select_training_complete_run_frame', return_value=target) as select, \
                 patch.object(core.dp, '_build_feature_sequence', return_value=(np.ones((96,10)), [])), \
                 patch.object(study, 'next_day_predictions', return_value=(np.full((1,24), 9.), np.full((1,24), 8.8))):
                rows = study.next_day_issue(None, obs, issue, None, checkpoint, scalers)
            self.assertEqual(hist.call_args.args[1][-1], issue-pd.Timedelta(hours=1))
            self.assertEqual(hist.call_args.args[2], int(issue.timestamp()*1000))
            self.assertEqual(select.call_args.args[2], int(issue.timestamp()*1000))
            self.assertEqual(len(rows), 15)
            hours = pd.to_datetime(rows.target_time_utc, utc=True).dt.tz_convert(core.TZ).dt.hour
            self.assertEqual(hours.tolist(), list(range(8,23)))

    def test_complete_confirmation_replays_twenty_dates_once_without_fitting(self):
        from next_day_wind_model.test_day_after_tomorrow_experiment import sample
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); experiment = root / 'experiment'; experiment.mkdir()
            source = root / 'source/weather.sqlite'; source.parent.mkdir()
            manifest = {'artifacts': {}, 'freeze_time_utc': '2026-10-02T20:30Z',
                'first_eligible_target_date': '2026-10-05', 'required_target_dates': 20,
                'minimum_issue_contexts_per_horizon': 60, 'calibrations': {'legacy': None, 'simple': None}}
            study.write_manifest(experiment, manifest)
            fit = SimpleNamespace(info={'trained_at_utc': '2026-10-02T20:00Z', 'max_label_end_utc': '2026-08-16T21:00Z'})
            obs = pd.DataFrame({'actual_avg': 9.}, index=pd.date_range('2026-10-01T00:00Z', '2026-10-26T23:00Z', freq='h'))
            def frozen_sample(issue):
                value = sample(issue.isoformat())
                value.frame['run_ts'] = int(issue.timestamp()*1000)-1000
                value.frame['fetched_ts'] = int(issue.timestamp()*1000)-500
                return value
            def d2_rows(_fit, selected, *_cal):
                result = pd.DataFrame(core.prediction_rows(selected, np.full((len(selected),15),8.5), None, 'test'))
                result['uncalibrated'] = 8.5; result['legacy'] = 8.; result['simple'] = 8.5
                return result
            def next_rows(_lookup, _obs, issue, *_model):
                targets = core.calendar_targets(issue, offset=1)
                return pd.DataFrame({'issue_time_utc': issue.isoformat(), 'target_time_utc': targets.astype(str),
                    'target_date': targets[0].tz_convert(core.TZ).date().isoformat(), 'actual': 9.,
                    'harmonie': 11., 'uncalibrated': 10., 'legacy': 9.5, 'scorable': True})
            with patch.object(core, 'snapshot', side_effect=lambda src,dst,**kw: dst.write_bytes(b'isolated snapshot')), \
                 patch.object(core, 'observations', return_value=(obs, core.utc('2026-10-26T23:00Z'))), \
                 patch.object(core.dp, 'load_training_forecast_lookup', return_value=None), \
                 patch.object(core, 'load_fit', return_value=fit), \
                 patch.object(study, 'load_next_day', return_value=(None, {'trained_at_utc':'2026-09-12T05:00Z'}, None)), \
                 patch.object(core, 'make_sample', side_effect=lambda lookup,obs,archive,issue: frozen_sample(issue)), \
                 patch.object(study, 'd2_rows', side_effect=d2_rows), \
                 patch.object(study, 'next_day_issue', side_effect=next_rows), \
                 patch.object(core, 'fit_model') as neural_fit, patch.object(study, 'fit_affine') as affine_fit:
                result = study.confirm(experiment, source, source.parent/'ec.sqlite', '2026-10-26T23:00Z')
                neural_fit.assert_not_called(); affine_fit.assert_not_called()
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['evaluation_dates']), 20)
            self.assertEqual(result['evaluation_dates'][-1], '2026-10-24')
            self.assertEqual(result['contexts'], {'d2':320, 'next_day':320})
            self.assertEqual(result['decisions']['d2']['outcome'], 'inconclusive')
            self.assertEqual(result['decisions']['next_day']['outcome'], 'promising')
            self.assertFalse(result['production_promotion'])
            with patch.object(core, 'snapshot') as snapshot:
                self.assertEqual(study.confirm(experiment, source, source.parent/'ec.sqlite', '2026-10-27T23:00Z'), result)
                snapshot.assert_not_called()
            csv = next(experiment.glob('confirmation/*/d2_predictions.csv'))
            predictions = pd.read_csv(csv)
            self.assertEqual(predictions.prediction_origin.unique().tolist(), ['replay_with_frozen_models'])
            self.assertFalse(predictions.saved_at_issuance.any())
            self.assertFalse(list(experiment.glob('confirmation/*/weather.sqlite')))

    def test_ephemeral_snapshot_cleanup_includes_wal_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / 'weather.sqlite'
            for suffix in ('', '-wal', '-shm'):
                Path(str(db)+suffix).write_bytes(b'local copy')
            study.remove_snapshot(db)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_confirmation_cannot_refit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            manifest = {'artifacts': {}, 'freeze_time_utc': '2026-10-02T20:30Z'}
            study.write_manifest(path, manifest)
            with patch.object(core, 'fit_model') as fit, patch.object(study, 'fit_affine') as affine:
                with self.assertRaisesRegex(ValueError, 'precede freezing'):
                    study.confirm(path, path/'source', path/'ec', '2026-10-02T20:00Z')
            fit.assert_not_called(); affine.assert_not_called()


if __name__ == '__main__':
    unittest.main()
