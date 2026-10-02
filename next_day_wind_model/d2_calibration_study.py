"""Read-only-source calibration research; never promotes operational models.

The frozen confirmation is a fixed-date, cutoff-filtered replay. Its manifest
is sealed separately so accidental changes to models or rules fail closed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from next_day_wind_model import day_after_tomorrow_core as core
from next_day_wind_model import day_after_tomorrow as d2
from next_day_wind_model import update_model_and_predict as shared

RIDGES = (0.1, 1., 10., 100.)
VARIANTS = ('harmonie', 'uncalibrated', 'legacy', 'simple')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def calibration_partitions(samples):
    dates = sorted({s.day for s in samples})
    boundary_index = int(len(dates) * .7)
    selected_dates = set(dates[boundary_index:])
    selection = [s for s in samples if s.day in selected_dates]
    if not selection:
        return [], []
    boundary = min(s.issue for s in selection)
    fitting = [s for s in samples if s.day not in selected_dates and s.label_end <= boundary]
    return fitting, selection


def fit_affine(pred, actual, mask, ridge):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    mask = np.asarray(mask, bool) & np.isfinite(pred) & np.isfinite(actual)
    if pred.shape != actual.shape or pred.shape != mask.shape:
        raise ValueError('Affine calibration arrays must have identical shapes')
    if not mask.any():
        raise ValueError('No matched calibration targets')
    p, y = pred[mask], actual[mask]
    mean, std = float(p.mean()), float(p.std()) or 1.
    x = np.column_stack([np.ones(len(p)), (p - mean) / std])
    # Mean loss makes the ridge grid independent of repeated-hour sample count.
    matrix = x.T @ x / len(x) + float(ridge) * np.eye(2)
    rhs = x.T @ (y - p) / len(x)
    coefficients = np.linalg.solve(matrix, rhs)
    # p + a + c*(p-mean)/std has slope 1+c/std. At the
    # constraint boundary, refit the free intercept rather than clipping it.
    if coefficients[1] < -std:
        coefficients[1] = -std
        coefficients[0] = (rhs[0] - matrix[0, 1] * coefficients[1]) / matrix[0, 0]
    return {'type': 'affine_ridge_v1', 'enabled': True, 'ridge': float(ridge),
            'pred_mean': mean, 'pred_std': std, 'offset': float(coefficients[0]),
            'coefficient': float(coefficients[1]), 'slope': float(1 + coefficients[1] / std),
            'fitting_points': int(mask.sum()), 'seasonal_terms': False}


def apply_affine(pred, calibration):
    pred = np.asarray(pred, float)
    if not calibration or not calibration.get('enabled'):
        return pred.copy()
    return np.maximum(0., pred + calibration['offset'] + calibration['coefficient']
                      * (pred - calibration['pred_mean']) / calibration['pred_std'])


def select_affine(fit, validation):
    fitting, selection = calibration_partitions(validation)
    result = {'status': 'insufficient_data', 'selected': None,
              'fitting_dates': sorted({s.day for s in fitting}),
              'selection_dates': sorted({s.day for s in selection}),
              'fitting_contexts': len(fitting), 'selection_contexts': len(selection),
              'neural_validation_reused_for_early_stopping': True,
              'independent_final_test': False, 'ridge_grid': list(RIDGES), 'candidates': []}
    if (len(result['fitting_dates']) < 7 or len(result['selection_dates']) < 5
            or len(fitting) < 32 or len(selection) < 32):
        return result
    fitting_pred = core.predict(fit, fitting, calibrate=False)
    fitting_actual = np.stack([s.actual for s in fitting])
    fitting_mask = np.stack([s.speed_mask for s in fitting])
    selected_pred = core.predict(fit, selection, calibrate=False)
    selected_actual = np.stack([s.actual for s in selection])
    selected_mask = np.stack([s.speed_mask for s in selection])
    baseline = core.metric(selected_pred[selected_mask], selected_actual[selected_mask])['mae']
    result['selection_uncalibrated_mae'] = baseline
    for ridge in RIDGES:
        cal = fit_affine(fitting_pred, fitting_actual, fitting_mask, ridge)
        score = core.metric(apply_affine(selected_pred, cal)[selected_mask], selected_actual[selected_mask])
        result['candidates'].append({'calibration': cal, 'selection_mae': score['mae']})
    best_mae = min(c['selection_mae'] for c in result['candidates'])
    best = max((c for c in result['candidates'] if c['selection_mae'] <= best_mae + 1e-9),
               key=lambda c: c['calibration']['ridge'])
    result['status'] = 'no_improving_candidate'
    result['best_candidate'] = best
    if best['selection_mae'] <= .99 * baseline and baseline > 0:
        result.update(status='selected_experimental', selected=best['calibration'])
    return result


def paired_report(frame, output: Path, label):
    """Score a single common mask, with date resampling for paired gains."""
    output.mkdir(parents=True, exist_ok=True)
    present = [v for v in VARIANTS if v in frame]
    mask = frame.scorable.astype(bool) & np.isfinite(frame[['actual', *present]].to_numpy(float)).all(axis=1)
    matched = frame.loc[mask].copy()
    complete_issues = frame.groupby('issue_time_utc').scorable.all()
    complete = matched[matched.issue_time_utc.isin(complete_issues[complete_issues].index)]
    result = {'label': label, 'dates': sorted(matched.target_date.unique()),
              'matched_hours': {}, 'complete_windows': {}, 'monthly_mean_correction_knots': {}}
    for window, subset in [('matched_hours', matched), ('complete_windows', complete)]:
        for variant in present:
            score = core.metric(subset[variant], subset.actual) if len(subset) else {'count': 0}
            if len(subset) and variant != 'harmonie':
                score['vs_harmonie'] = core.bootstrap(subset, variant, 'harmonie')
                score['vs_uncalibrated'] = core.bootstrap(subset.assign(comparison_prediction=subset[variant]), 'comparison_prediction', 'uncalibrated')
                base = core.metric(subset.uncalibrated, subset.actual)['mae']
                score['relative_improvement_vs_uncalibrated'] = (base - score['mae']) / base if base else 0.
            result[window][variant] = score
    matched['month'] = matched.target_date.str[:7]
    matched['issue_hour'] = pd.to_datetime(matched.issue_time_utc, utc=True).dt.tz_convert(core.TZ).dt.hour
    matched['target_hour'] = pd.to_datetime(matched.target_time_utc, utc=True).dt.tz_convert(core.TZ).dt.hour
    matched['lead_time'] = ((pd.to_datetime(matched.target_time_utc, utc=True)
                             - pd.to_datetime(matched.issue_time_utc, utc=True)).dt.total_seconds() / 3600)
    breakdowns = []
    for dimension in ['issue_hour', 'target_hour', 'lead_time', 'target_date', 'month']:
        for value, group in matched.groupby(dimension):
            for variant in present:
                breakdowns.append({'dimension': dimension, 'value': value, 'model': variant,
                                   **core.metric(group[variant], group.actual)})
    for variant in present:
        if variant not in ('harmonie', 'uncalibrated'):
            result['monthly_mean_correction_knots'][variant] = (matched[variant] - matched.uncalibrated).groupby(matched.month).mean().to_dict()
    frame.to_csv(output / f'{label}_predictions.csv', index=False)
    pd.DataFrame(breakdowns).to_csv(output / f'{label}_breakdowns.csv', index=False)
    d2.write_json(output / f'{label}_report.json', result)
    return result


def d2_rows(fit, samples, legacy, simple):
    pred = core.predict(fit, samples, calibrate=False)
    baseline = np.stack([s.frame.forecast_avg.to_numpy(float) for s in samples])
    corrected = shared.apply_speed_regime_calibration(pred, baseline, legacy,
                    core.calibration_context(samples), target_mask=np.stack([s.speed_mask for s in samples]))
    rows = pd.DataFrame(core.prediction_rows(samples, pred, fit, 'calibration_research'))
    rows['uncalibrated'] = pred.reshape(-1)
    rows['legacy'] = corrected.reshape(-1)
    rows['simple'] = apply_affine(pred, simple).reshape(-1)
    rows['issue_time_utc'] = rows.issue_time_utc.astype(str)
    rows['speed_model_id'] = fit.info['model_id']
    return rows


def copy_next_day_reference(source: Path, output: Path):
    metadata = json.loads((source / 'metadata_update.json').read_text())
    detail = Path(metadata['model_selection_gate']['speed_eval_details_csv'])
    if not detail.is_absolute():
        detail = source.parents[1] / detail
    names = ['next_day_lstm_speed_residual.pt', 'x_mean_speed.npy', 'x_std_speed.npy',
             'y_mean_speed.npy', 'y_std_speed.npy', 'metadata_update.json']
    sources = [source / n for n in names] + [detail]
    before = {str(p): sha256(p) for p in sources}
    output.mkdir(parents=True, exist_ok=True)
    for path in sources:
        shutil.copy2(path, output / path.name)
    after = {str(p): sha256(p) for p in sources}
    if before != after or any(sha256(output / p.name) != before[str(p)] for p in sources):
        raise RuntimeError('Next-day artifacts changed during copying; retry a consistent snapshot')
    return {'source_sha256': before, 'gate_details': detail.name}


def load_next_day(directory):
    checkpoint = torch.load(directory / 'next_day_lstm_speed_residual.pt', map_location='cpu', weights_only=True)
    cls = core.TargetAwareNextDayLSTM if checkpoint['model_class'] == 'TargetAwareNextDayLSTM' else core.NextDayLSTM
    kwargs = dict(n_features=checkpoint['n_features'], target_hours=checkpoint['target_hours'],
                  output_activation=checkpoint.get('output_activation', 'linear'))
    if cls is core.TargetAwareNextDayLSTM:
        kwargs['history_hours'] = checkpoint.get('history_hours', 72)
    model = cls(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scalers = {key: np.load(directory / f'{key}_speed.npy') for key in ['x_mean', 'x_std', 'y_mean', 'y_std']}
    return model, checkpoint, scalers


def next_day_predictions(model, checkpoint, scalers, x, baseline, context):
    scaled = core.dp._apply_standardizer(x, scalers['x_mean'], scalers['x_std']).astype(np.float32)
    batches = [shared._predict_speed_batch(model, scaled[i:i+256], baseline[i:i+256],
        float(scalers['y_mean'][0]), float(scalers['y_std'][0]), checkpoint['target_mode'],
        checkpoint.get('constraint_eps'), None, None, torch.device('cpu')) for i in range(0, len(x), 256)]
    raw = np.concatenate(batches)
    calibrated = shared.apply_speed_regime_calibration(raw, baseline, checkpoint.get('speed_regime_calibration'), context)
    return raw, calibrated


def next_day_historical(db, reference, output, reference_info):
    model, checkpoint, scalers = load_next_day(reference)
    schema = shared._next_day_feature_schema_from_scalers(scalers)
    saved = pd.read_csv(reference / reference_info['gate_details'])
    saved_times = pd.to_datetime(saved.target_time_utc, utc=True)
    # The daily gate may have included its latest partially observed hour.
    # Preserve the actual values in its saved aligned artifact, rather than
    # silently substituting the later completed-hour mean from a fresh snapshot.
    observations = core.dp.load_training_observations(db, core.dp.DatasetConfig(), read_only=True)
    revised = observations.reindex(saved_times).actual_avg.to_numpy(float) - saved.actual_wind_speed.to_numpy(float)
    observations.loc[saved_times, 'actual_avg'] = saved.actual_wind_speed.to_numpy(float)
    arrays = core.dp.build_all_training_arrays(db, core.dp.DatasetConfig(), target_mode='residual',
                                              feature_schema=schema, observations=observations, read_only=True)
    targets = pd.to_datetime(arrays['target_times_all'].reshape(-1), utc=True).to_numpy().reshape(arrays['target_times_all'].shape)
    valid = (targets[:, 0] >= saved_times.min()) & (targets[:, -1] <= saved_times.max())
    metadata = json.loads((reference / 'metadata_update.json').read_text())
    expected = metadata['model_selection_gate']['speed_eval_samples']
    if int(valid.sum()) != expected:
        raise ValueError(f'Next-day gate context mismatch: {valid.sum()} reconstructed, {expected} saved')
    anchor = pd.to_datetime(arrays['timestamps'][valid], utc=True)
    context = shared._build_speed_calibration_context(arrays['anchor_forecast_dir_all_raw'][valid], anchor + pd.Timedelta(hours=1))
    context.update(target_forecast_dir_deg=arrays['target_forecast_dir_all_raw'][valid],
                   target_times_utc=arrays['target_times_all'][valid], target_horizon_hr=arrays['target_horizon_hr_all'][valid])
    baseline = arrays['y_forecast_all_raw'][valid]
    raw, calibrated = next_day_predictions(model, checkpoint, scalers, arrays['X_all_raw'][valid], baseline, context)
    rows = pd.DataFrame({'issue_time_utc': np.repeat(anchor.astype(str), checkpoint['target_hours']),
        'target_time_utc': arrays['target_times_all'][valid].reshape(-1), 'actual': arrays['y_actual_all_raw'][valid].reshape(-1),
        'harmonie': baseline.reshape(-1), 'uncalibrated': raw.reshape(-1), 'legacy': calibrated.reshape(-1),
        'harmonie_run_ts': arrays['target_run_ts_all'][valid].reshape(-1),
        'harmonie_fetched_ts': arrays['target_fetched_ts_all'][valid].reshape(-1),
        'native_horizon_hr': arrays['target_horizon_hr_all'][valid].reshape(-1), 'scorable': True})
    rows['target_date'] = pd.to_datetime(rows.target_time_utc, utc=True).dt.tz_convert(core.TZ).dt.date.astype(str)
    rows['model_trained_at_utc'] = checkpoint['trained_at_utc']
    rows['neural_checkpoint_sha256'] = sha256(reference / 'next_day_lstm_speed_residual.pt')
    averaged = rows.groupby(pd.to_datetime(rows.target_time_utc, utc=True))[['actual', 'harmonie', 'legacy']].mean()
    reference_rows = saved.set_index(saved_times)
    averaged = averaged.reindex(reference_rows.index)
    differences = {}
    for calculated, stored in [('actual', 'actual_wind_speed'), ('harmonie', 'forecast_wind_speed'), ('legacy', 'champion_wind_speed')]:
        differences[calculated] = float(np.max(np.abs(averaged[calculated].to_numpy() - reference_rows[stored].to_numpy())))
        if not np.allclose(averaged[calculated], reference_rows[stored], atol=1e-4, rtol=1e-5, equal_nan=False):
            raise ValueError(f'Saved next-day gate cannot be reproduced: {calculated}, max error {differences[calculated]}')
    report = paired_report(rows, output, 'next_day_historical')
    report.update(saved_gate_reproduced=True, maximum_reproduction_error=differences,
                  observation_source='Saved aligned gate observations; avoids replacing its partial-hour labels with later means',
                  later_snapshot_max_observation_revision=float(np.nanmax(np.abs(revised))),
                  model_trained_at_utc=checkpoint['trained_at_utc'],
                  interpretation='Historical paired diagnostic; preserves the existing rolling 24-hour gate definition.')
    d2.write_json(output / 'next_day_historical_report.json', report)
    return report


def write_manifest(directory, manifest):
    path = directory / 'frozen_manifest.json'
    if path.exists() or path.with_suffix('.sha256').exists():
        raise FileExistsError('Frozen experiment already exists; never overwrite its rules or artifacts')
    d2.write_json(path, manifest)
    path.with_suffix('.sha256').write_text(sha256(path) + '\n')
    path.chmod(0o444)
    path.with_suffix('.sha256').chmod(0o444)


def validate_manifest(directory):
    path = directory / 'frozen_manifest.json'
    if sha256(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Frozen manifest was altered')
    manifest = json.loads(path.read_text())
    for relative, expected in manifest['artifacts'].items():
        candidate = directory / relative
        if candidate.resolve() == directory.resolve() or directory.resolve() not in candidate.resolve().parents:
            raise ValueError('Frozen artifact path escapes the experiment directory')
        if sha256(candidate) != expected:
            raise ValueError(f'Frozen artifact was altered: {relative}')
    return manifest


def first_confirmation_date(freeze):
    local = core.utc(freeze).tz_convert(core.TZ)
    issue_day = local.date()
    start = pd.Timestamp(datetime.combine(issue_day, time(7)), tz=core.TZ)
    if start <= local:
        issue_day += timedelta(days=1)
    return issue_day + timedelta(days=2)


def prepare(review, reference_source, output):
    if output.exists():
        raise FileExistsError('Use a new research directory to preserve previous experiments')
    core.validate_output(output.resolve(), review / 'snapshots/weather.sqlite', review / 'snapshots/ecmwf.sqlite', reference_source)
    output.mkdir(parents=True)
    artifact = review / 'models/day_after_tomorrow'
    manifest = json.loads((artifact / 'champions.json').read_text())
    fit = core.load_fit(artifact / manifest['speed']['checkpoint'])
    if fit.calibration or fit.info.get('calibration_policy') != 'none':
        raise ValueError('Research baseline must be the selected uncalibrated operational D+2 model')
    frozen = output / 'frozen'
    frozen.mkdir()
    shutil.copy2(artifact / manifest['speed']['checkpoint'], frozen / 'd2_speed.pt')
    shutil.copy2((artifact / manifest['speed']['checkpoint']).with_suffix('.scalers.npz'), frozen / 'd2_speed.scalers.npz')
    samples = core.load_samples(artifact / 'samples.npz')
    fitting, validation = core.purged_split(samples, core.utc(fit.info['cutoff_utc']))
    if sorted({s.day for s in validation}) != fit.info['validation_dates']:
        raise ValueError('Research validation differs from the frozen network training dates')
    simple = select_affine(fit, validation)
    d2.write_json(output / 'affine_selection.json', simple)
    d2.write_json(frozen / 'affine_selection.json', simple)
    raw = core.predict(fit, validation, calibrate=False)
    legacy = shared.fit_speed_regime_calibration(raw, np.stack([s.frame.forecast_avg.to_numpy(float) for s in validation]),
        np.stack([s.actual for s in validation]), core.calibration_context(validation),
        target_mask=np.stack([s.speed_mask for s in validation]))
    calibrations = {'legacy': legacy, 'simple': simple['selected'], 'simple_selection_status': simple['status']}
    d2.write_json(frozen / 'd2_calibrations.json', calibrations)
    gate_dates = set(fit.info['gate_dates'])
    gate = [s for s in samples if s.day in gate_dates]
    report = paired_report(d2_rows(fit, gate, legacy, simple['selected']), output, 'd2_historical')
    report.update(historical_choice_informed_by_gate=True, simple_selection=simple)
    d2.write_json(output / 'd2_historical_report.json', report)
    reference_dir = frozen / 'next_day'
    reference_info = copy_next_day_reference(reference_source, reference_dir)
    d2.write_json(output / 'next_day_reference_copy.json', reference_info)
    next_report = next_day_historical(review / 'snapshots/weather.sqlite', reference_dir, output, reference_info)
    freeze_time = pd.Timestamp.now(tz='UTC').isoformat()
    files = {str(p.relative_to(output)): sha256(p) for p in frozen.rglob('*') if p.is_file()}
    protocol = {'version': 1, 'development_only': True, 'freeze_time_utc': freeze_time,
        'first_eligible_target_date': first_confirmation_date(freeze_time).isoformat(),
        'required_target_dates': 20, 'minimum_issue_contexts_per_horizon': 60,
        'information_cutoff': 'Hourly local issues 07:00–22:00; history ends before issue; eligible vintages only',
        'target_window': '08:00–22:00 local for both horizons', 'artifacts': files,
        'calibrations': calibrations, 'simple_selection_status': simple['status'], 'selection_rules': {'ridges': list(RIDGES), 'minimum_gain': .01,
            'partition': '70/30 chronological validation dates; purge labels crossing selection issues',
            'minimum_fitting_dates': 7, 'minimum_selection_dates': 5, 'minimum_contexts_each': 32,
            'tie_break': 'strongest ridge within 1e-9 MAE', 'no_refit_after_freeze': True},
        'assessment': {'minimum_relative_mae_improvement': .01, 'bootstrap_ci_above_zero': True,
                       'bootstrap_unit': 'local target date', 'bootstrap_iterations': 2000,
                       'bootstrap_seed': 42, 'no_optional_extension': True},
        'd2_model_id': fit.info['model_id'], 'next_day_model_trained_at_utc': next_report['model_trained_at_utc']}
    write_manifest(output, protocol)
    for p in frozen.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    d2.write_json(output / 'confirmation_status.json', {'status': 'pending', 'completed_dates': 0,
        'required_dates': 20, 'first_eligible_target_date': protocol['first_eligible_target_date']})
    write_research_review(output, report, next_report, protocol)
    return protocol


def write_research_review(output, report, next_report, protocol):
    import matplotlib.pyplot as plt
    table = []
    for horizon, metrics in [('D+2', report['matched_hours']), ('Next-day', next_report['matched_hours'])]:
        for name, score in metrics.items():
            table.append(f"| {horizon} | {name} | {score['mae']:.3f} | {score['rmse']:.3f} | {score['bias']:+.3f} |")
    next_gain = next_report['matched_hours']['legacy']['relative_improvement_vs_uncalibrated']
    next_ci = next_report['matched_hours']['legacy']['vs_uncalibrated']['ci95']
    text = f'''# Calibration investigation — development only

[Forecasts](../dashboard/index.html) · [Evaluation](../dashboard/evaluation.html)

D+2 speed calibration is disabled in the local operational pipeline. Historical gate results informed that choice;
these comparisons are diagnostics, not fresh confirmation. Production is unchanged.

| Horizon | Variant | MAE (knots) | RMSE (knots) | Bias (knots) |
|---|---|---:|---:|---:|
{chr(10).join(table)}

Next-day calibration reduces historical MAE by **{next_gain:.1%}**. The 95% target-date bootstrap interval
for its absolute MAE gain is **{next_ci[0]:.3f}–{next_ci[1]:.3f} knots**; it crosses zero, so the evidence is inconclusive.

D+2 variants share the identical neural predictions. Simple affine selection: **{report['simple_selection']['status']}**.
When no affine candidate passes selection, the `simple` variant is the identity/no-correction fallback.
It uses no seasonal, direction or target-hour terms. Fitting and selection use separate chronological validation-date
partitions, purged at the issue boundary. Neural early stopping previously used that validation period,
so it is not an independent final model test. Selection details: [JSON](affine_selection.json).

The copied next-day champion reproduces its saved aligned gate predictions within numerical tolerance.
The historical comparison preserves the observations recorded in the saved gate artifact,
including its latest partial observation hour; it does not substitute later completed-hour means.
Its historical gate covers rolling 24-hour targets; the D+2 gate covers 08:00–22:00.
Compare calibration variants within each horizon, not their headline MAEs across different windows.

[Paired D+2 predictions](d2_historical_predictions.csv) · [Next-day predictions](next_day_historical_predictions.csv)

![Calibration by target month](calibration_comparison.png)

## Frozen confirmation

Status: **pending — 20 new completed target dates required**.
Frozen at {protocol['freeze_time_utc']}; first eligible target date {protocol['first_eligible_target_date']}.
Rules, coefficients and model/scaler hashes are in [the immutable manifest](frozen_manifest.json).
No refitting, tuning, production promotion or scheduling occurs. Replay is explicitly distinguished from predictions saved at issuance.
Results are assessed once after 20 eligible dates; at least 60 usable contexts per horizon are required.
A ≥1% MAE improvement with a target-date bootstrap interval above zero is promising; otherwise report inconclusive or worse.
'''
    (output / 'calibration_review.md').write_text(text)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, label in zip(axes, ['d2_historical', 'next_day_historical']):
        data = pd.read_csv(output / f'{label}_breakdowns.csv')
        months = data[data.dimension == 'month']
        for name, color in [('harmonie', 'grey'), ('uncalibrated', '#1f77b4'), ('legacy', '#ff7f0e'), ('simple', '#228833')]:
            part = months[months.model == name]
            if len(part):
                ax.plot(part.value, part.mae, marker='o',
                        label='simple (none selected)' if name == 'simple' and report['simple_selection']['selected'] is None else name, color=color)
        ax.set_title('D+2' if label.startswith('d2') else 'Next-day — saved gate reproduced')
        ax.set_ylabel('Speed MAE (knots)'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'calibration_comparison.png', dpi=150); plt.close(fig)


def next_day_issue(lookup, obs, issue, model, checkpoint, scalers):
    """Native 24-hour next-day input, with an actual issue information cutoff.

Unlike the production calendar-anchor helper, this replay never advances the
history/cutoff to tonight. Forecast selection and feature construction are the
shared functions; learned next-day weights and scalers are unchanged.
    """
    target_day = issue.tz_convert(core.TZ).date() + timedelta(days=1)
    start = pd.Timestamp(datetime.combine(target_day, time(0)), tz=core.TZ)
    targets = pd.date_range(start, periods=checkpoint['target_hours'], freq='h').tz_convert('UTC')
    history_times = pd.date_range(end=issue - pd.Timedelta(hours=1), periods=checkpoint.get('history_hours', 72), freq='h')
    history = core.dp._build_training_history_forecast_frame(lookup, history_times, core.dp._target_ms(issue))
    target = core.dp._select_training_complete_run_frame(lookup, targets, core.dp._target_ms(issue))
    if history is None or target is None:
        return None
    schema = shared._next_day_feature_schema_from_scalers(scalers)
    if schema == 'speed_v3_actual_history':
        # Complete observation hours only; future rows remain unavailable.
        raise ValueError('Frozen replay currently requires the existing forecast-only speed_v2 schema')
    built = core.dp._build_feature_sequence(history, target, feature_schema=schema)
    if built is None or not np.isfinite(built[0]).all():
        return None
    context = shared._build_speed_calibration_context(float(history.forecast_dir.iloc[-1]), [targets[0]])
    context.update(target_forecast_dir_deg=target.forecast_dir.to_numpy(np.float32)[None],
                   target_times_utc=targets.astype(str).to_numpy()[None],
                   target_horizon_hr=target.horizon_hr.to_numpy(np.float32)[None])
    raw, calibrated = next_day_predictions(model, checkpoint, scalers, built[0][None],
                                            target.forecast_avg.to_numpy(np.float32)[None], context)
    actual = obs.reindex(targets).actual_avg.to_numpy(float)
    rows = pd.DataFrame({'issue_time_utc': issue.isoformat(), 'target_time_utc': targets.astype(str),
        'target_date': [t.tz_convert(core.TZ).date().isoformat() for t in targets],
        'actual': actual, 'harmonie': target.forecast_avg.to_numpy(float),
        'uncalibrated': raw[0], 'legacy': calibrated[0],
        'harmonie_run_ts': target.run_ts.to_numpy(float), 'harmonie_fetched_ts': target.fetched_ts.to_numpy(float),
        'native_horizon_hr': target.horizon_hr.to_numpy(float), 'scorable': np.isfinite(actual),
        'model_trained_at_utc': checkpoint['trained_at_utc']})
    hours = targets.tz_convert(core.TZ).hour
    return rows[(hours >= 8) & (hours <= 22) & (rows.target_date == target_day.isoformat())].copy()


def assess_confirmation(reports, contexts, manifest):
    minimum = manifest['minimum_issue_contexts_per_horizon']
    if any(count < minimum for count in contexts.values()):
        return {'status': 'insufficient_contexts', 'contexts': contexts, 'required_contexts': minimum,
                'period_extended': False}
    decisions = {}
    for horizon, report in reports.items():
        variant = 'simple' if horizon == 'd2' else 'legacy'
        score = report['matched_hours'][variant]
        improvement = score['relative_improvement_vs_uncalibrated']
        interval = score['vs_uncalibrated']['ci95']
        decisions[horizon] = {'relative_improvement': improvement, 'ci95_absolute_gain': interval,
            'outcome': ('promising' if improvement >= .01 and interval[0] > 0
                        else 'worse' if improvement < 0 and interval[1] < 0 else 'inconclusive')}
    return {'status': 'complete', 'contexts': contexts, 'decisions': decisions, 'period_extended': False,
            'production_promotion': False, 'operational_d2_calibration': 'none'}


def remove_snapshot(db):
    # All connections have closed; these files belong only to this ephemeral copy.
    for suffix in ('', '-wal', '-shm'):
        Path(str(db) + suffix).unlink(missing_ok=True)


def confirm(output, source_db, source_ecmwf, as_of):
    """Manually refresh snapshots and replay without fitting or promotion."""
    protocol = validate_manifest(output)
    previous_path = output / 'confirmation_status.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous.get('status') in {'complete', 'insufficient_contexts'}:
        return previous  # Fixed period assessed once; idempotent subsequent calls.
    as_of = core.utc(as_of)
    if as_of < core.utc(protocol['freeze_time_utc']):
        raise ValueError('Confirmation cannot precede freezing')
    core.validate_output(output.resolve(), source_db, source_ecmwf, None)
    run = output / 'confirmation' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    run.mkdir(parents=True)
    db = run / 'weather.sqlite'
    core.snapshot(source_db, db, tables=('observations', 'forecasts'))
    obs, last = core.observations(db)
    first = pd.Timestamp(protocol['first_eligible_target_date']).date()
    # Completion requires the full 22:00–23:00 observation hour, not its first reading.
    end = min(as_of, last).tz_convert(core.TZ).date()
    if end < first:
        status = {'status': 'pending', 'completed_dates': 0, 'required_dates': 20,
                  'first_eligible_target_date': first.isoformat(), 'replay_as_of_utc': as_of.isoformat()}
        d2.write_json(previous_path, status)
        d2.write_json(run / 'status.json', status)
        d2.write_json(run / 'snapshot_provenance.json', {'source': str(source_db.resolve()),
            'snapshot_sha256': sha256(db), 'last_observation_utc': last.isoformat(),
            'as_of_utc': as_of.isoformat(), 'read_only_source': True})
        remove_snapshot(db)
        return status
    start_ms = int((pd.Timestamp(datetime.combine(first - timedelta(days=2), time(7)), tz=core.TZ)
                    - pd.Timedelta(hours=80)).timestamp() * 1000)
    lookup = core.dp.load_training_forecast_lookup(db, core.dp.DatasetConfig(), read_only=True,
                 target_start_ts_ms=start_ms, target_end_ts_ms=int((as_of + pd.Timedelta(days=1)).timestamp() * 1000))
    fit = core.load_fit(output / 'frozen/d2_speed.pt')
    model, checkpoint, scalers = load_next_day(output / 'frozen/next_day')
    cal = protocol['calibrations']
    dates, exclusions, d2_frames, next_frames = [], [], [], []
    for day in pd.date_range(first, end, freq='D'):
        date_value = day.date()
        completed = pd.Timestamp(datetime.combine(date_value, time(23)), tz=core.TZ).tz_convert('UTC')
        if completed > as_of or completed > last:
            break
        samples, next_rows = [], []
        for hour in range(7, 23):
            issue_d2 = pd.Timestamp(datetime.combine(date_value - timedelta(days=2), time(hour)), tz=core.TZ).tz_convert('UTC')
            if issue_d2 < core.utc(protocol['freeze_time_utc']):
                raise ValueError('Issue predates frozen experiment')
            if core.utc(fit.info['trained_at_utc']) > issue_d2 or core.utc(fit.info['max_label_end_utc']) > issue_d2:
                raise ValueError('Frozen D+2 model contains observations unavailable at issue')
            sample = core.make_sample(lookup, obs, core.ECMWFArchive(None), issue_d2)
            if sample is not None and sample.speed_mask.any():
                samples.append(sample)
            issue_next = pd.Timestamp(datetime.combine(date_value - timedelta(days=1), time(hour)), tz=core.TZ).tz_convert('UTC')
            if core.utc(checkpoint['trained_at_utc']) > issue_next:
                raise ValueError('Frozen next-day model was unavailable at issue')
            rows = next_day_issue(lookup, obs, issue_next, model, checkpoint, scalers)
            if rows is not None and rows.scorable.any():
                next_rows.append(rows)
        if not samples or not next_rows:
            exclusions.append({'target_date': date_value.isoformat(), 'reason': 'no_matched_observations_in_both_horizons',
                               'd2_contexts': len(samples), 'next_day_contexts': len(next_rows)})
            continue
        date_d2 = d2_rows(fit, samples, cal['legacy'], cal['simple'])
        date_next = pd.concat(next_rows, ignore_index=True)
        # Qualification uses shared target hours, never forecast error magnitude.
        common = set(date_d2.loc[date_d2.scorable, 'target_time_utc'].map(core.utc)) & set(date_next.loc[date_next.scorable, 'target_time_utc'].map(core.utc))
        if not common:
            exclusions.append({'target_date': date_value.isoformat(), 'reason': 'no_common_matched_target_hours'})
            continue
        for frame in [date_d2, date_next]:
            frame['prediction_origin'] = 'replay_with_frozen_models'
            frame['saved_at_issuance'] = False
            frame['freeze_time_utc'] = protocol['freeze_time_utc']
        d2_frames.append(date_d2); next_frames.append(date_next); dates.append(date_value.isoformat())
        if len(dates) == protocol['required_target_dates']:
            break
    status = {'status': 'pending', 'completed_dates': len(dates), 'required_dates': 20,
              'first_eligible_target_date': first.isoformat(), 'evaluation_dates': dates,
              'exclusions': exclusions, 'replay_as_of_utc': as_of.isoformat(),
              'models_refitted': False, 'prediction_origin': 'replay_with_frozen_models'}
    pd.DataFrame(exclusions).to_csv(run / 'coverage_exclusions.csv', index=False)
    if d2_frames:
        frames = {'d2': pd.concat(d2_frames, ignore_index=True), 'next_day': pd.concat(next_frames, ignore_index=True)}
        for horizon, frame in frames.items():
            frame.to_csv(run / f'{horizon}_predictions.csv', index=False)
        status['contexts'] = {h: int(f.loc[f.scorable].issue_time_utc.nunique()) for h, f in frames.items()}
        if len(dates) == protocol['required_target_dates']:
            reports = {h: paired_report(f, run, f'{h}_confirmation') for h, f in frames.items()}
            # Explicit common-hour cross-horizon summaries; no calibration tuning.
            common = set(frames['d2'].loc[frames['d2'].scorable, 'target_time_utc'].map(core.utc)) & set(frames['next_day'].loc[frames['next_day'].scorable, 'target_time_utc'].map(core.utc))
            for horizon, frame in frames.items():
                paired_report(frame[frame.target_time_utc.map(core.utc).isin(common)], run, f'{horizon}_common_hours')
            status.update(assess_confirmation(reports, status['contexts'], protocol))
    validate_manifest(output)
    d2.write_json(previous_path, status)
    d2.write_json(run / 'status.json', status)
    # Keep compact provenance and predictions, rather than accumulating live-size snapshots.
    d2.write_json(run / 'snapshot_provenance.json', {'source': str(source_db.resolve()), 'snapshot_sha256': sha256(db),
        'last_observation_utc': last.isoformat(), 'as_of_utc': as_of.isoformat(), 'read_only_source': True})
    remove_snapshot(db)
    return status
