# Day-after-tomorrow experiment

The D+2 runner evaluates Valkenburgse Meer at hourly issue times from 07:00
through 22:00 Europe/Amsterdam. Its target is the local calendar date two days
later, 08:00–22:00 inclusive. It trains dedicated models using the existing
next-day LSTM classes: target-aware constrained log-ratio speed correction and
circular direction residuals. Existing production checkpoints are neither
read for evaluation nor replaced.

Run from an isolated development worktree using the model Python environment:

```bash
python next_day_wind_model/day_after_tomorrow_experiment.py \
  --db /path/to/wind_data_all_sites.db \
  --ecmwf-archive /path/to/ecmwf_archive/ecmwf_shadow.sqlite \
  --issue-time 2026-10-01T18:00:00+02:00 \
  --output-dir next_day_wind_model/artifacts_dev/day_after_tomorrow/20261001 \
  --reference-dashboard /path/to/production/docs
```

The issue time must carry an offset. Output must be a new directory outside
the production main checkout and source runtime directories. The runner opens
source databases read-only/query-only and uses SQLite's online backup API to
create consistent local snapshots, including committed WAL rows. Allow space
for a primary database snapshot; only forecasts and observations are retained
in the local primary snapshot. It never calls collectors, live prediction
logging, promotion, publication, cron, or service management.

Defaults are 30 maximum epochs, batch size 32, seed 42, and two CPU threads.
`--resume` continues an interrupted run from its isolated snapshots and numeric
sample cache. The issue time, input paths, seed, epochs, and batch size must
match the original manifest. Completed evaluation dates are retained.

## Availability and missing coverage

The 72 history hours end before the issue hour. Future targets come from one
latest HARMONIE run entirely available at the issue cutoff. Missing target
hours stay missing, even if an older run covers them. Input padding has a
separate availability flag; masked losses exclude absent forecasts and labels.
Padded values never appear as predictions, baselines, or scored observations.

ECMWF selection uses run time, completion time, and point fetch time. Target
features are 10 m u/v components, derived speed, and gust, converted to knots.
Interpolation is allowed between finite native points at most three hours
apart; there is no extrapolation or bridging over missing native points.
ECMWF adds target features to the speed model; the dedicated direction model
uses the existing direction schema. P5 is excluded.

## Evaluation

The latest 20% of completed local target dates form the historical evaluation.
For each evaluation target date, models are refitted using only labels complete
at that date's earliest issue cutoff, then replay all available issue hours.
The latest 20% of eligible fitting dates are validation. Training samples whose
labels would finish after the earliest validation issue are purged. Input and
target scalers are fitted only on the remaining training samples. Speed uses
the existing target-hour ridge calibration on valid validation points.

The ECMWF comparison reserves the latest five usable target dates and requires
at least seven training dates **after validation and purging** before the first
evaluation issue. Short archives are reported as `insufficient_data`, including
the dates remaining for training and validation. This is stricter than counting
archive days: D+2 targets and validation each consume lead time. Both feature
variants use the same samples, masks, fitting/evaluation dates, and seed.

Speed scores are MAE, RMSE, and bias in knots. Direction scores wrap errors
onto [-180, 180) degrees, and observation directions are aggregated circularly.
Bootstrap intervals resample local target dates, keeping all hourly issues for
a date together. A >=1% relative MAE improvement with its 95% date-bootstrap
interval above zero is promising evidence, not automatic promotion.

Observation first-arrival timestamps are absent from the source schema. Label
eligibility therefore conservatively uses the end of each observation hour.
The source snapshot freezes what is available for this retrospective study.

## Outputs

- `report.md`, `report.json`: overall, complete-window, direction, and ECMWF
  results, training method, limitations, and preview metadata.
- `*_predictions.csv`: every issue/target pair, masks, baselines, observations,
  run/fetch provenance, and model training/label cutoffs.
- `*_by_issue_hour.csv`, `*_by_target_hour.csv`, `*_by_lead_hours.csv`,
  `*_by_target_date.csv`: grouped performance; `coverage.csv`: missing history,
  forecast, observation, and ECMWF feature coverage.
- `*_training.json`, `models/*.pt`, `models/*.scalers.npz`: isolated experimental
  checkpoints, train-only scalers, calibration, and exact fitting dates.
- `daily_improvement.png`: diagnostic figure.
- `day_after_tomorrow_direction_spider.png` and
  `day_after_tomorrow_speed_by_direction.csv`: speed MAE by the eight HARMONIE
  forecast direction sectors, using the existing next-day spider renderer,
  colours, legend, dimensions, and fixed 0–3.5-knot scale. Title and model label
  identify the experimental D+2 model. As in the next-day gate, overlapping
  issues are averaged per target hour before scoring and forecast directions
  are averaged circularly. Missing or unscorable pairs are excluded. Counts
  are unique target hours, so these MAEs can differ from issue-level scores.
  `day_after_tomorrow_direction_eval_details.csv` records those aggregated
  hours and overlap counts. The shared renderer's `champion_*` CSV fields
  represent the experimental D+2 model here, without implying promotion.
- `dashboard/index.html`: current-day and next-day reference panels plus the
  experimental D+2 desktop/mobile plot, wind-direction MAE spider diagram,
  forecast and sector CSVs, and rendering metadata.
  Reference panel timestamps remain visible; they are not relabeled as D+2
  forecasts or regenerated by this runner.
- `snapshots/`, `samples.npz`, `preview_sample.npz`: reproducible local inputs.

Serve the output directory locally to browse the dashboard and report:

```bash
python -m http.server 8081 --bind 127.0.0.1 \
  --directory next_day_wind_model/artifacts_dev/day_after_tomorrow/20261001
```

The D+2 plot keeps the full 08:00–22:00 axis, including missing-hour gaps,
weather strip, direction arrows, native ECMWF line, issue time, and coverage.
It is labeled experimental. Existing current-day/next-day rendering defaults
and production publication remain unchanged.
