# D+2 operational integration — development and review

The implementation is opt-in and has not been deployed. It adds a dedicated
experimental Valkenburgse Meer forecast for the local date two days after
issuance, 08:00–22:00. Current-day and next-day models retain their checkpoints,
defaults and operational paths. The public forecast publisher now generates
`index.html` and `evaluation.html` together.

## Runtime contract

Enable through `--enable-day-after-tomorrow` or
`WIND_ENABLE_DAY_AFTER_TOMORROW=1`. The default is off;
`--no-enable-day-after-tomorrow` overrides the environment. Other sites remain
disabled. The dispatcher includes the D+2 activation flag and champion manifest
in its model fingerprint, so enabling, disabling or changing champions forces
the normal full update instead of silently reusing old website state.

Training runs with the existing daily training stage. D+2 fits use two CPU
threads and restore the caller's thread setting afterward. Predictions refresh
at hourly issues from 07:00 through 22:00. Cached and measured-only refreshes
read forecast CSVs and eligible ECMWF runs without training, inference or
prediction-log writes. A D+2 failure returns an unavailable/stale status and
allows the other horizons to finish.

Artifacts live under `<model-artifact-dir>/day_after_tomorrow/`. Candidate
checkpoints have unique immutable names; `champions.json` activates independently
selected speed/direction models only after both fits and evaluation exports
succeed. Experimental preview checkpoints are not valid operational champions.
Output forecast PNGs, mobile PNGs, CSV, interactive JSON and metadata use the
`day_after_tomorrow_` prefix. The CSV records the information cutoff, coverage,
model identities and HARMONIE run/fetch provenance.

The operational pipeline reuses shared masked sampling/training in
`day_after_tomorrow_core.py`; the historical experiment remains available in
`day_after_tomorrow_experiment.py`. Neither source collectors nor the ECMWF
environment are changed. P5 and learned ECMWF features are excluded.

## Gate and reporting

The latest 15% of completed local target dates are the promotion holdout, with
at least 60 usable issue contexts. Fitting labels are purged before the earliest
holdout issue. Within fitting data, 20% of dates form validation; training labels
are purged before validation issues. Scalers, calibration and early stopping
never see gate observations. Direction requires its own supervised contexts.

Speed calibration now uses the complete next-day selector: threshold,
contextual and target-hour ridge methods, choosing the improving candidate
with the lowest fitting MAE. An optional mask excludes unavailable targets
without flattening forecast windows or changing existing next-day defaults.
As in next-day, validation observations are reused for early stopping and
calibration; they are not an independent calibration acceptance set. The gate
remains separate. Do not tune calibration using its scores.

The full-archive scan writes `coverage.csv`, `coverage_by_month.csv` and
`coverage_exclusions.csv`. Candidate diagnostics compare HARMONIE, the neural
forecast before calibration and the final calibrated forecast on matched hours
and complete windows. Calibration type, fitting dates, seasonal corrections,
date-bootstrap intervals and grouped scores are exported alongside the gate.

Promotion requires at least 1% MAE improvement against the prior D+2 champion,
independently for speed and circular direction. As in the existing next-day
rule, a valid initial candidate becomes champion when none exists, even if it
loses to HARMONIE. The page explicitly reports that case. Target-date bootstrap
intervals are diagnostic rather than promotion conditions. Insufficient data
and failed fits retain active champions.

The spider uses the active selected model on the latest gate holdout, averages
overlapping issues per target hour, and assigns sectors by circular mean
HARMONIE direction. It retains the next-day colours and 0–3.5-knot radial scale;
values beyond the scale are marked and annotated. Gate history compares the
challenger against the prior champion. Holdout predictions, aggregated details,
sector scores, model IDs and date ranges are exported separately.

Issued predictions use the existing `prediction_log` schema with
`model_type='day_after_tomorrow'`, `wind_speed`/`wind_direction`, and the actual
issue cutoff as anchor. Logging is idempotent. D+2 realized scoring filters by
model type, aggregates speed observations hourly and directions circularly,
and excludes unfinished observation hours. Existing horizon queries keep
their own filters.

Forecasts stay on the homepage. Its “How much better are the super local
forecasts?” link opens the evaluation page, containing all existing evaluation
content in this order: current-day, next-day and D+2 spiders, followed by
current-day, next-day and D+2 model gates. The current-day gate is rendered
from its actual aligned holdout artifacts; its model pipeline is unchanged.
Navigation uses two equal-width blue
44-pixel buttons: Forecasts and Rider portal. Existing evaluation asset URLs
and CSV downloads remain available. Missing D+2 evaluations have an explicit
empty state; cached forecasts retain their actual target date and are labeled
stale across date rollover or after missed daytime updates.

D+2 uses the exact next-day desktop/mobile renderer and forecast-card
structure. The former D+2-only interactive renderer has been removed. Shared
interactive enhancements, when available, follow the next-day component.
“HARMONIE available for N of 15 forecast hours” describes coverage of the
displayed 08:00–22:00 window, not the number of training contexts.

## Safe local rehearsal

Run from the isolated worktree with the selected model interpreter:

```bash
python scripts/preview_day_after_tomorrow.py \
  --db /path/to/source/wind_data_all_sites.db \
  --ecmwf-archive /path/to/source/ecmwf_shadow.sqlite \
  --reference-dashboard /path/to/source/docs \
  --reference-model-artifacts /path/to/source/next_day_wind_model/artifacts \
  --output-dir next_day_wind_model/artifacts_dev/d2_aligned_review/20261002 \
  --issue-time 2026-10-02T15:00:00+02:00
```

Sources are opened read-only/query-only. The command creates online SQLite
backups in the output directory; only weather tables are retained from the
primary source. All training, logging, evaluation and publication writes go to
the isolated copies. It refuses production-checkout/live-runtime output paths.
It never invokes wrappers, collectors, Git publication, cron or services.
Source paths must refer to the same archive across repeated stages.
The reference-model-artifacts option copies only current-day gate reporting
inputs; it never imports existing model weights into D+2 training. Preview
defaults match the deployed next-day training settings: 30 maximum epochs,
batch size 16, constrained-speed epsilon 0.2 and seed 42. All model weights
are initialized from scratch. Legacy caches without an archive coverage
inventory are rebuilt rather than assumed to contain the full history.

Use `--stage predict` for hourly inference or `--stage refresh` for cached plots.
Use a later explicit issue cutoff to exercise rollover. `--sample-cache` may
seed historical weather samples from the same source archive; recent dates
are rebuilt from the local snapshot. This option never accepts model weights.
Current/next-day reference images are public artifact copies and retain their
own displayed timestamps, rather than being relabeled as D+2 issues.

Serve the output root:

```bash
python -m http.server 8766 --bind 127.0.0.1 \
  --directory next_day_wind_model/artifacts_dev/d2_operational_review/20261002
```

Open `/dashboard/index.html` and `/dashboard/evaluation.html`. Static plots
remain available if Plotly cannot load. The interactive D+2 view provides exact
speed/direction/weather values and an ECMWF line; its “Show full forecast plot”
button opens the complete existing static presentation.

## Deployment procedure — explicit approval required

1. Review the local pages, test results, operational gate results and current
   resource measurements. Production remains unchanged until approval.
2. Preserve the current production commit and scheduler configuration. Create
   and verify fresh online database backups using
   [database_backup_and_restore.md](database_backup_and_restore.md); also back
   up model artifacts and the separate ECMWF archive. Do not copy a live WAL DB.
3. Under the existing updater lock, install the reviewed integration commit in
   production. Keep D+2 disabled while verifying existing forecasts and the
   new evaluation-page navigation. Preserve scheduled execution times,
   collector guards, portal configuration and the ECMWF environment.
4. Enable D+2 for both existing training and six-minute update commands through
   the shared crontab environment `WIND_ENABLE_DAY_AFTER_TOMORROW=1` (or their
   explicit CLI switch). Initialize fresh production champions with the existing
   locked training command. Do not copy development databases or checkpoints.
5. Publish both pages and assets through the existing configured publication
   workflow. Verify issue/target date, gaps, model IDs, evaluation dates,
   navigation, downloads and desktop/mobile output on the live website.
6. Observe the next hourly inference and next daily training/gate run. Verify
   idempotent D+2 logging and that other horizons continue after a D+2 failure.

Rollback: set `WIND_ENABLE_DAY_AFTER_TOMORROW=0` for both jobs, removing any
explicit enable argument, and run the usual locked hourly update to restore the
two-horizon homepage. Retain D+2 models, prediction logs and evaluation history.
If broader website/code rollback is needed, restore the recorded production
commit and publish the previous website outputs. No data deletion is required.

The initial local frozen gate can perform differently from the rolling
historical experiment. A worse or inconclusive gate result is reported plainly;
it is not grounds to silently change the agreed promotion rule or deploy.

### Initial review finding on 2 October 2026

The rehearsal initialized champions, then retained them against a second
challenger. On 44 gate target dates (9,227 matched issue/target hours), the
calibrated speed model has MAE **3.03 knots**, versus **2.25** for HARMONIE.
The same speed model without calibration has MAE **1.86 knots**. The inherited
calibration improves its fitting validation data but degrades the later gate
data and clips all speeds to zero for the 4 October preview window. Calibration
was fitted on June–August validation dates; this later-season failure needs
review before deployment. The implemented policy and displayed forecasts have
not been silently altered to select a variant using gate observations.

Production is unchanged. Review the calibration diagnostic and holdout reports
alongside the website before approving any deployment.

### Aligned review on 2 October 2026

A fresh snapshot/full-archive scan found 4,653 usable issue contexts and 47,999
matched target hours. Fresh speed/direction candidates used the next-day
settings (30 maximum epochs, batch size 16, epsilon 0.2, seed 42). The speed
network stopped after 13 epochs; the fit used 3,102 training contexts and 786
validation/calibration contexts, with the same 44-date untouched gate.

The complete three-method selector still selected target-hour ridge calibration
on validation-fitting MAE. Gate speed MAE is **1.84 knots before calibration**,
**3.27 knots after calibration**, versus **2.25 knots for HARMONIE**. The
calibrated improvement interval is **-1.56 to -0.51 knots**, showing worse
performance. Calibration still clips the displayed D+2 speeds to zero.
No alternative was selected using gate observations. The seasonal calibration
failure remains a separate investigation before deployment.

The corrected development site uses the exact next-day forecast template and
all three spiders before all three model gates. Review artifacts live in
`next_day_wind_model/artifacts_dev/d2_aligned_review/20261002`; prior historical
and operational experiments remain separate. Repository validation passed
257 tests and 60 subtests, plus desktop/mobile navigation, download, fallback
and cache-refresh checks. Production remains unchanged by this work.
