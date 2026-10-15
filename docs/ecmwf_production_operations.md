# Production ECMWF operations

The production dashboard reads ECMWF from its own forecast-vintage archive. It does not write ECMWF rows to `wind_data_all_sites.db`, change model artefacts, or train a model.

## Runtime paths

- checkout: the primary `main` checkout
- Python: `.venv-ecmwf/bin/python`
- requirements: `next_day_wind_model/requirements-ecmwf-production.txt`
- configuration: `config/ecmwf_production.json`
- archive directory: `data/ecmwf_archive`
- archive database: `data/ecmwf_archive/ecmwf_shadow.sqlite`
- service: `wind-fetcher2-ecmwf.service`
- timer: `wind-fetcher2-ecmwf.timer`

The configuration has no experiment end date or run-count cap. The one-shot collector polls every 15 minutes, preserves every completed IFS vintage, and uses an exclusive archive lock. Its stdout and failures are recorded in the user journal.

## Selection and dashboard refresh

At one explicit `plot_updated_at_utc` cutoff, both current-day and next-day plots select the latest `complete` ECMWF IFS run whose run time and completion time are no later than that cutoff. Only that run's points covering the two plot windows plus six hours are queried. The archive is opened read-only/query-only, and recent arrival estimation loads only run/completion timestamps.

The normal six-minute updater gate reads one scalar ECMWF identity (`run_time|completed_time`). A changed identity runs the lightweight measured/dashboard stage; it never requests model training or full inference. That stage refreshes both plots and ECMWF metadata while keeping the cached HARMONIE and Superlocal forecast tables unchanged. If the archive is missing, busy, invalid, or has no eligible run, normal plotting continues without ECMWF.

`Last ECMWF fetch` is the selected run's recorded completion time. `Next expected fetch` starts with the next six-hour IFS cycle and adds the median completion latency of up to the 12 most recent successful runs. At least four valid observations are required; otherwise the configured eight-hour delay is used. Six-hour cycles are advanced until the estimate is later than the plot cutoff, so the displayed expectation is normally in the future.

ECMWF is a solid `#007f7f` wind-speed line at width 1.9 on both plots. It is excluded from the forecast plots' axes, inline MAE, direction, and variability calculations. The separate evaluation page includes ECMWF speed MAE as described below. The current-day renderer includes the first native point after the unchanged x-axis maximum so matplotlib clips the final native segment at the existing boundary.

## Direction-sector MAE on the evaluation page

The current-day, next-day, and D+2 windroses include a teal ECMWF speed-MAE
curve. Each chart retains the existing HARMONIE and Super local results and
uses HARMONIE forecast directions for all eight sectors. ECMWF coverage starts
later, so its own target dates and sample counts appear beside each chart.
The curves therefore do not necessarily cover identical historical weather.

`next_day_wind_model/ecmwf_evaluation.py` opens the archive read-only and
selects the newest complete IFS run whose run and completion timestamps are
no later than each original issue time. Exact points and both interpolation
brackets must also have been fetched by that issue. Scalar wind speed is
linearly interpolated between native points no more than three hours apart,
then converted to knots. There is no extrapolation, gap filling from older
vintages, or dependency on gust availability. This matches the dashboard's
piecewise-linear speed curve; the D+2 model-feature helper continues to use
its existing vector interpolation.

Current-day evaluation retains the first logged issue per anchor/target and
scores each case separately. Next-day uses the historical holdout anchor;
D+2 uses its stored issue time. Next-day and D+2 average available ECMWF
forecasts per target hour before scoring. The new issue-level CSVs preserve
run/completion provenance and reasons for missing ECMWF values. Sector CSVs
add `ecmwf_mae`, bias, scored-case and distinct-target-hour counts, UTC coverage
bounds, and `ecmwf_report_version`. Legend MAEs use each model's own counts.
Missing sectors stay at their true compass positions and leave line gaps;
values above 3.5 knots are annotated at the radial limit.

Full evaluations refresh these reports. Lightweight refreshes reuse them;
old current-day reports are migrated once when a cached dashboard is rebuilt.
Old D+2 reports can be rebuilt from `gate_predictions.csv` during a full or
prediction stage without fitting. Old next-day target-averaged exports lack
issue times: the next normal full gate evaluation writes the new issue CSV
and ECMWF results. No extra model training or database migration is needed.
Missing or unreadable archives produce a newly generated two-model report,
replacing any prior ECMWF values. ECMWF does not enter learned production
features or model-promotion decisions.

## Installation and persistence

Use a dedicated environment so ECMWF dependencies do not alter the production model environment:

```bash
python3 -m venv .venv-ecmwf
.venv-ecmwf/bin/python -m pip install -r next_day_wind_model/requirements-ecmwf-production.txt
```

Before installing production units, stop and uninstall the development collector timer. There must never be two scheduled collectors. Preserve the development archive with SQLite's backup API while the development service is stopped; do not copy a live WAL database directly.

Enable lingering once for the production user so the user systemd manager starts at boot and remains after logout:

```bash
sudo loginctl enable-linger sandervancranenburgh
loginctl show-user sandervancranenburgh -p Linger
```

Then install from the primary `main` checkout:

```bash
scripts/manage_ecmwf_service.py install
```

The manager refuses a linked worktree, a branch other than `main`, redirected runtime paths, an active/enabled development collector, or `Linger` other than `yes`. The timer uses `Persistent=yes`, so a missed scheduled activation is handled after the user manager restarts.

Inspect operation with:

```bash
scripts/manage_ecmwf_service.py status
scripts/manage_ecmwf_service.py logs --lines 100
systemctl --user list-timers wind-fetcher2-ecmwf.timer --no-pager
```

A oneshot service is normally `inactive (dead)` after a successful run. Healthy state is an active timer plus `Result=success` and `ExecMainStatus=0` for the service. `ProtectSystem=strict`, `ProtectHome=read-only`, `ReadWritePaths` for only the archive, `PrivateTmp`, and `NoNewPrivileges` constrain the collector.

## Controlled deployment checks

Before code integration, compare functional source changes rather than production's automatic dashboard commits and verify the memory-efficient/OOM fixes remain present. Confirm no updater or training process is active. Deploy only committed functional changes, create the isolated venv, migrate the archive, enable lingering, disable the development timer, and install the production timer.

Run the first dashboard canary with `--skip-training --skip-data-refresh-check`; do not launch training. Record peak RSS, inspect current-day and next-day plots and metadata, confirm publication, then wait for and verify one normal six-minute pipeline cycle. Roll back the production code commit and disable the production timer if the canary fails; retain the independent ECMWF archive for diagnosis.
