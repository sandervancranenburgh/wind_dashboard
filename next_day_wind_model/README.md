# Next-Day Wind Model

This folder contains a trainable LSTM pipeline that predicts the **next 24 hours** of actual wind values from a moving window of historical forecasts.

## Data source

- SQLite database: `data/wind_data_all_sites.db`
- Input table: `forecasts`
- Target table: `observations`

The loader reads raw JSON payload columns to extract values from keys like:
- Forecast: `WindForecastAvr`, `WindForecastMax`, `WindDirection`
- Observation: `AverageWind`, `MaxWind`, `WindDirection`

## Features used by the LSTM

- `forecast_avg`
- `forecast_max`
- `forecast_dir`
- `month_sin` and `month_cos` (cyclical month-of-year seasonal features)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r next_day_wind_model/requirements.txt
```

## Train (Residual Speed)

```bash
python3 next_day_wind_model/train_lstm.py \
  --db data/wind_data_all_sites.db \
  --site valkenburgsemeer \
  --model HARMONIE \
  --window-hours 72 \
  --target-hours 24 \
  --epochs 40 \
  --batch-size 32 \
  --out-dir next_day_wind_model/artifacts
```
This trains the **residual speed model** only: target = `(actual_speed - forecast_speed)`.

## Hyperparameter Tuning (Both Models)

Tune both:
- next-day constrained residual model
- intraday residual model

```bash
python3 next_day_wind_model/tune_hyperparameters.py \
  --db data/wind_data_all_sites.db \
  --site valkenburgsemeer \
  --model HARMONIE \
  --n-splits 3 \
  --max-configs-next-day 18 \
  --max-configs-intraday 24 \
  --out-dir next_day_wind_model/artifacts
```

Outputs:
- `next_day_wind_model/artifacts/tuning_next_day_results.csv`
- `next_day_wind_model/artifacts/tuning_intraday_results.csv`
- `next_day_wind_model/artifacts/tuning_summary.json` (includes recommended training command)

## Output artifacts

- `next_day_wind_model/artifacts/next_day_lstm_speed_residual.pt` (residual speed)
- `x_mean_speed.npy`, `x_std_speed.npy`, `y_mean_speed.npy`, `y_std_speed.npy`
- `metadata.json`

`metadata.json` includes validation metrics and training config.

## Update Model + Next-Day Outputs

Run this script to retrain on **all available data** and produce next-day outputs.
It trains two residual models:
- wind speed residual model
- wind direction residual model

```bash
python3 next_day_wind_model/update_model_and_predict.py \
  --db data/wind_data_all_sites.db \
  --site valkenburgsemeer \
  --model HARMONIE \
  --window-hours 72 \
  --target-hours 24 \
  --validation-split 0.2 \
  --epochs 30 \
  --batch-size 32 \
  --out-dir next_day_wind_model/artifacts
```

Before training, the script now checks forecast freshness in the DB and can auto-run `source_fetch.py` when data are stale.

Useful options:
- `--max-forecast-age-hours 8`
- `--expected-update-hour-utc 1`
- `--skip-data-refresh-check` (disable auto-refresh)
- `--validation-split 0.2` (chronological holdout to monitor overfitting)

This creates:
- `next_day_wind_model/artifacts/next_day_predictions.csv`
- `next_day_wind_model/artifacts/next_day_predictions.png`
- `next_day_wind_model/artifacts/current_day_predictions.csv`
- `next_day_wind_model/artifacts/current_day_predictions.png`
- `next_day_wind_model/artifacts/metadata_update.json`
- `next_day_wind_model/artifacts/next_day_lstm_speed_residual.pt`
- `next_day_wind_model/artifacts/next_day_lstm_direction_residual.pt`
- `next_day_wind_model/web_dashboard/index.html` (static dashboard)

## Browser Dashboard + Rider Portal

The forecast dashboard and the rider/session features are intentionally deployed as two linked apps:

- Forecast dashboard: static HTML/images/CSVs, suitable for GitHub Pages.
- Rider portal: separate Flask app for login, profiles, surf experience submission, measured-wind attachment, and submissions/detail/delete.

### Preview worktree using production Rider Portal data

This helper is intended only for temporary UI validation. A separate preview worktree normally has no Rider Portal database, uploads, analyses, or archived current-day plots. The helper links those runtime paths to the production checkout so the UI can be checked against representative live data without copying private or generated files.

Run this from anywhere inside the preview worktree:

```bash
./scripts/setup_rider_portal_preview.sh \
  ~/Documents/repos/wind_fetcher2
```

The production path is optional. Without it, the helper uses the single other Git worktree on the `main` branch when one can be identified, then falls back to `~/Documents/repos/wind_fetcher2`.

Review the planned changes safely before configuring anything:

```bash
./scripts/setup_rider_portal_preview.sh \
  --dry-run \
  ~/Documents/repos/wind_fetcher2
```

By default, the helper does not replace `data/.wind_dashboard_secret`. Using a different Flask secret for the preview is harmless, but existing browser sessions will not carry over and a fresh login is expected. Link the production secret only when needed:

```bash
./scripts/setup_rider_portal_preview.sh \
  --link-secret \
  --force
```

Use `--force` only after reviewing the output if local non-symlink files or directories must be replaced. Running the helper again is safe; correctly configured links are left unchanged. Remove only links owned by the helper and restore tracked paths afterward with:

```bash
./scripts/setup_rider_portal_preview.sh --undo
```

**Warning:** the preview instance reads from and may write to the live production Rider Portal database and shared runtime directories. Avoid creating, editing, or deleting submissions during visual testing, and do not stage or commit the linked runtime paths. When `--link-secret` is used, the production secret is also shared with the preview.

Each normal model run (non-test mode) updates the static dashboard folder:

- `next_day_wind_model/web_dashboard/index.html`
- `next_day_wind_model/web_dashboard/current_day_predictions.png`
- `next_day_wind_model/web_dashboard/next_day_predictions.png`
- `next_day_wind_model/web_dashboard/daily_mae_history.png`

You can open locally:

```bash
python3 -m http.server 8080 -d next_day_wind_model/web_dashboard
```

Then browse to `http://<server-ip>:8080`.

For the rider portal companion app, run Flask separately:

```bash
pip install -r next_day_wind_model/requirements.txt
python3 next_day_wind_model/web_dashboard/app.py
```

Then browse to `http://127.0.0.1:8080`.

For public hosting, publish the static dashboard folder via:

- GitHub Pages (commit and push updated dashboard files on your cron cadence), or
- any web server (Nginx/Caddy/Apache) serving that directory.

Host the Flask rider portal separately, for example on a small VM, PaaS, or internal server. The static dashboard links to the portal; GitHub Pages does not run Flask.

Deployment URL settings:

- Static dashboard -> rider portal: set `COMPANION_APP_BASE_URL` or pass `--companion-app-base-url`.
- Rider portal -> static dashboard: set `FORECAST_DASHBOARD_BASE_URL`.
- For production, set `WIND_DASHBOARD_SECRET_KEY` and usually `WIND_DASHBOARD_COOKIE_SECURE=true`.
- Do not point users at the rider portal's `/forecast-preview` path for the forecast dashboard. That path redirects to `FORECAST_DASHBOARD_BASE_URL` so the portal cannot serve an outdated dashboard copy.

Useful web options:

- `--web-out-dir next_day_wind_model/web_dashboard`
- `--web-out-dir docs` when GitHub Pages is configured to publish the repository `docs/` folder
- `--web-refresh-seconds 900`
- `--companion-app-base-url https://your-rider-portal.example.com`

Example local split:

```bash
python3 -m http.server 8081 -d next_day_wind_model/web_dashboard

FORECAST_DASHBOARD_BASE_URL=http://127.0.0.1:8081 \
COMPANION_APP_BASE_URL=http://127.0.0.1:8080 \
python3 next_day_wind_model/web_dashboard/app.py
```

If your GitHub Pages site is served from `docs/`, use `--web-out-dir docs` in the prediction/update job and set `FORECAST_DASHBOARD_BASE_URL` to the public GitHub Pages URL. Serving an older generated folder locally will show older plot timestamps even though the rider portal itself is current.

Plot notes:
- Date title uses European style (e.g. `1 March 2026`).
- Wind direction is shown below the x-axis for each hour:
  - `F dir` = forecast direction
  - `L dir` = LSTM-predicted direction
- Current-day plot includes actual wind speed up to the present hour and forecast/LSTM for the remaining hours.

## Cadence Probe

Use this to measure when HARMONIE source data actually changes:

```bash
python3 next_day_wind_model/probe_harmonie_cadence.py \
  --iterations 24 \
  --interval-minutes 60 \
  --log-csv next_day_wind_model/artifacts/harmonie_cadence_probe.csv \
  --state-json next_day_wind_model/artifacts/harmonie_cadence_state.json
```

For a one-shot check:

```bash
python3 next_day_wind_model/probe_harmonie_cadence.py
```
