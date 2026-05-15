# KNMI shadow fetch workflow

This workflow archives KNMI HARMONIE P1 forecast vintages for verification
without changing the production Windsurfice fetch, dashboard, or model path.

## What it writes

`scripts/knmi_extract_latest_to_db.py` writes:

- `harmonie_knmi_features`: canonical KNMI feature rows, including 10/50/100/200/300 m wind features;
- `knmi_forecasts_shadow`: a compatibility table shaped like the production `forecasts` table, using KNMI 10 m wind speed converted from m/s to knots.

It does not write to production `forecasts` unless `--write-production` is
explicitly passed. That flag is intended only for manual controlled tests.

## Timestamp semantics

- `run_ts`: KNMI model run parsed from the filename.
- `fetched_ts`: when this worker processed/downloaded the file.
- `target_ts`: `run_ts + horizon_hr`.
- `horizon_hr`: forecast lead time in hours.

Different `run_ts` values for the same `target_ts` are distinct forecast
vintages and must remain distinct. Do not collapse them except in explicit
comparison views that report their selection policy.

## Manual runs

Latest available KNMI tar:

```bash
python3 scripts/knmi_extract_latest_to_db.py
```

Specific filename, suitable for a future Notification Service listener:

```bash
python3 scripts/knmi_extract_latest_to_db.py --filename HARM43_V1_P1_2026051504.tar
```

Existing local tar, useful for offline verification:

```bash
python3 scripts/knmi_extract_latest_to_db.py \
  --tar-path data/raw/knmi/harmonie_arome_cy43_p1/HARM43_V1_P1_2026051504.tar
```

The worker is idempotent: uniqueness keys include source, dataset, run, target,
and site, so repeated runs update existing rows rather than duplicating them.

## Shell wrapper

```bash
bash scripts/run_knmi_shadow_fetch.sh
```

The wrapper:

- changes to the repository root;
- activates `.venv` if present;
- requires `KNMI_API_KEY`;
- does not print the API key;
- appends logs to `logs/knmi_shadow_fetch.log`.

Example hourly cron fallback:

```cron
7 * * * * cd /home/sandervancranenburgh/Documents/repos/wind_fetcher2 && bash scripts/run_knmi_shadow_fetch.sh
```

## Inspection

Recent KNMI runs:

```bash
python3 scripts/knmi_extract_latest_to_db.py --inspect-runs --inspect-limit 10
```

Archive diagnostic:

```bash
python3 scripts/knmi_extract_latest_to_db.py --archive-diagnostic
```

Latest shadow rows:

```bash
python3 scripts/knmi_extract_latest_to_db.py --inspect-shadow --inspect-limit 5
```

Compare latest KNMI shadow and Windsurfice snapshots:

```bash
python3 scripts/compare_knmi_shadow_vs_windsurfice.py
```

Compare the latest KNMI shadow run with the closest available Windsurfice
snapshot by fallback run/fetch timestamp:

```bash
python3 scripts/compare_knmi_shadow_vs_windsurfice.py --windsurfice-policy closest-to-knmi-run
```

The latest-vs-latest comparison can be misleading while only one KNMI run is
archived. A single run is enough for extraction verification, but not enough for
model evaluation or training.

## Future Notification Service

Do not implement the listener here. A future Notification Service/MQTT listener
should call the same worker with `--filename <notified-tar-filename>` so manual,
cron fallback, and notification-triggered ingestion all share one code path.
