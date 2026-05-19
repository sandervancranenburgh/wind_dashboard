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

## Raw data cleanup

KNMI HARMONIE P1 tar files are large, around 900 MB per run. The SQLite database
is the long-term forecast archive; the raw tar directory is only a download
staging area. GRIB members are extracted into a temporary directory and should
not persist after processing.

By default, after successful extraction and successful writes to
`harmonie_knmi_features` and `knmi_forecasts_shadow`, the worker deletes the
processed tar file:

```bash
python3 scripts/knmi_extract_latest_to_db.py
```

For debugging, keep the processed tar:

```bash
python3 scripts/knmi_extract_latest_to_db.py --keep-raw
```

To retain only the latest N matching raw HARMONIE P1 tar files in `--raw-dir`:

```bash
python3 scripts/knmi_extract_latest_to_db.py --raw-retention-runs 2
```

When `--raw-retention-runs N` is supplied, retention controls which
`HARM43_V1_P1_*.tar` files remain. When combined with `--keep-raw`, the current
processed tar is retained even if it is older than the latest N retained runs.
Cleanup only runs after successful database writes and only deletes regular
files whose names match the expected KNMI HARMONIE P1 tar pattern. Use
`--cleanup-dry-run` to print cleanup actions without deleting files. Cleanup
does not delete SQLite databases, processed CSVs, model artifacts, dashboard
outputs, or production Windsurfice data.

## Shell wrapper

```bash
bash scripts/run_knmi_shadow_fetch.sh
```

The wrapper:

- changes to the repository root;
- activates `.venv` if present;
- requires `KNMI_API_KEY`;
- does not print the API key;
- uses the operational default raw cleanup policy, deleting the processed tar
  after a successful DB write;
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
