# KNMI Notification Service Listener

This repository can run KNMI HARMONIE P1 in shadow mode from notification events. The listener subscribes to the KNMI Notification Service and uses each file-created event only as a trigger. The actual tar download still uses the KNMI Open Data API through `KNMI_API_KEY`.

This does not replace the Windsurfice production fetch. It writes KNMI shadow tables only unless an operator explicitly uses the dangerous production flag on the extractor CLI.

## Environment

Required variables:

```bash
export KNMI_API_KEY="open-data-api-key"
export KNMI_NOTIFICATION_API_KEY="notification-service-api-key"
export KNMI_NOTIFICATION_CLIENT_ID="stable-unique-client-id"
```

Do not print these values in logs or shell history. The MQTT username is always `token`; the Notification Service API key is used as the MQTT password.

Request the Notification Service API key from the KNMI Data Platform account/API-key flow for the Notification Service. It is separate from the Open Data API key used to download files.

## Topic

The HARMONIE P1 file-created topic is:

```text
dataplatform/file/v1/harmonie_arome_cy43_p1/1.0/created
```

The listener connects to `mqtt.dataplatform.knmi.nl:443` using MQTT over websockets with TLS. It requests MQTTv5, QoS 1, a stable client ID, `clean_start=False`, and a 24 hour session expiry when supported by the installed `paho-mqtt` version.

## Manual Listener

```bash
python3 scripts/knmi_notification_listener.py
```

Useful limited runs:

```bash
python3 scripts/knmi_notification_listener.py --once --log-level INFO
python3 scripts/knmi_notification_listener.py --max-events 3 --log-level INFO
```

Wrapper with log append:

```bash
scripts/run_knmi_notification_listener.sh
tail -f logs/knmi_notification_listener.log
```

The wrapper checks that `KNMI_API_KEY`, `KNMI_NOTIFICATION_API_KEY`, and `KNMI_NOTIFICATION_CLIENT_ID` are set, then runs the listener from the repository root.

## Operational Setup After Validation

The direct tmux command has been validated end-to-end:

```bash
python scripts/knmi_notification_listener.py --log-level INFO
```

That run confirmed MQTT connection, persistent session replay, QoS 1 subscription, event receipt, filename parsing, file-specific processing, SQLite writes, and a complete 61-horizon archive. For ongoing operation, prefer the wrapper so stdout/stderr are appended to `logs/knmi_notification_listener.log`.

Stop any existing direct listener first:

```bash
pkill -f "scripts/knmi_notification_listener.py"
```

Start the wrapper in tmux:

```bash
tmux new -s knmi_notify
cd ~/Documents/repos/wind_fetcher2_dev
source ~/.bashrc
source /home/sandervancranenburgh/Documents/python_envs/env/bin/activate
bash scripts/run_knmi_notification_listener.sh
```

Detach from tmux with `Ctrl+B`, then `D`.

Monitor the running listener:

```bash
ps aux | grep knmi_notification_listener | grep -v grep
tail -f ~/Documents/repos/wind_fetcher2_dev/logs/knmi_notification_listener.log
```

Run the health check:

```bash
python scripts/diagnose_knmi_notification_lag.py \
  --db data/wind_data_all_sites.db \
  --site valkenburgsemeer \
  --latest-api-count 12 \
  --health-check
```

Confirm latest archive horizon counts directly:

```bash
sqlite3 data/wind_data_all_sites.db "
SELECT run_ts, COUNT(*) AS n, MIN(horizon_hr), MAX(horizon_hr)
FROM harmonie_knmi_features
GROUP BY run_ts
ORDER BY run_ts DESC
LIMIT 10;
"
```

The setup is healthy when the latest KNMI API run equals the latest archived DB run and the latest DB runs have 61 horizons each.

## Fallback Polling

Use the fallback job if the listener was down or may have missed events. It processes recent files idempotently through the existing extractor:

```bash
python3 scripts/knmi_extract_latest_to_db.py --latest-count 3
scripts/run_knmi_shadow_fetch_fallback.sh
tail -f logs/knmi_shadow_fetch_fallback.log
```

The wrapper reads `KNMI_FALLBACK_LATEST_COUNT` and defaults to `3`.

Catch up a wider window manually:

```bash
python3 scripts/knmi_extract_latest_to_db.py --latest-count 6
```

Example crontab line for a 30 minute fallback cadence:

```cron
*/30 * * * * cd /home/sandervancranenburgh/Documents/repos/wind_fetcher2_dev && scripts/run_knmi_shadow_fetch_fallback.sh
```

Do not edit crontab blindly. While the notification/fallback code exists only on the dev branch, use the dev clone. After this branch is merged to production `main`, switch the cron line to the production clone.

Before merge:

```cron
*/30 * * * * cd /home/sandervancranenburgh/Documents/repos/wind_fetcher2_dev && /home/sandervancranenburgh/Documents/python_envs/env/bin/python scripts/knmi_extract_latest_to_db.py --latest-count 3 >> /home/sandervancranenburgh/Documents/repos/wind_fetcher2_dev/logs/knmi_shadow_fetch_fallback.log 2>&1
```

After merge to main:

```cron
*/30 * * * * cd /home/sandervancranenburgh/Documents/repos/wind_fetcher2 && /home/sandervancranenburgh/Documents/python_envs/env/bin/python scripts/knmi_extract_latest_to_db.py --latest-count 3 >> /home/sandervancranenburgh/Documents/repos/wind_fetcher2/logs/knmi_shadow_fetch_fallback.log 2>&1
```

## Optional Systemd User Service

Example only; do not install blindly:

```ini
[Unit]
Description=KNMI HARMONIE P1 notification listener
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/sandervancranenburgh/Documents/repos/wind_fetcher2_dev
Environment=KNMI_API_KEY=replace-with-open-data-key
Environment=KNMI_NOTIFICATION_API_KEY=replace-with-notification-key
Environment=KNMI_NOTIFICATION_CLIENT_ID=wind-fetcher-knmi-shadow
ExecStart=/home/sandervancranenburgh/Documents/repos/wind_fetcher2_dev/scripts/run_knmi_notification_listener.sh
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
```

Prefer an environment file with restricted permissions for real secrets.

## Troubleshooting Lag

Use the lag diagnostic first. By default it only lists KNMI Open Data API files, reads SQLite, parses logs, and tests the notification payload parser. It does not download or process files unless `--process-missing-latest` is set.

```bash
python3 scripts/diagnose_knmi_notification_lag.py \
  --db data/wind_data_all_sites.db \
  --site valkenburgsemeer \
  --latest-api-count 12
```

For a concise operational check:

```bash
python3 scripts/diagnose_knmi_notification_lag.py \
  --db data/wind_data_all_sites.db \
  --site valkenburgsemeer \
  --latest-api-count 12 \
  --health-check
```

The key comparison is the latest API `run_ts` versus the archive `max_run_ts`. Both are UTC internally. The diagnostic also prints Europe/Amsterdam display times using the IANA timezone, so summer time is shown as CEST/UTC+2 without manual offsets.

For a quick DB-only archive summary:

```bash
python3 scripts/knmi_extract_latest_to_db.py --archive-diagnostic
```

Observation joinability counts are slower and opt-in:

```bash
python3 scripts/knmi_extract_latest_to_db.py --archive-diagnostic --include-observation-joinability
```

Interpretation:

- Latest API run equals latest DB run, with 61 rows: KNMI is caught up.
- API has newer files than DB, and listener log has no event/parsed filename: likely listener was not running, not logging to the wrapper file, or missed notification replay.
- API has newer files than DB, listener parsed/queued the filename, but processing failed: inspect the listener error and run fallback.
- API has newer files than DB, listener received payloads but parsed no filename: payload format changed; use the logged payload preview to update parser handling.
- API itself has no newer file: KNMI has not published the next tar yet. Compare API `created` time to `run_ts`; HARMONIE files can be created well after the model run hour.

Fallback catch-up options:

```bash
python3 scripts/knmi_extract_latest_to_db.py --latest-count 6
python3 scripts/diagnose_knmi_notification_lag.py --process-missing-latest 6
```

The diagnostic catch-up mode processes only the latest missing or incomplete API files, using the same idempotent shadow worker as the listener.
