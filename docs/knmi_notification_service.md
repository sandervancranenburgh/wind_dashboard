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

## Fallback Polling

Use the fallback job if the listener was down or may have missed events. It processes recent files idempotently through the existing extractor:

```bash
python3 scripts/knmi_extract_latest_to_db.py --latest-count 3
scripts/run_knmi_shadow_fetch_fallback.sh
tail -f logs/knmi_shadow_fetch_fallback.log
```

The wrapper reads `KNMI_FALLBACK_LATEST_COUNT` and defaults to `3`.

Example crontab line for a 30 minute fallback cadence:

```cron
*/30 * * * * cd /home/sandervancranenburgh/Documents/repos/wind_fetcher2_dev && scripts/run_knmi_shadow_fetch_fallback.sh
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
