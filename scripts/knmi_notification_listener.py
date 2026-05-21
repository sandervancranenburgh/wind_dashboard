#!/usr/bin/env python3
"""Listen for KNMI HARMONIE P1 file notifications and run shadow extraction."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import signal
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.knmi_extract_latest_to_db import process_knmi_file_to_db


HOST = "mqtt.dataplatform.knmi.nl"
PORT = 443
DEFAULT_TOPIC = "dataplatform/file/v1/harmonie_arome_cy43_p1/1.0/created"
FILENAME_PATTERN = re.compile(r"HARM43_V1_P1_\d{10}\.tar")
FILENAME_KEYS = {"filename", "fileName", "name", "key", "path"}
SESSION_EXPIRY_SECONDS = 24 * 60 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to KNMI Notification Service HARMONIE P1 events and update shadow SQLite tables.",
    )
    parser.add_argument("--db", type=Path, default=Path("data/wind_data_all_sites.db"))
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--raw-retention-runs", type=int, default=None)
    parser.add_argument("--cleanup-dry-run", action="store_true")
    parser.add_argument("--once", action="store_true", help="Exit after processing one valid file event.")
    parser.add_argument("--max-events", type=int, default=None, help="Exit after processing N valid file events.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set.")
    return value


def safe_payload_text(payload: bytes, limit: int = 1000) -> str:
    text = payload.decode("utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def values_for_filename_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FILENAME_KEYS and isinstance(child, (str, int, float)):
                found.append(str(child))
            found.extend(values_for_filename_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(values_for_filename_keys(child))
    return found


def extract_filename_from_payload(payload: bytes) -> str | None:
    text = safe_payload_text(payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        for candidate in values_for_filename_keys(parsed):
            match = FILENAME_PATTERN.search(Path(candidate).name) or FILENAME_PATTERN.search(candidate)
            if match:
                return match.group(0)
        logging.info("KNMI notification JSON did not contain an obvious HARMONIE P1 filename: %s", text)

    match = FILENAME_PATTERN.search(text)
    if match:
        return match.group(0)
    if parsed is None:
        logging.info("KNMI notification payload was not JSON and did not contain a HARMONIE P1 filename: %s", text)
    return None


def import_mqtt() -> Any:
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise SystemExit("paho-mqtt is not installed. Install next_day_wind_model/requirements.txt.") from exc
    return mqtt


def mqtt_connect_properties() -> Any | None:
    try:
        from paho.mqtt.packettypes import PacketTypes
        from paho.mqtt.properties import Properties
    except ModuleNotFoundError:
        return None
    properties = Properties(PacketTypes.CONNECT)
    properties.SessionExpiryInterval = SESSION_EXPIRY_SECONDS
    return properties


def make_client(mqtt: Any, client_id: str, api_key: str, topic: str, filenames: "queue.Queue[str]") -> Any:
    seen_lock = threading.Lock()
    seen: set[str] = set()

    try:
        client = mqtt.Client(
            client_id=client_id,
            protocol=mqtt.MQTTv5,
            transport="websockets",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5, transport="websockets")

    client.username_pw_set("token", api_key)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

    def on_connect(client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        logging.info("Connected to KNMI Notification Service: reason=%s flags=%s", reason_code, flags)
        client.subscribe(topic, qos=1)

    def on_subscribe(client: Any, userdata: Any, mid: int, reason_codes: Any, properties: Any = None) -> None:
        logging.info("Subscribed to %s with QoS 1: mid=%s reason=%s", topic, mid, reason_codes)

    def on_disconnect(client: Any, userdata: Any, *args: Any) -> None:
        logging.warning("Disconnected from KNMI Notification Service: %s", args)

    def on_message(client: Any, userdata: Any, message: Any) -> None:
        filename = extract_filename_from_payload(message.payload)
        if filename is None:
            return
        with seen_lock:
            if filename in seen:
                logging.info("Ignoring duplicate notification for %s", filename)
                return
            seen.add(filename)
        logging.info("Queued KNMI HARMONIE P1 notification for %s", filename)
        filenames.put(filename)

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    return client


def connect_client(client: Any) -> None:
    properties = mqtt_connect_properties()
    try:
        client.connect(HOST, PORT, keepalive=60, clean_start=False, properties=properties)
        logging.info("MQTT connect requested with clean_start=False and 24h session expiry.")
    except TypeError:
        client.connect(HOST, PORT, keepalive=60)
        logging.warning("MQTT client did not accept MQTTv5 session options; connected with default session settings.")


def process_filename(filename: str, args: argparse.Namespace) -> bool:
    logging.info("Processing KNMI notification filename=%s", filename)
    try:
        result = process_knmi_file_to_db(
            filename=filename,
            db_path=args.db,
            site=args.site,
            keep_raw=args.keep_raw,
            raw_retention_runs=args.raw_retention_runs,
            cleanup_dry_run=args.cleanup_dry_run,
        )
    except Exception:
        logging.exception("KNMI notification processing failed for %s", filename)
        return False

    diagnostic = result.archive_diagnostic or {}
    logging.info(
        "KNMI notification processed filename=%s run_ts=%s rows_written=%s shadow_rows_written=%s "
        "distinct_knmi_runs=%s latest_run_horizon_count=%s",
        result.filename,
        result.run_ts,
        result.rows_written,
        result.shadow_rows_written,
        diagnostic.get("distinct_run_ts"),
        result.latest_run_horizon_count,
    )
    return True


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.raw_retention_runs is not None and args.raw_retention_runs < 0:
        raise SystemExit("--raw-retention-runs must be zero or greater.")
    if args.max_events is not None and args.max_events < 1:
        raise SystemExit("--max-events must be one or greater.")

    require_env("KNMI_API_KEY")
    notification_key = require_env("KNMI_NOTIFICATION_API_KEY")
    client_id = require_env("KNMI_NOTIFICATION_CLIENT_ID")

    mqtt = import_mqtt()
    filenames: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    max_events = 1 if args.once else args.max_events
    processed_events = 0

    def handle_signal(signum: int, frame: Any) -> None:
        logging.info("Received signal %s, stopping listener.", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    client = make_client(mqtt, client_id, notification_key, args.topic, filenames)
    logging.info("Connecting to %s:%s over MQTT websockets.", HOST, PORT)
    connect_client(client)
    client.loop_start()
    try:
        while not stop_event.is_set():
            try:
                filename = filenames.get(timeout=1.0)
            except queue.Empty:
                continue
            process_filename(filename, args)
            processed_events += 1
            filenames.task_done()
            if max_events is not None and processed_events >= max_events:
                logging.info("Processed %s valid file event(s); exiting.", processed_events)
                stop_event.set()
    finally:
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            logging.exception("Error while disconnecting MQTT client.")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
