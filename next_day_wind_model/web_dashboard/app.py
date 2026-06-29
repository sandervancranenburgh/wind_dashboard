from __future__ import annotations

import base64
import csv
import functools
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from zoneinfo import ZoneInfo

from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import db_store
from wingfoil_analysis import analyze_session_file, build_wind_context


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "next_day_wind_model" / "artifacts"
CURRENT_DAY_PLOT_ARCHIVE_DIR = ARTIFACTS_DIR / "current_day_plot_archive"
LOCAL_TZ = ZoneInfo(os.environ.get("WIND_DASHBOARD_TZ", "Europe/Amsterdam"))
COMPANION_APP_BASE_URL = os.environ.get("COMPANION_APP_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
FORECAST_DASHBOARD_BASE_URL = os.environ.get("FORECAST_DASHBOARD_BASE_URL", "http://127.0.0.1:8081").rstrip("/")
SPOT_OPTIONS = [
    "Valkenburgse meer",
    "Oostvoornse meer",
    "Brouwersdam",
    "Noord Aa",
    "Other",
]
WING_SIZE_OPTIONS = [2, 3, 4, 5, 6, 7, 8]
FOIL_SIZE_OPTIONS = list(range(700, 2501, 100))
SESSION_TIME_OPTIONS = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in (0, 30)]
PERCEIVED_WIND_VARIABILITY_OPTIONS = [
    ("very_steady", "Very steady"),
    ("steady", "Steady"),
    ("moderate", "Moderate"),
    ("gusty", "Gusty"),
    ("very_gusty", "Very gusty"),
]
PERCEIVED_WIND_VARIABILITY_LABELS = dict(PERCEIVED_WIND_VARIABILITY_OPTIONS)
PERCEIVED_WIND_VARIABILITY_VALUES = set(PERCEIVED_WIND_VARIABILITY_LABELS)
SORT_OPTIONS = {
    "date",
    "visibility",
    "rider",
    "spot",
    "start_time",
    "end_time",
    "session_rating",
    "wing_size",
    "foil_size",
    "avg_measured_wind_speed",
    "max_measured_wind_speed",
    "min_measured_wind_speed",
    "max_measured_wind_gust",
    "perceived_wind_variability",
    "wind_variability",
    "mean_measured_direction",
    "avg_forecast_temperature",
}
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_ACTIVITY_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ACTIVITY_ZIP_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
ACTIVITY_UPLOAD_EXTENSIONS = {".fit", ".tcx", ".gpx", ".kml", ".zip"}
ACTIVITY_ZIP_INNER_EXTENSIONS = ACTIVITY_UPLOAD_EXTENSIONS - {".zip"}
ACTIVITY_UPLOAD_ACCEPT = (
    ".fit,.FIT,.tcx,.TCX,.gpx,.GPX,.kml,.KML,.zip,.ZIP,"
    "application/xml,text/xml,application/octet-stream,application/zip,"
    "application/x-zip-compressed,application/gpx+xml,application/vnd.google-earth.kml+xml"
)
ACTIVITY_UPLOAD_FIELD = "ActivityFile"
ACTIVITY_UPLOAD_FORMATS = "FIT, TCX, GPX, KML, or ZIP"
ACTIVITY_ARTIFACT_LABELS = {
    "map_svg": "Session map",
    "run_distance_distribution_svg": "Run distance distribution",
    "run_speed_distribution_svg": "Run speed distribution",
    "run_wind_angle_distribution_svg": "Run wind angle distribution",
    "run_speed_profile_svg": "Run speed profile",
}



def _load_secret_key() -> str:
    env_value = os.environ.get("WIND_DASHBOARD_SECRET_KEY")
    if env_value:
        return env_value

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    secret_path = DATA_DIR / ".wind_dashboard_secret"
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()

    secret = secrets.token_urlsafe(48)
    secret_path.write_text(secret, encoding="utf-8")
    try:
        secret_path.chmod(0o600)
    except OSError:
        pass
    return secret


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=None,
)
app.secret_key = _load_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("WIND_DASHBOARD_COOKIE_SECURE", "").lower()
    in {"1", "true", "yes"},
    MAX_CONTENT_LENGTH=max(MAX_ACTIVITY_UPLOAD_BYTES, int(os.environ.get("WIND_DASHBOARD_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))),
)


def _connect_db():
    conn = db_store.connect_db(str(DATA_DIR))
    conn.row_factory = None
    db_store.init_account_db(conn)
    return conn


def get_db():
    if "db" not in g:
        g.db = _connect_db()
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    iterations = 390_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt_raw, digest_raw = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_raw.encode("ascii"))
        expected = base64.b64decode(digest_raw.encode("ascii"))
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _validate_csrf() -> None:
    token = session.get("_csrf_token")
    submitted = request.form.get("_csrf_token")
    if not token or not submitted or not hmac.compare_digest(token, submitted):
        abort(400)


def _safe_next_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("/") and not value.startswith("//"):
        return value
    return None


def _is_valid_email(value: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(value.strip()))


@app.context_processor
def inject_globals():
    return {
        "current_user": current_user(),
        "current_profile": current_profile(),
        "csrf_token": _csrf_token,
        "companion_app_base_url": COMPANION_APP_BASE_URL,
        "forecast_dashboard_base_url": FORECAST_DASHBOARD_BASE_URL,
        "perceived_wind_variability_labels": PERCEIVED_WIND_VARIABILITY_LABELS,
    }


def current_user() -> dict[str, Any] | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db_store.get_user_by_id(get_db(), int(user_id))


def current_profile() -> dict[str, Any] | None:
    user = current_user()
    if not user:
        return None
    return db_store.get_user_profile(get_db(), int(user["id"]))


def _effective_profile_rider_name(profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    return (profile.get("rider_name") or profile.get("public_username") or "").strip()


def _profile_form_values(profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = profile or {}
    return {
        "public_username": profile.get("public_username") or "",
        "rider_name": _effective_profile_rider_name(profile),
        "rider_weight": profile.get("rider_weight"),
        "default_spot": profile.get("default_spot") or "",
    }


def login_required(view: Callable):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            flash("Please log in first.", "error")
            return redirect(url_for("portal_home", login=1, next=request.path))
        return view(*args, **kwargs)

    return wrapped


def _parse_int(value: str | None, field: str, errors: list[str], required: bool = True) -> int | None:
    if value is None or value.strip() == "":
        if required:
            errors.append(f"{field} is required.")
        return None
    try:
        return int(value)
    except ValueError:
        errors.append(f"{field} must be a whole number.")
        return None


def _parse_session_time(value: str) -> tuple[int, int] | None:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (AttributeError, ValueError):
        return None
    if 0 <= hour <= 23 and minute in {0, 30}:
        return hour, minute
    return None


def _local_session_bounds(day: str, start_time: str, end_time: str) -> tuple[int | None, int | None]:
    try:
        start_parts = _parse_session_time(start_time)
        end_parts = _parse_session_time(end_time)
        if start_parts is None or end_parts is None:
            return None, None
        session_date = date.fromisoformat(day)
        start_dt = datetime.combine(session_date, datetime.min.time(), tzinfo=LOCAL_TZ).replace(
            hour=start_parts[0],
            minute=start_parts[1],
        )
        end_dt = datetime.combine(session_date, datetime.min.time(), tzinfo=LOCAL_TZ).replace(
            hour=end_parts[0],
            minute=end_parts[1],
        )
    except (ValueError, TypeError):
        return None, None
    return (
        int(start_dt.astimezone(timezone.utc).timestamp() * 1000),
        int(end_dt.astimezone(timezone.utc).timestamp() * 1000),
    )


def _experience_form_defaults() -> dict[str, Any]:
    profile = current_profile() or {}
    start_time = "12:00"
    return {
        "Rider": _effective_profile_rider_name(profile),
        "Spot": profile.get("default_spot") or "Valkenburgse meer",
        "Date": datetime.now(LOCAL_TZ).date().isoformat(),
        "StartTime": start_time,
        "EndTime": "14:00",
        "SessionRating": "3",
        "PerceivedWindVariability": "moderate",
        "RiderReview": "",
        "RiderWeight": "" if profile.get("rider_weight") is None else str(profile.get("rider_weight")),
        "WingSize": "5",
        "FoilSize": "1500",
        "RiderNotes": "",
        "Visibility": "private",
    }


def _validate_experience_form(form: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rider = form.get("Rider", "").strip()
    spot = form.get("Spot", "")
    day = form.get("Date", "")
    start_time = form.get("StartTime", "")
    end_time = form.get("EndTime", "")
    rating = _parse_int(form.get("SessionRating"), "SessionRating", errors)
    rider_weight = _parse_int(form.get("RiderWeight"), "RiderWeight", errors)
    wing_size = _parse_int(form.get("WingSize"), "WingSize", errors)
    foil_size = _parse_int(form.get("FoilSize"), "FoilSize", errors)
    visibility = (form.get("Visibility") or "private").strip().lower()
    perceived_wind_variability = (form.get("PerceivedWindVariability") or "").strip().lower()

    if not rider:
        errors.append("Rider is required.")
    if spot not in SPOT_OPTIONS:
        errors.append("Spot must be one of the allowed options.")
    try:
        date.fromisoformat(day)
    except ValueError:
        errors.append("Date must be valid.")
    if start_time not in SESSION_TIME_OPTIONS:
        errors.append("StartTime must be a half-hour time.")
    if end_time not in SESSION_TIME_OPTIONS:
        errors.append("EndTime must be a half-hour time.")
    if start_time in SESSION_TIME_OPTIONS and end_time in SESSION_TIME_OPTIONS and end_time < start_time:
        errors.append("EndTime cannot be earlier than StartTime.")
    if rating is not None and not 1 <= rating <= 5:
        errors.append("SessionRating must be between 1 and 5.")
    if rider_weight is not None and rider_weight <= 0:
        errors.append("RiderWeight must be greater than zero.")
    if wing_size is not None and wing_size not in WING_SIZE_OPTIONS:
        errors.append("WingSize must be one of the allowed options.")
    if foil_size is not None and foil_size not in FOIL_SIZE_OPTIONS:
        errors.append("FoilSize must be one of the allowed options.")
    if visibility not in {"private", "public"}:
        errors.append("Visibility must be private or public.")
    if perceived_wind_variability not in PERCEIVED_WIND_VARIABILITY_VALUES:
        errors.append("PerceivedWindVariability must be one of the allowed options.")

    start_ts, end_ts = _local_session_bounds(day, start_time, end_time)
    if start_ts is None or end_ts is None:
        errors.append("Date and time range must be valid.")

    return (
        {
            "rider": rider,
            "spot": spot,
            "date": day,
            "start_time": start_time,
            "end_time": end_time,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "session_rating": rating,
            "perceived_wind_variability": perceived_wind_variability,
            "rider_review": form.get("RiderReview", "").strip(),
            "rider_weight": rider_weight,
            "wing_size": wing_size,
            "foil_size": foil_size,
            "rider_notes": form.get("RiderNotes", "").strip(),
            "visibility": visibility,
        },
        errors,
    )


def _experience_form_values_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Rider": row.get("rider") or "",
        "Spot": row.get("spot") or "Valkenburgse meer",
        "Date": row.get("date") or datetime.now(LOCAL_TZ).date().isoformat(),
        "StartTime": row.get("start_time") or "12:00",
        "EndTime": row.get("end_time") or "14:00",
        "SessionRating": "" if row.get("session_rating") is None else str(row.get("session_rating")),
        "PerceivedWindVariability": row.get("perceived_wind_variability") or "",
        "RiderReview": row.get("rider_review") or "",
        "RiderWeight": "" if row.get("rider_weight") is None else str(row.get("rider_weight")),
        "WingSize": "" if row.get("wing_size") is None else str(row.get("wing_size")),
        "FoilSize": "" if row.get("foil_size") is None else str(row.get("foil_size")),
        "RiderNotes": row.get("rider_notes") or "",
        "Visibility": row.get("visibility") or "private",
    }


def _dashboard_asset_url(filename: str) -> str:
    return url_for("dashboard_asset", filename=filename)


@app.template_filter("overview_date")
def _format_overview_date(value: str) -> str:
    try:
        day = date.fromisoformat(value)
    except (TypeError, ValueError):
        return value or ""
    return day.strftime("%d-%m-%Y")


@app.template_filter("long_session_date")
def _format_long_session_date(value: str) -> str:
    try:
        day = date.fromisoformat(value)
    except (TypeError, ValueError):
        return value or ""
    return f"{day.strftime('%A')} {day.day} {day.strftime('%B %Y')}"


def _current_day_archive_plot_for_submission(submission_date: str) -> dict[str, str] | None:
    try:
        day = datetime.strptime(submission_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    if not CURRENT_DAY_PLOT_ARCHIVE_DIR.exists():
        return None

    stable_path = CURRENT_DAY_PLOT_ARCHIVE_DIR / f"current_day_predictions_{day.isoformat()}.png"
    if stable_path.exists():
        path = stable_path
    else:
        prefix = day.strftime("%Y%m%d")
        matches = sorted(CURRENT_DAY_PLOT_ARCHIVE_DIR.glob(f"{prefix}-*_current_day_predictions.png"))
        if not matches:
            return None
        path = matches[-1]
    return {
        "filename": path.name,
        "url": url_for("current_day_archive_asset", filename=path.name),
    }


class ActivityUploadError(ValueError):
    pass


def _activity_upload_root() -> Path:
    return Path(app.config.get("RIDER_ACTIVITY_UPLOAD_DIR") or (DATA_DIR / "rider_activity_uploads"))


def _activity_analysis_root() -> Path:
    return Path(app.config.get("RIDER_ACTIVITY_ANALYSIS_DIR") or (DATA_DIR / "rider_activity_analysis"))


def _activity_upload_dir(experience_id: int) -> Path:
    return _activity_upload_root() / str(int(experience_id))


def _activity_output_dir(experience_id: int) -> Path:
    return _activity_analysis_root() / str(int(experience_id))


def _ensure_child_path(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ActivityUploadError("Activity file path is outside the allowed directory.") from exc


def _uploaded_activity_file():
    uploaded = request.files.get(ACTIVITY_UPLOAD_FIELD)
    if uploaded is None or not (uploaded.filename or "").strip():
        return None
    return uploaded


def _unsupported_activity_message() -> str:
    return f"Unsupported activity file. Please upload a {ACTIVITY_UPLOAD_FORMATS} file."


def _activity_too_large_message() -> str:
    limit_mb = MAX_ACTIVITY_UPLOAD_BYTES // (1024 * 1024)
    return f"Activity file is too large. Please upload a file up to {limit_mb} MB."


def _uploaded_stream_size(uploaded) -> int | None:
    stream = getattr(uploaded, "stream", None)
    if stream is not None:
        try:
            position = stream.tell()
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(position)
            return int(size)
        except (OSError, ValueError):
            try:
                stream.seek(0)
            except (OSError, ValueError):
                pass
    content_length = getattr(uploaded, "content_length", None)
    if content_length is None:
        return None
    try:
        return int(content_length)
    except (TypeError, ValueError):
        return None


def _reset_uploaded_stream(uploaded) -> None:
    stream = getattr(uploaded, "stream", None)
    if stream is None:
        return
    try:
        stream.seek(0)
    except (OSError, ValueError):
        pass


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _zip_member_has_unsafe_path(info: zipfile.ZipInfo) -> bool:
    name = info.filename
    if not name or info.is_dir() or "\\" in name:
        return True
    member_path = PurePosixPath(name)
    if member_path.is_absolute() or len(member_path.parts) != 1:
        return True
    if any(part in {"", ".", ".."} for part in member_path.parts):
        return True
    return _zip_member_is_symlink(info)


def _validate_activity_zip_upload(uploaded) -> None:
    stream = getattr(uploaded, "stream", None)
    if stream is None:
        raise ActivityUploadError("ZIP uploads must contain exactly one supported activity file.")
    position = None
    try:
        position = stream.tell()
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            if any(_zip_member_has_unsafe_path(info) for info in infos):
                raise ActivityUploadError("ZIP upload rejected because it contains unsafe paths or is too large when extracted.")
            file_infos = [info for info in infos if not info.is_dir()]
            total_uncompressed = sum(info.file_size for info in file_infos)
            total_compressed = sum(info.compress_size for info in file_infos)
            if total_uncompressed > MAX_ACTIVITY_ZIP_UNCOMPRESSED_BYTES or total_compressed > MAX_ACTIVITY_UPLOAD_BYTES:
                raise ActivityUploadError("ZIP upload rejected because it contains unsafe paths or is too large when extracted.")
            supported_files = [
                info
                for info in file_infos
                if PurePosixPath(info.filename).suffix.lower() in ACTIVITY_ZIP_INNER_EXTENSIONS
            ]
            if len(file_infos) != 1 or len(supported_files) != 1:
                raise ActivityUploadError("ZIP uploads must contain exactly one supported activity file.")
    except ActivityUploadError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ActivityUploadError("ZIP uploads must contain exactly one supported activity file.") from exc
    finally:
        try:
            stream.seek(position or 0)
        except (OSError, ValueError):
            pass


def _save_activity_upload(uploaded, experience_id: int) -> dict[str, Any]:
    original_filename = (uploaded.filename or "").strip()
    safe_name = secure_filename(original_filename)
    if not safe_name:
        raise ActivityUploadError(_unsupported_activity_message())
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ACTIVITY_UPLOAD_EXTENSIONS:
        raise ActivityUploadError(_unsupported_activity_message())

    upload_size = _uploaded_stream_size(uploaded)
    if upload_size is not None and upload_size > MAX_ACTIVITY_UPLOAD_BYTES:
        raise ActivityUploadError(_activity_too_large_message())
    if suffix == ".zip":
        _validate_activity_zip_upload(uploaded)

    upload_dir = _activity_upload_dir(experience_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(8)}{suffix}"
    stored_path = upload_dir / stored_filename
    _ensure_child_path(stored_path, upload_dir)
    _reset_uploaded_stream(uploaded)
    uploaded.save(stored_path)
    if stored_path.stat().st_size > MAX_ACTIVITY_UPLOAD_BYTES:
        stored_path.unlink(missing_ok=True)
        raise ActivityUploadError(_activity_too_large_message())
    return {
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "stored_path": stored_path,
        "file_type": suffix.lstrip("."),
    }


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_safe_artifact_name(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    artifact_path = Path(value)
    return not artifact_path.is_absolute() and artifact_path.name == value and ".." not in artifact_path.parts


def _registered_activity_artifacts(payload: dict[str, Any]) -> dict[str, str]:
    registered: dict[str, str] = {}
    for key in ("summary_json", "runs_csv", "map_svg", "map_html"):
        value = payload.get(key)
        if _is_safe_artifact_name(value):
            registered[key] = str(value)
    for group_name in ("artifacts", "plots"):
        group = payload.get(group_name)
        if not isinstance(group, dict):
            continue
        for key, value in group.items():
            if _is_safe_artifact_name(value):
                registered[str(key)] = str(value)
    return registered


def _build_wind_context(row: dict[str, Any]) -> tuple[Any, list[str]]:
    measured = row.get("measured_wind") or {}
    summary = measured.get("summary") if isinstance(measured, dict) else {}
    summary = summary or {}
    direction = row.get("mean_measured_direction")
    if direction is None:
        direction = summary.get("mean_wind_dir")
    speed = row.get("avg_measured_wind_speed")
    if speed is None:
        speed = summary.get("avg_wind_speed")

    warnings: list[str] = []
    if speed is None and direction is None:
        warnings.append("Wind context was unavailable for this activity analysis.")
    wind_context = build_wind_context(
        wind_direction_deg=direction,
        wind_speed_kts=speed,
        spot_name=row.get("spot"),
    )
    return wind_context, warnings


def _format_stat_value(value: Any, suffix: str = "", digits: int = 1) -> str | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits <= 0:
        formatted = f"{number:.0f}"
    else:
        formatted = f"{number:.{digits}f}"
    return f"{formatted} {suffix}".strip()


def _activity_summary_items(analysis: dict[str, Any]) -> list[dict[str, str]]:
    stats = analysis.get("stats") or {}
    summary = analysis.get("summary") or {}
    activity = summary.get("activity") if isinstance(summary, dict) else {}
    activity = activity or {}
    items = [
        ("Distance", _format_stat_value(stats.get("distance_km") or activity.get("total_distance_m") and float(activity["total_distance_m"]) / 1000, "km", 2)),
        ("Max speed", _format_stat_value(stats.get("max_speed_kmh"), "km/h", 1)),
        ("Avg foil speed", _format_stat_value(stats.get("avg_speed_on_foil_kmh") or activity.get("avg_speed_on_foil_kmh"), "km/h", 1)),
        ("Runs", _format_stat_value(stats.get("run_count") or (summary.get("runs_summary") or {}).get("count"), "", 0)),
        ("Falls", _format_stat_value(stats.get("fall_count") or (summary.get("falls_summary") or {}).get("count"), "", 0)),
        ("Track points", _format_stat_value(stats.get("track_point_count") or activity.get("sample_count"), "", 0)),
    ]
    water_time = activity.get("water_time_formatted")
    if water_time:
        items.insert(1, ("Water time", str(water_time)))
    return [{"label": label, "value": value} for label, value in items if value]


def _read_activity_runs(analysis: dict[str, Any], output_dir: Path) -> list[dict[str, str]]:
    runs_name = (analysis.get("artifacts") or {}).get("runs_csv")
    if not _is_safe_artifact_name(runs_name):
        return []
    runs_path = output_dir / str(runs_name)
    _ensure_child_path(runs_path, output_dir)
    if not runs_path.is_file():
        return []
    rows: list[dict[str, str]] = []
    try:
        with runs_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                distance_m = row.get("distance_m") or ""
                if not distance_m and row.get("distance_km"):
                    try:
                        distance_m = str(round(float(row["distance_km"]) * 1000))
                    except (TypeError, ValueError):
                        distance_m = ""
                else:
                    try:
                        distance_m = str(round(float(distance_m))) if distance_m else ""
                    except (TypeError, ValueError):
                        distance_m = str(distance_m)
                def one_decimal(value: str | None) -> str:
                    if value is None or value == "":
                        return ""
                    try:
                        return f"{float(value):.1f}"
                    except (TypeError, ValueError):
                        return str(value)

                rows.append(
                    {
                        "run_id": row.get("run_id") or "",
                        "distance_m": distance_m,
                        "distance_km": row.get("distance_km") or "",
                        "mean_speed_kmh": one_decimal(row.get("mean_speed_kmh")),
                        "max_speed_kmh": one_decimal(row.get("max_speed_kmh")),
                        "wind_angle_class": row.get("wind_angle_class") or "",
                    }
                )
    except OSError:
        return []
    return rows


def _activity_warning_for_display(warning: Any) -> str:
    text = str(warning or "").strip()
    lower = text.lower()
    if (
        "timestamps were somewhat irregular" in lower
        or "timestamps are somewhat irregular" in lower
        or "timestamps are irregular" in lower
    ):
        return "GPS timestamps were somewhat irregular; speed and distance were reconstructed where needed."
    if "speed had to be reconstructed" in lower:
        return "GPS speed was missing for some points, so speed was reconstructed from the GPS track."
    if "distance had to be reconstructed" in lower:
        return "GPS distance was missing for some points, so distance was reconstructed from the GPS track."
    return text


def _activity_warnings_for_display(warnings: Any) -> list[str]:
    if isinstance(warnings, str):
        raw_warnings = [warnings]
    elif isinstance(warnings, list):
        raw_warnings = warnings
    else:
        raw_warnings = []
    display: list[str] = []
    seen: set[str] = set()
    for warning in raw_warnings:
        text = _activity_warning_for_display(warning)
        if text and text not in seen:
            seen.add(text)
            display.append(text)
    return display


def _activity_analysis_view_model(
    analysis: dict[str, Any] | None,
    artifact_url_builder: Callable[[str], str | None] | None = None,
) -> dict[str, Any] | None:
    if not analysis:
        return None
    output_dir = _activity_output_dir(int(analysis["experience_id"]))
    artifacts = analysis.get("artifacts") or {}

    def artifact_url(key: str) -> str | None:
        filename = artifacts.get(key)
        if not _is_safe_artifact_name(filename):
            return None
        if artifact_url_builder is not None:
            return artifact_url_builder(str(filename))
        return url_for("experience_activity_artifact", experience_id=analysis["experience_id"], filename=filename)

    plot_urls = []
    for key, label in ACTIVITY_ARTIFACT_LABELS.items():
        url = artifact_url(key)
        if url:
            plot_urls.append({"key": key, "label": label, "url": url})
    return {
        "status": analysis.get("status"),
        "original_filename": analysis.get("original_filename"),
        "file_type": analysis.get("file_type"),
        "uploaded_at": analysis.get("uploaded_at"),
        "analysis_version": analysis.get("analysis_version"),
        "warnings": _activity_warnings_for_display(analysis.get("warnings")),
        "errors": analysis.get("errors") or [],
        "summary_items": _activity_summary_items(analysis),
        "map_html_url": artifact_url("map_html"),
        "map_svg_url": artifact_url("map_svg"),
        "plot_urls": plot_urls,
        "runs": _read_activity_runs(analysis, output_dir),
    }



def _map_html_selection_shim() -> str:
    return """
<script>
(() => {
  try {
    if (typeof data === "undefined" || typeof lineLayer === "undefined") return;
    const layers = typeof lineLayer.getLayers === "function" ? lineLayer.getLayers() : [];
    const runs = Array.isArray(data.runs) ? data.runs : [];
    const segments = Array.isArray(data.segments) ? data.segments : [];
    const runIdForSegment = (segmentIndex) => {
      const sampleIndex = segmentIndex + 1;
      const run = runs.find((candidate) => sampleIndex >= Number(candidate.start_index) && sampleIndex <= Number(candidate.end_index));
      return run && run.run_id !== undefined ? String(run.run_id) : "";
    };
    const segmentLayers = segments.map((segment, index) => ({
      line: layers[index],
      inRun: Boolean(segment.in_run),
      run_id: String(segment.run_id || runIdForSegment(index)),
      speedKmh: Number(segment.speed_kmh),
    })).filter((item) => item.line);
    for (const item of segmentLayers) {
      if (!item.inRun) continue;
      item.line.bindTooltip(`Run ${item.run_id}<br>Speed: ${Number.isFinite(item.speedKmh) ? item.speedKmh.toFixed(1) : "n/a"} km/h`, { sticky: true });
    }
    window.wingfoilRunLayers = segmentLayers;
    window.applyWingfoilRunSelection = (selectedRunIds) => {
      const selected = new Set((selectedRunIds || []).map((runId) => String(runId)).filter((runId) => runId !== ""));
      const showAllRuns = selected.size === 0;
      for (const item of segmentLayers) {
        const shouldShow = !item.inRun || showAllRuns || selected.has(item.run_id);
        const isVisible = lineLayer.hasLayer(item.line);
        if (shouldShow && !isVisible) item.line.addTo(lineLayer);
        if (!shouldShow && isVisible) lineLayer.removeLayer(item.line);
      }
      if (window.__wingfoilMapDebug) window.__wingfoilMapDebug.selectedRunIds = Array.from(selected);
    };
    window.addEventListener("message", (event) => {
      const payload = event.data || {};
      if (payload.type !== "wingfoil-run-selection") return;
      window.applyWingfoilRunSelection(payload.selectedRunIds || payload.runIds || []);
    });

    if (typeof fallLayer !== "undefined") {
      const fallMarkers = typeof fallLayer.getLayers === "function" ? fallLayer.getLayers() : [];
      for (const marker of fallMarkers) {
        if (typeof marker.setStyle === "function") {
          marker.setStyle({ radius: 4, color: "rgba(127, 29, 29, 0.72)", weight: 1.2, fillColor: "#ef4444", fillOpacity: 0.62, opacity: 0.72 });
        }
      }
      let fallsVisible = true;
      window.setWingfoilFallsVisible = (visible) => {
        fallsVisible = Boolean(visible);
        if (fallsVisible) {
          if (!map.hasLayer(fallLayer)) fallLayer.addTo(map);
        } else if (map.hasLayer(fallLayer)) {
          map.removeLayer(fallLayer);
        }
        const button = document.querySelector(".falls-toggle");
        if (button) {
          button.classList.toggle("is-off", !fallsVisible);
          button.setAttribute("aria-pressed", fallsVisible ? "true" : "false");
        }
        if (window.__wingfoilMapDebug) window.__wingfoilMapDebug.fallsVisible = fallsVisible;
      };
      if (!document.querySelector(".falls-toggle") && typeof L !== "undefined") {
        const fallsToggle = L.control({ position: "topright" });
        fallsToggle.onAdd = function() {
          const button = L.DomUtil.create("button", "map-overlay falls-toggle");
          button.type = "button";
          button.textContent = "Falls";
          button.setAttribute("aria-pressed", "true");
          L.DomEvent.disableClickPropagation(button);
          L.DomEvent.on(button, "click", (event) => {
            L.DomEvent.preventDefault(event);
            window.setWingfoilFallsVisible(!fallsVisible);
          });
          return button;
        };
        fallsToggle.addTo(map);
      }
    }

    if (typeof map !== "undefined" && typeof fallLayer !== "undefined") {
      const fallMarkers = new Set(typeof fallLayer.getLayers === "function" ? fallLayer.getLayers() : []);
      map.eachLayer((layer) => {
        if (!fallMarkers.has(layer) && typeof layer.setStyle === "function" && typeof layer.getLatLng === "function") {
          layer.setStyle({ radius: 5, weight: 1.5, opacity: 0.86, fillOpacity: 0.72 });
        }
      });
    }
  } catch (error) {
  }
})();
</script>
"""


def _send_activity_artifact_file(output_dir: Path, filename: str, artifact_path: Path):
    if filename == "map.html":
        try:
            html = artifact_path.read_text(encoding="utf-8")
        except OSError:
            abort(404)
        shim = _map_html_selection_shim()
        if "applyWingfoilRunSelection" not in html or "falls-toggle" not in html:
            html = html.replace("</body>", f"{shim}\n</body>") if "</body>" in html else f"{html}\n{shim}"
        return Response(html, mimetype="text/html")
    return send_from_directory(output_dir, filename)

def _store_activity_analysis_result(
    conn,
    row: dict[str, Any],
    upload_info: dict[str, Any],
    payload: dict[str, Any],
    context_warnings: list[str],
) -> dict[str, Any]:
    output_dir = _activity_output_dir(int(row["id"]))
    status = str(payload.get("status") or "error")
    artifacts = _registered_activity_artifacts(payload) if status == "ok" else {}
    summary = {}
    summary_name = artifacts.get("summary_json")
    if summary_name:
        summary_path = output_dir / summary_name
        _ensure_child_path(summary_path, output_dir)
        summary = _read_json_file(summary_path)

    warnings = list(context_warnings)
    payload_warnings = payload.get("warnings") or []
    if isinstance(payload_warnings, str):
        payload_warnings = [payload_warnings]
    warnings.extend(str(item) for item in payload_warnings if item)
    errors = []
    if status != "ok":
        errors.append(str(payload.get("error") or "Activity analysis failed."))

    persisted = {
        "experience_id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "uploaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "original_filename": upload_info.get("original_filename"),
        "stored_filename": upload_info.get("stored_filename"),
        "file_type": upload_info.get("file_type") or payload.get("input_type"),
        "status": status,
        "summary": summary,
        "stats": payload.get("stats") if isinstance(payload.get("stats"), dict) else {},
        "artifacts": artifacts,
        "warnings": warnings,
        "errors": errors,
        "analysis_version": payload.get("analysis_version"),
    }
    db_store.upsert_surf_experience_activity_analysis(conn, persisted)
    return persisted


def _run_activity_analysis_for_upload(conn, row: dict[str, Any], uploaded) -> dict[str, Any]:
    upload_info = _save_activity_upload(uploaded, int(row["id"]))
    output_dir = _activity_output_dir(int(row["id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_child_path(output_dir, _activity_analysis_root())
    wind_context, context_warnings = _build_wind_context(row)
    try:
        payload = analyze_session_file(
            input_file=upload_info["stored_path"],
            output_dir=output_dir,
            wind_context=wind_context,
            raise_on_error=False,
        )
    except Exception as exc:
        payload = {
            "status": "error",
            "input_type": upload_info.get("file_type"),
            "error": str(exc),
            "warnings": [],
        }
    return _store_activity_analysis_result(conn, row, upload_info, payload, context_warnings)


def _measured_wind_variability_plot(row: dict[str, Any]) -> dict[str, Any]:
    measured = row.get("measured_wind") or {}
    records = measured.get("records") or measured.get("plot_records") or []
    session_start_ms, session_end_ms = _local_session_bounds(row["date"], row["start_time"], row["end_time"])
    if session_start_ms is None or session_end_ms is None or session_end_ms <= session_start_ms:
        session_start_ms = None
        session_end_ms = None

    points = []
    for record in records:
        speed = record.get("measured_wind_speed")
        minimum = record.get("measured_wind_min")
        maximum = record.get("measured_wind_max")
        timestamp = record.get("timestamp")
        try:
            ts_value = int(timestamp)
        except (TypeError, ValueError):
            continue
        if session_start_ms is not None and not session_start_ms <= ts_value <= session_end_ms:
            continue
        try:
            speed_value = None if speed is None else float(speed)
            min_value = None if minimum is None else float(minimum)
            max_value = None if maximum is None else float(maximum)
        except (TypeError, ValueError):
            speed_value = None
            min_value = None
            max_value = None
        points.append(
            {
                "timestamp": ts_value,
                "speed": speed_value,
                "minimum": min_value,
                "maximum": max_value,
            }
        )

    points.sort(key=lambda point: point["timestamp"])
    if len(points) < 3:
        return {"available": False}

    window_ms = 30 * 60 * 1000
    min_periods = 3
    min_mean_speed = 0.1
    raw_values: list[dict[str, float | int | None]] = []
    for point in points:
        if point["speed"] is None or point["minimum"] is None or point["maximum"] is None:
            raw_values.append({"timestamp": point["timestamp"], "value": None})
            continue
        mean_speed = float(point["speed"])
        min_speed = float(point["minimum"])
        max_speed = float(point["maximum"])
        if abs(mean_speed) <= min_mean_speed:
            raw_values.append({"timestamp": point["timestamp"], "value": None})
            continue
        variability = ((max_speed * max_speed) - (min_speed * min_speed)) / (mean_speed * mean_speed)
        raw_values.append({"timestamp": point["timestamp"], "value": variability})

    trend_values: list[dict[str, float | int | None]] = []
    trend_left = 0
    for right, point in enumerate(raw_values):
        while raw_values[trend_left]["timestamp"] < point["timestamp"] - window_ms:
            trend_left += 1
        window_values = [
            float(candidate["value"])
            for candidate in raw_values[trend_left : right + 1]
            if candidate["value"] is not None
        ]
        trend_values.append(
            {
                "timestamp": point["timestamp"],
                "value": (sum(window_values) / len(window_values)) if len(window_values) >= min_periods else None,
            }
        )

    if not any(point["value"] is not None for point in raw_values):
        return {"available": False}

    min_ts = session_start_ms if session_start_ms is not None else min(point["timestamp"] for point in points)
    max_ts = session_end_ms if session_end_ms is not None else max(point["timestamp"] for point in points)
    if max_ts <= min_ts:
        max_ts = min_ts + 3_600_000

    width = 820
    height = 178
    pad_left = 48
    pad_right = 22
    pad_top = 18
    axis_y = 132
    plot_width = width - pad_left - pad_right
    plot_height = axis_y - pad_top
    min_value = 0.5
    max_value = 2.0
    threshold = 1.2

    def to_x(timestamp_ms: int | float) -> float:
        return pad_left + ((float(timestamp_ms) - min_ts) / (max_ts - min_ts)) * plot_width

    def to_y(value: float) -> float:
        return pad_top + (1.0 - ((value - min_value) / (max_value - min_value))) * plot_height

    def polyline(series: list[dict[str, float | int | None]]) -> str:
        coords = []
        for point in series:
            value = point["value"]
            if value is None:
                continue
            coords.append(f"{to_x(float(point['timestamp'])):.1f},{to_y(float(value)):.1f}")
        return " ".join(coords)

    start_local = datetime.fromtimestamp(min_ts / 1000, tz=LOCAL_TZ)
    end_local = datetime.fromtimestamp(max_ts / 1000, tz=LOCAL_TZ)
    tick_dt = start_local.replace(second=0, microsecond=0)
    minute_remainder = tick_dt.minute % 30
    if minute_remainder:
        tick_dt += timedelta(minutes=30 - minute_remainder)
    if tick_dt < start_local:
        tick_dt += timedelta(minutes=30)
    hour_ticks = []
    while tick_dt <= end_local:
        tick_ts = int(tick_dt.astimezone(timezone.utc).timestamp() * 1000)
        hour_ticks.append({"x": f"{to_x(tick_ts):.1f}", "label": tick_dt.strftime("%H:%M")})
        tick_dt += timedelta(minutes=30)
    if not hour_ticks:
        hour_ticks = [
            {"x": f"{to_x(min_ts):.1f}", "label": start_local.strftime("%H:%M")},
            {"x": f"{to_x(max_ts):.1f}", "label": end_local.strftime("%H:%M")},
        ]

    y_ticks = [
        {"y": f"{to_y(value):.1f}", "label_y": f"{to_y(value) + 4.0:.1f}", "label": f"{value:.1f}"}
        for value in (0.5, 1.0, 1.5, 2.0)
    ]
    threshold_y = to_y(threshold)
    session_variability = db_store.measured_wind_power_variability_mean(measured)
    if session_variability is None:
        raw_numeric_values = [float(point["value"]) for point in raw_values if point["value"] is not None]
        session_variability = None if not raw_numeric_values else sum(raw_numeric_values) / len(raw_numeric_values)

    return {
        "available": True,
        "width": width,
        "height": height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "plot_right": pad_left + plot_width,
        "axis_y": axis_y,
        "plot_width": plot_width,
        "plot_height": plot_height,
        "raw_points": polyline(raw_values),
        "trend_points": polyline(trend_values),
        "hour_ticks": hour_ticks,
        "y_ticks": y_ticks,
        "threshold_y": f"{threshold_y:.1f}",
        "threshold_label_y": f"{threshold_y - 5.0:.1f}",
        "latest_label": None if session_variability is None else f"Variability: {session_variability:.2f}",
        "min_value": min_value,
        "max_value": max_value,
        "window_minutes": 30,
        "min_periods": min_periods,
    }


def _measured_wind_plot(row: dict[str, Any], predictions: dict[str, Any] | None = None) -> dict[str, Any]:
    measured = row.get("measured_wind") or {}
    records = measured.get("plot_records") or measured.get("records") or []
    prediction_records = (predictions or {}).get("records") or []
    session_start_ms, session_end_ms = _local_session_bounds(row["date"], row["start_time"], row["end_time"])
    if session_start_ms is None or session_end_ms is None or session_end_ms <= session_start_ms:
        session_start_ms = None
        session_end_ms = None

    points = []
    for record in records:
        speed = record.get("measured_wind_speed")
        gust = record.get("measured_wind_gust")
        minimum = record.get("measured_wind_min")
        maximum = record.get("measured_wind_max", gust)
        timestamp = record.get("timestamp")
        if speed is None and gust is None and minimum is None and maximum is None:
            continue
        try:
            ts_value = int(timestamp)
        except (TypeError, ValueError):
            ts_value = len(points)
        if session_start_ms is not None and not session_start_ms <= ts_value <= session_end_ms:
            continue
        points.append(
            {
                "timestamp": ts_value,
                "speed": None if speed is None else float(speed),
                "gust": None if gust is None else float(gust),
                "minimum": None if minimum is None else float(minimum),
                "maximum": None if maximum is None else float(maximum),
                "direction": None if record.get("measured_wind_direction") is None else float(record.get("measured_wind_direction")),
                "iso_time": record.get("iso_time"),
            }
        )
    if not points:
        return {"available": False}

    points.sort(key=lambda point: point["timestamp"])

    values = [
        value
        for point in points
        for value in (point["speed"], point["minimum"], point["maximum"])
        if value is not None
    ]
    values.extend(
        float(value)
        for point in prediction_records
        for value in (point.get("superlocal_wind_speed"), point.get("harmonie_wind_speed"))
        if value is not None
    )
    if not values:
        return {"available": False}

    min_value = 0.0
    max_observed = max(values)
    max_value = max(10.0, math.ceil(max_observed / 5.0) * 5.0)

    min_ts = session_start_ms if session_start_ms is not None else min(point["timestamp"] for point in points)
    max_ts = session_end_ms if session_end_ms is not None else max(point["timestamp"] for point in points)
    if max_ts <= min_ts:
        max_ts = min_ts + 3_600_000

    width = 820
    height = 380
    pad_left = 48
    pad_right = 22
    pad_top = 20
    axis_y = 292
    arrow_y = 326
    plot_width = width - pad_left - pad_right
    plot_height = axis_y - pad_top

    def to_x(timestamp_ms: int) -> float:
        return pad_left + ((timestamp_ms - min_ts) / (max_ts - min_ts)) * plot_width

    def to_y(value: float) -> float:
        y = pad_top + (1.0 - ((value - min_value) / (max_value - min_value))) * plot_height
        return y

    def to_xy(point: dict[str, Any], value: float) -> tuple[float, float]:
        x = to_x(point["timestamp"])
        y = to_y(value)
        return x, y

    def polyline(key: str) -> str:
        coords = []
        for point in points:
            value = point[key]
            if value is None:
                continue
            x, y = to_xy(point, value)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    def prediction_polyline(key: str) -> str:
        coords = []
        for point in prediction_records:
            value = point.get(key)
            if value is None:
                continue
            try:
                timestamp = int(point.get("timestamp"))
            except (TypeError, ValueError):
                continue
            coords.append(f"{to_x(timestamp):.1f},{to_y(float(value)):.1f}")
        if len(coords) == 1:
            _, y = coords[0].split(",", 1)
            return f"{pad_left:.1f},{y} {pad_left + plot_width:.1f},{y}"
        return " ".join(coords)

    start_local = datetime.fromtimestamp(min_ts / 1000, tz=LOCAL_TZ)
    end_local = datetime.fromtimestamp(max_ts / 1000, tz=LOCAL_TZ)
    tick_dt = start_local.replace(second=0, microsecond=0)
    minute_remainder = tick_dt.minute % 30
    if minute_remainder:
        tick_dt += timedelta(minutes=30 - minute_remainder)
    if tick_dt < start_local:
        tick_dt += timedelta(minutes=30)
    hour_ticks = []
    while tick_dt <= end_local:
        tick_ts = int(tick_dt.astimezone(timezone.utc).timestamp() * 1000)
        hour_ticks.append(
            {
                "x": f"{to_x(tick_ts):.1f}",
                "label": tick_dt.strftime("%H:%M"),
            }
        )
        tick_dt += timedelta(minutes=30)

    if not hour_ticks:
        hour_ticks = [
            {"x": f"{to_x(min_ts):.1f}", "label": start_local.strftime("%H:%M")},
            {"x": f"{to_x(max_ts):.1f}", "label": end_local.strftime("%H:%M")},
        ]

    arrow_candidates = []
    arrow_dt = start_local.replace(second=0, microsecond=0)
    minute_offset = arrow_dt.minute % 15
    if minute_offset:
        arrow_dt += timedelta(minutes=15 - minute_offset)
    while arrow_dt <= end_local:
        tick_ts = int(arrow_dt.astimezone(timezone.utc).timestamp() * 1000)
        nearby = [
            point
            for point in points
            if point["direction"] is not None and abs(point["timestamp"] - tick_ts) <= 8 * 60 * 1000
        ]
        if nearby:
            point = min(nearby, key=lambda p: abs(p["timestamp"] - tick_ts))
            direction_deg = float(point["direction"])
            theta = math.radians((direction_deg + 180.0) % 360.0)
            arrow_len = 15.0
            dx = arrow_len * math.sin(theta)
            dy = arrow_len * math.cos(theta)
            x0 = to_x(tick_ts)
            y0 = arrow_y
            arrow_candidates.append(
                {
                    "x1": f"{x0:.1f}",
                    "y1": f"{y0:.1f}",
                    "x2": f"{x0 + dx:.1f}",
                    "y2": f"{y0 - dy:.1f}",
                }
            )
        arrow_dt += timedelta(minutes=15)

    y_tick_values = [value for value in range(0, int(max_value) + 1, 5)]
    if y_tick_values[-1] < max_value:
        y_tick_values.append(int(max_value))
    y_ticks = [
        {"y": f"{to_y(float(value)):.1f}", "label_y": f"{to_y(float(value)) + 4.0:.1f}", "label": f"{value:.0f}"}
        for value in y_tick_values
    ]
    y_minor_ticks = [
        {"y": f"{to_y(value):.1f}"}
        for value in [tick + 2.5 for tick in y_tick_values[:-1]]
        if value < max_value
    ]
    threshold_y = to_y(10.0)
    threshold_label_y = threshold_y - 6.0 if threshold_y > pad_top + 14.0 else threshold_y + 16.0

    return {
        "available": True,
        "width": width,
        "height": height,
        "pad_left": pad_left,
        "plot_right": pad_left + plot_width,
        "pad_top": pad_top,
        "axis_y": axis_y,
        "arrow_y": arrow_y,
        "plot_width": plot_width,
        "plot_height": plot_height,
        "speed_points": polyline("speed"),
        "min_points": polyline("minimum"),
        "max_points": polyline("maximum"),
        "superlocal_points": prediction_polyline("superlocal_wind_speed"),
        "harmonie_points": prediction_polyline("harmonie_wind_speed"),
        "prediction_issued_iso": (predictions or {}).get("issued_iso"),
        "min_value": min_value,
        "max_value": max_value,
        "hour_ticks": hour_ticks,
        "y_ticks": y_ticks,
        "y_minor_ticks": y_minor_ticks,
        "threshold_y": f"{threshold_y:.1f}",
        "threshold_label_y": f"{threshold_label_y:.1f}",
        "direction_arrows": arrow_candidates,
    }


def _dashboard_last_updated() -> str:
    candidates = [
        BASE_DIR / "current_day_predictions.png",
        BASE_DIR / "next_day_predictions.png",
        BASE_DIR / "model_gate_eval_history.png",
    ]
    existing = [path.stat().st_mtime for path in candidates if path.exists()]
    if not existing:
        return "unknown"
    return datetime.fromtimestamp(max(existing), tz=LOCAL_TZ).strftime("%d %B %Y %H:%M:%S %Z")


@app.route("/")
def portal_home():
    next_url = _safe_next_url(request.args.get("next"))
    if current_user() is not None:
        return redirect(next_url or url_for("experiences"))
    return render_template("portal_home.html")


@app.route("/forecast-preview")
def forecast_preview():
    return redirect(FORECAST_DASHBOARD_BASE_URL)


@app.route("/index.html")
def legacy_index():
    return redirect(url_for("portal_home"))


@app.route("/dashboard-assets/<path:filename>")
def dashboard_asset(filename: str):
    if "/" in filename or not filename.endswith((".png", ".csv")):
        abort(404)
    return send_from_directory(BASE_DIR, filename)


@app.route("/current-day-plot-archive/<path:filename>")
def current_day_archive_asset(filename: str):
    old_archive_name = re.fullmatch(r"\d{8}-\d{6}_current_day_predictions\.png", filename)
    daily_archive_name = re.fullmatch(r"current_day_predictions(?:_mobile)?_\d{4}-\d{2}-\d{2}\.png", filename)
    if "/" in filename or not (old_archive_name or daily_archive_name):
        abort(404)
    return send_from_directory(CURRENT_DAY_PLOT_ARCHIVE_DIR, filename)


@app.post("/register")
def register():
    _validate_csrf()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not _is_valid_email(username):
        flash("Enter a valid email address to create a new account.", "error")
        return redirect(url_for("portal_home", login=1))
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("portal_home", login=1))
    try:
        user_id = db_store.create_user(get_db(), username, _hash_password(password))
    except Exception:
        flash("That email address is already in use.", "error")
        return redirect(url_for("portal_home", login=1))
    session.clear()
    session["user_id"] = user_id
    flash("Account created. Add your profile defaults when you are ready.", "success")
    return redirect(url_for("profile"))


@app.post("/login")
def login():
    _validate_csrf()
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    user = db_store.get_user_by_username(get_db(), username)
    if user is None or not _verify_password(password, user["password_hash"]):
        flash("Invalid email/login name or password.", "error")
        return redirect(url_for("portal_home", login=1))
    session.clear()
    session["user_id"] = int(user["id"])
    db_store.mark_user_login(get_db(), int(user["id"]))
    flash("Logged in.", "success")
    return redirect(_safe_next_url(request.form.get("next")) or url_for("experiences"))


@app.post("/logout")
def logout():
    _validate_csrf()
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("portal_home"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user()
    profile_row = db_store.get_user_profile(get_db(), int(user["id"]))
    if request.method == "POST":
        _validate_csrf()
        errors: list[str] = []
        public_username = request.form.get("PublicUsername", "").strip()
        rider_name = request.form.get("RiderName", "").strip()
        rider_weight = _parse_int(request.form.get("RiderWeight"), "RiderWeight", errors, required=False)
        default_spot = request.form.get("DefaultSpot", "")
        if rider_weight is not None and rider_weight <= 0:
            errors.append("RiderWeight must be greater than zero.")
        if default_spot and default_spot not in SPOT_OPTIONS:
            errors.append("DefaultSpot must be one of the allowed options.")
        conflicts = db_store.find_user_profile_identity_conflicts(
            get_db(),
            int(user["id"]),
            public_username=public_username,
            rider_name=rider_name,
        )
        if "public_username" in conflicts:
            errors.append("Public username is already in use.")
        if "rider_name" in conflicts:
            errors.append("Rider name is already in use.")
        if errors:
            for error in errors:
                flash(error, "error")
        else:
            db_store.upsert_user_profile(
                get_db(), int(user["id"]), public_username, rider_name, rider_weight, default_spot
            )
            flash("Profile saved.", "success")
            return redirect(url_for("profile"))
        profile_row = {
            "public_username": public_username,
            "rider_name": rider_name,
            "rider_weight": rider_weight,
            "default_spot": default_spot,
        }
    return render_template("profile.html", profile=_profile_form_values(profile_row), spot_options=SPOT_OPTIONS)


@app.route("/experience/new", methods=["GET", "POST"])
@login_required
def new_experience():
    form_values = _experience_form_defaults()
    if request.method == "POST":
        _validate_csrf()
        submitted_values = request.form.to_dict()
        form_values.update(submitted_values)
        experience, errors = _validate_experience_form(submitted_values)
        if not errors:
            measured = db_store.get_measured_wind_for_session(
                get_db(),
                experience["spot"],
                int(experience["start_ts"]),
                int(experience["end_ts"]),
            )
            experience["user_id"] = int(current_user()["id"])
            experience["measured_wind"] = measured
            conn = get_db()
            experience_id = db_store.create_surf_experience(conn, experience)
            if measured.get("status") == "ok":
                flash("Experience submitted with measured wind data attached.", "success")
            else:
                flash("Experience submitted. Measured wind data was unavailable for that session.", "success")
            uploaded = _uploaded_activity_file()
            if uploaded is not None:
                row = db_store.get_surf_experience(conn, int(current_user()["id"]), experience_id)
                if row is not None:
                    try:
                        analysis = _run_activity_analysis_for_upload(conn, row, uploaded)
                    except ActivityUploadError as exc:
                        flash(str(exc), "error")
                    else:
                        if analysis.get("status") == "ok":
                            flash("Activity file analyzed and attached to this submission.", "success")
                        else:
                            flash("Activity file was stored, but analysis failed. See the detail page for the error.", "error")
            return redirect(url_for("experience_detail", experience_id=experience_id))
        for error in errors:
            flash(error, "error")
    return render_template(
        "submit_experience.html",
        form_values=form_values,
        form_title="New submission",
        form_action=url_for("new_experience"),
        submit_label="Save submission",
        spot_options=SPOT_OPTIONS,
        hour_options=SESSION_TIME_OPTIONS,
        wing_size_options=WING_SIZE_OPTIONS,
        foil_size_options=FOIL_SIZE_OPTIONS,
        perceived_wind_variability_options=PERCEIVED_WIND_VARIABILITY_OPTIONS,
    )


@app.route("/experiences/<int:experience_id>/edit", methods=["GET", "POST"])
@login_required
def edit_experience(experience_id: int):
    conn = get_db()
    user_id = int(current_user()["id"])
    row = db_store.get_surf_experience(conn, user_id, experience_id)
    if row is None:
        abort(404)
    form_values = _experience_form_values_from_row(row)
    if request.method == "POST":
        _validate_csrf()
        submitted_values = request.form.to_dict()
        form_values.update(submitted_values)
        experience, errors = _validate_experience_form(submitted_values)
        if not errors:
            updated = db_store.update_surf_experience(conn, experience_id, user_id, experience)
            if not updated:
                abort(404)
            flash("Submission updated.", "success")
            return redirect(url_for("experience_detail", experience_id=experience_id))
        for error in errors:
            flash(error, "error")
    return render_template(
        "submit_experience.html",
        form_values=form_values,
        form_title="Modify submission",
        form_action=url_for("edit_experience", experience_id=experience_id),
        submit_label="Save changes",
        spot_options=SPOT_OPTIONS,
        hour_options=SESSION_TIME_OPTIONS,
        wing_size_options=WING_SIZE_OPTIONS,
        foil_size_options=FOIL_SIZE_OPTIONS,
        perceived_wind_variability_options=PERCEIVED_WIND_VARIABILITY_OPTIONS,
    )


@app.route("/experiences")
@login_required
def experiences():
    sort_key = request.args.get("sort", "date")
    sort_dir = request.args.get("dir", "desc")
    scope = request.args.get("scope", "mine")
    if sort_key not in SORT_OPTIONS:
        sort_key = "date"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"
    if scope not in {"mine", "all"}:
        scope = "mine"
    db_store.backfill_surf_experience_measured_summaries(get_db(), user_id=int(current_user()["id"]))
    rows = db_store.list_surf_experiences(get_db(), int(current_user()["id"]), sort_key, sort_dir, scope=scope)
    return render_template(
        "submissions.html",
        rows=rows,
        sort_key=sort_key,
        sort_dir=sort_dir,
        scope=scope,
    )


@app.post("/experiences/<int:experience_id>/delete")
@login_required
def delete_experience(experience_id: int):
    _validate_csrf()
    deleted = db_store.delete_surf_experience(get_db(), int(current_user()["id"]), experience_id)
    if deleted:
        flash("Submission deleted.", "success")
    else:
        flash("Submission not found.", "error")
    return redirect(url_for("experiences"))


@app.post("/experiences/<int:experience_id>/share")
@login_required
def share_experience(experience_id: int):
    _validate_csrf()
    token = db_store.create_or_get_surf_experience_share_token(
        get_db(),
        experience_id,
        int(current_user()["id"]),
    )
    if token is None:
        abort(404)
    share_url = url_for("public_experience_share", share_token=token, _external=True)
    wants_json = request.headers.get("Accept", "").startswith("application/json")
    if wants_json:
        return jsonify({"share_url": share_url, "share_token": token})
    flash("Share link ready.", "success")
    return redirect(url_for("experience_detail", experience_id=experience_id, shared=1))


@app.route("/experiences/<int:experience_id>")
@login_required
def experience_detail(experience_id: int):
    conn = get_db()
    user_id = int(current_user()["id"])
    row = db_store.get_visible_surf_experience(conn, user_id, experience_id)
    if row is None:
        abort(404)
    measured_summary = (row.get("measured_wind") or {}).get("summary") or {}
    if row["is_owner"] and (
        row.get("avg_forecast_temperature") is None
        or not (row.get("measured_wind") or {}).get("plot_records")
        or measured_summary.get("max_wind_speed_kind") != "average_wind"
        or measured_summary.get("max_wind_gust") is None
        or "wind_variability" not in measured_summary
        or not any(
            "measured_wind_min" in record and "measured_wind_max" in record
            for record in (row.get("measured_wind") or {}).get("plot_records", [])
        )
    ):
        db_store.refresh_surf_experience_measured_wind(conn, experience_id, user_id=user_id)
        row = db_store.get_visible_surf_experience(conn, user_id, experience_id)
        if row is None:
            abort(404)
    session_start_ms, session_end_ms = _local_session_bounds(row["date"], row["start_time"], row["end_time"])
    predictions = (
        db_store.get_prediction_lines_for_session(conn, row["spot"], session_start_ms, session_end_ms)
        if session_start_ms is not None and session_end_ms is not None
        else {"status": "unavailable", "records": []}
    )
    stored_analysis = db_store.get_surf_experience_activity_analysis(conn, experience_id)
    activity_analysis = _activity_analysis_view_model(stored_analysis) if (row["is_owner"] or row["visibility"] == "public") else None
    return render_template(
        "submission_detail.html",
        row=row,
        wind_plot=_measured_wind_plot(row, predictions),
        wind_variability_plot=_measured_wind_variability_plot(row),
        current_day_archive_plot=_current_day_archive_plot_for_submission(row["date"]),
        activity_analysis=activity_analysis,
    )


@app.post("/experiences/<int:experience_id>/activity-upload")
@login_required
def upload_experience_activity(experience_id: int):
    _validate_csrf()
    conn = get_db()
    user_id = int(current_user()["id"])
    row = db_store.get_surf_experience(conn, user_id, experience_id)
    if row is None:
        abort(404)
    uploaded = _uploaded_activity_file()
    if uploaded is None:
        flash(f"Upload a {ACTIVITY_UPLOAD_FORMATS} activity file.", "error")
        return redirect(url_for("experience_detail", experience_id=experience_id))
    try:
        analysis = _run_activity_analysis_for_upload(conn, row, uploaded)
    except ActivityUploadError as exc:
        flash(str(exc), "error")
    else:
        if analysis.get("status") == "ok":
            flash("Activity file analyzed and attached to this submission.", "success")
        else:
            flash("Activity file was stored, but analysis failed. See the error below.", "error")
    return redirect(url_for("experience_detail", experience_id=experience_id))


def _send_registered_activity_artifact(experience_id: int, filename: str, analysis: dict[str, Any] | None):
    if analysis is None:
        abort(404)
    if not _is_safe_artifact_name(filename):
        abort(404)
    registered = set((analysis.get("artifacts") or {}).values())
    if filename not in registered:
        abort(404)
    output_dir = _activity_output_dir(experience_id)
    artifact_path = output_dir / filename
    try:
        _ensure_child_path(artifact_path, output_dir)
    except ActivityUploadError:
        abort(404)
    if not artifact_path.is_file():
        abort(404)
    return _send_activity_artifact_file(output_dir, filename, artifact_path)


@app.route("/experiences/<int:experience_id>/activity-artifact/<path:filename>")
@login_required
def experience_activity_artifact(experience_id: int, filename: str):
    conn = get_db()
    user_id = int(current_user()["id"])
    row = db_store.get_visible_surf_experience(conn, user_id, experience_id)
    if row is None:
        abort(404)
    analysis = db_store.get_surf_experience_activity_analysis(conn, experience_id)
    return _send_registered_activity_artifact(experience_id, filename, analysis)


@app.route("/share/experience/<share_token>/activity-artifact/<path:filename>")
def shared_experience_activity_artifact(share_token: str, filename: str):
    conn = get_db()
    row = db_store.get_shared_public_surf_experience(conn, share_token)
    if row is None:
        abort(404)
    analysis = db_store.get_surf_experience_activity_analysis(conn, int(row["id"]))
    return _send_registered_activity_artifact(int(row["id"]), filename, analysis)


@app.route("/share/experience/<share_token>")
def public_experience_share(share_token: str):
    conn = get_db()
    row = db_store.get_shared_public_surf_experience(conn, share_token)
    if row is None:
        abort(404)
    session_start_ms, session_end_ms = _local_session_bounds(row["date"], row["start_time"], row["end_time"])
    predictions = (
        db_store.get_prediction_lines_for_session(conn, row["spot"], session_start_ms, session_end_ms)
        if session_start_ms is not None and session_end_ms is not None
        else {"status": "unavailable", "records": []}
    )
    stored_analysis = db_store.get_surf_experience_activity_analysis(conn, int(row["id"]))
    activity_analysis = _activity_analysis_view_model(
        stored_analysis,
        artifact_url_builder=lambda filename: url_for(
            "shared_experience_activity_artifact",
            share_token=share_token,
            filename=filename,
        ),
    )
    return render_template(
        "submission_public.html",
        row=row,
        wind_plot=_measured_wind_plot(row, predictions),
        wind_variability_plot=_measured_wind_variability_plot(row),
        activity_analysis=activity_analysis,
    )


if __name__ == "__main__":
    app.run(host=os.environ.get("WIND_DASHBOARD_HOST", "127.0.0.1"), port=int(os.environ.get("WIND_DASHBOARD_PORT", "8080")))
