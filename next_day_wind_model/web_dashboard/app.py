from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import math
import os
import secrets
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import db_store


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
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
HOUR_OPTIONS = [f"{hour:02d}:00" for hour in range(24)]
SORT_OPTIONS = {
    "date",
    "spot",
    "start_time",
    "session_rating",
    "wing_size",
    "foil_size",
    "avg_measured_wind_speed",
    "max_measured_wind_speed",
    "min_measured_wind_speed",
    "mean_measured_direction",
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


@app.context_processor
def inject_globals():
    return {
        "current_user": current_user(),
        "current_profile": current_profile(),
        "csrf_token": _csrf_token,
        "companion_app_base_url": COMPANION_APP_BASE_URL,
        "forecast_dashboard_base_url": FORECAST_DASHBOARD_BASE_URL,
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


def _local_session_bounds(day: str, start_time: str, end_time: str) -> tuple[int | None, int | None]:
    try:
        start_hour = int(start_time[:2])
        end_hour = int(end_time[:2])
        session_date = date.fromisoformat(day)
        start_dt = datetime.combine(session_date, datetime.min.time(), tzinfo=LOCAL_TZ).replace(hour=start_hour)
        end_dt = datetime.combine(session_date, datetime.min.time(), tzinfo=LOCAL_TZ).replace(hour=end_hour)
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
        "Rider": profile.get("rider_name", ""),
        "Spot": profile.get("default_spot") or "Valkenburgse meer",
        "Date": datetime.now(LOCAL_TZ).date().isoformat(),
        "StartTime": start_time,
        "EndTime": "14:00",
        "SessionRating": "3",
        "RiderReview": "",
        "RiderWeight": "" if profile.get("rider_weight") is None else str(profile.get("rider_weight")),
        "WingSize": "5",
        "FoilSize": "1500",
        "RiderNotes": "",
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

    if not rider:
        errors.append("Rider is required.")
    if spot not in SPOT_OPTIONS:
        errors.append("Spot must be one of the allowed options.")
    try:
        date.fromisoformat(day)
    except ValueError:
        errors.append("Date must be valid.")
    if start_time not in HOUR_OPTIONS:
        errors.append("StartTime must be a full-hour time.")
    if end_time not in HOUR_OPTIONS:
        errors.append("EndTime must be a full-hour time.")
    if start_time in HOUR_OPTIONS and end_time in HOUR_OPTIONS and end_time < start_time:
        errors.append("EndTime cannot be earlier than StartTime.")
    if rating is not None and not 1 <= rating <= 5:
        errors.append("SessionRating must be between 1 and 5.")
    if rider_weight is not None and rider_weight <= 0:
        errors.append("RiderWeight must be greater than zero.")
    if wing_size is not None and wing_size not in WING_SIZE_OPTIONS:
        errors.append("WingSize must be one of the allowed options.")
    if foil_size is not None and foil_size not in FOIL_SIZE_OPTIONS:
        errors.append("FoilSize must be one of the allowed options.")

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
            "rider_review": form.get("RiderReview", "").strip(),
            "rider_weight": rider_weight,
            "wing_size": wing_size,
            "foil_size": foil_size,
            "rider_notes": form.get("RiderNotes", "").strip(),
        },
        errors,
    )


def _dashboard_asset_url(filename: str) -> str:
    return url_for("dashboard_asset", filename=filename)


def _measured_wind_plot(row: dict[str, Any]) -> dict[str, Any]:
    records = (row.get("measured_wind") or {}).get("records") or []
    session_start_ms, session_end_ms = _local_session_bounds(row["date"], row["start_time"], row["end_time"])
    if session_start_ms is None or session_end_ms is None or session_end_ms <= session_start_ms:
        session_start_ms = None
        session_end_ms = None

    points = []
    for record in records:
        speed = record.get("measured_wind_speed")
        gust = record.get("measured_wind_gust")
        timestamp = record.get("timestamp")
        if speed is None and gust is None:
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
                "direction": None if record.get("measured_wind_direction") is None else float(record.get("measured_wind_direction")),
                "iso_time": record.get("iso_time"),
            }
        )
    if not points:
        return {"available": False}

    values = [
        value
        for point in points
        for value in (point["speed"], point["gust"])
        if value is not None
    ]
    if not values:
        return {"available": False}

    min_value = 0.0
    max_observed = max(values)
    max_value = max(20.0, math.ceil(max_observed * 1.12 / 5.0) * 5.0)

    min_ts = session_start_ms if session_start_ms is not None else min(point["timestamp"] for point in points)
    max_ts = session_end_ms if session_end_ms is not None else max(point["timestamp"] for point in points)
    if max_ts <= min_ts:
        max_ts = min_ts + 3_600_000

    width = 820
    height = 300
    pad_left = 48
    pad_right = 22
    pad_top = 20
    axis_y = 218
    arrow_y = 246
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

    start_local = datetime.fromtimestamp(min_ts / 1000, tz=LOCAL_TZ)
    end_local = datetime.fromtimestamp(max_ts / 1000, tz=LOCAL_TZ)
    tick_dt = start_local.replace(minute=0, second=0, microsecond=0)
    if tick_dt < start_local:
        tick_dt += timedelta(hours=1)
    hour_ticks = []
    while tick_dt <= end_local:
        tick_ts = int(tick_dt.astimezone(timezone.utc).timestamp() * 1000)
        hour_ticks.append(
            {
                "x": f"{to_x(tick_ts):.1f}",
                "label": tick_dt.strftime("%H:%M"),
            }
        )
        tick_dt += timedelta(hours=1)

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

    y_ticks = [
        {"y": f"{to_y(max_value):.1f}", "label_y": f"{to_y(max_value) + 4.0:.1f}", "label": f"{max_value:.0f}"},
        {
            "y": f"{to_y(max_value / 2.0):.1f}",
            "label_y": f"{to_y(max_value / 2.0) + 4.0:.1f}",
            "label": f"{max_value / 2.0:.0f}",
        },
        {"y": f"{to_y(0.0):.1f}", "label_y": f"{to_y(0.0) + 4.0:.1f}", "label": "0"},
    ]

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
        "gust_points": polyline("gust"),
        "min_value": min_value,
        "max_value": max_value,
        "hour_ticks": hour_ticks,
        "y_ticks": y_ticks,
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
    if current_user() is not None:
        return redirect(url_for("experiences"))
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


@app.post("/register")
def register():
    _validate_csrf()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if len(username) < 3:
        flash("Username must be at least 3 characters.", "error")
        return redirect(url_for("portal_home", login=1))
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("portal_home", login=1))
    try:
        user_id = db_store.create_user(get_db(), username, _hash_password(password))
    except Exception:
        flash("That username is already in use.", "error")
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
        flash("Invalid username or password.", "error")
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
        rider_name = request.form.get("RiderName", "").strip()
        rider_weight = _parse_int(request.form.get("RiderWeight"), "RiderWeight", errors, required=False)
        default_spot = request.form.get("DefaultSpot", "")
        if rider_weight is not None and rider_weight <= 0:
            errors.append("RiderWeight must be greater than zero.")
        if default_spot and default_spot not in SPOT_OPTIONS:
            errors.append("DefaultSpot must be one of the allowed options.")
        if errors:
            for error in errors:
                flash(error, "error")
        else:
            db_store.upsert_user_profile(get_db(), int(user["id"]), rider_name, rider_weight, default_spot)
            flash("Profile saved.", "success")
            return redirect(url_for("profile"))
        profile_row = {
            "rider_name": rider_name,
            "rider_weight": rider_weight,
            "default_spot": default_spot,
        }
    return render_template("profile.html", profile=profile_row or {}, spot_options=SPOT_OPTIONS)


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
            experience_id = db_store.create_surf_experience(get_db(), experience)
            if measured.get("status") == "ok":
                flash("Experience submitted with measured wind data attached.", "success")
            else:
                flash("Experience submitted. Measured wind data was unavailable for that session.", "success")
            return redirect(url_for("experience_detail", experience_id=experience_id))
        for error in errors:
            flash(error, "error")
    return render_template(
        "submit_experience.html",
        form_values=form_values,
        spot_options=SPOT_OPTIONS,
        hour_options=HOUR_OPTIONS,
        wing_size_options=WING_SIZE_OPTIONS,
        foil_size_options=FOIL_SIZE_OPTIONS,
    )


@app.route("/experiences")
@login_required
def experiences():
    sort_key = request.args.get("sort", "date")
    sort_dir = request.args.get("dir", "desc")
    if sort_key not in SORT_OPTIONS:
        sort_key = "date"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"
    db_store.backfill_surf_experience_measured_summaries(get_db(), user_id=int(current_user()["id"]))
    rows = db_store.list_surf_experiences(get_db(), int(current_user()["id"]), sort_key, sort_dir)
    return render_template(
        "submissions.html",
        rows=rows,
        sort_key=sort_key,
        sort_dir=sort_dir,
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


@app.route("/experiences/<int:experience_id>")
@login_required
def experience_detail(experience_id: int):
    row = db_store.get_surf_experience(get_db(), int(current_user()["id"]), experience_id)
    if row is None:
        abort(404)
    return render_template("submission_detail.html", row=row, wind_plot=_measured_wind_plot(row))


if __name__ == "__main__":
    app.run(host=os.environ.get("WIND_DASHBOARD_HOST", "127.0.0.1"), port=int(os.environ.get("WIND_DASHBOARD_PORT", "8080")))
