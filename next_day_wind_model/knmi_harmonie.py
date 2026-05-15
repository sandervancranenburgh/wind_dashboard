"""Reusable KNMI HARMONIE P1 wind feature extraction helpers."""

from __future__ import annotations

import math
import os
import json
import sqlite3
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd
import requests

if TYPE_CHECKING:
    import xarray as xr


DATASET = "harmonie_arome_cy43_p1"
VERSION = "1.0"
SOURCE = "knmi_harmonie_p1"
TABLE_NAME = "harmonie_knmi_features"
SHADOW_TABLE_NAME = "knmi_forecasts_shadow"
LEVELS = (10, 50, 100, 200, 300)
U_WIND_PARAMETER = 33
V_WIND_PARAMETER = 34
KNOTS_PER_MPS = 1.9438444924406


@dataclass(frozen=True)
class SitePoint:
    site: str
    lat: float
    lon: float


@dataclass(frozen=True)
class KnmiFileInfo:
    filename: str
    last_modified: str | None = None
    size: int | None = None


@dataclass(frozen=True)
class ExtractionResult:
    frame: pd.DataFrame
    errors: tuple[str, ...]


class KnmiApiError(RuntimeError):
    pass


class KnmiExtractionError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot parse UTC timestamp: {value!r}")
    return pd.Timestamp(ts)


def timestamp_ms(value: Any) -> int:
    ts = parse_utc_timestamp(value)
    return int(ts.timestamp() * 1000)


def json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_run_and_horizon(filename: str) -> tuple[pd.Timestamp, int, pd.Timestamp]:
    """Parse run and forecast horizon from a KNMI HARMONIE GRIB member name."""
    name = Path(filename).name
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse KNMI run/horizon from filename: {filename}")

    run_str = parts[2]
    horizon_str = parts[3]
    try:
        run_ts = pd.to_datetime(run_str, format="%Y%m%d%H%M", utc=True)
        horizon_hr = int(horizon_str[:3])
    except Exception as exc:
        raise ValueError(f"Cannot parse KNMI run/horizon from filename: {filename}") from exc

    target_ts = run_ts + pd.Timedelta(hours=horizon_hr)
    return run_ts, horizon_hr, target_ts


def parse_run_from_tar_filename(filename: str) -> pd.Timestamp:
    """Parse run timestamp from names like HARM43_V1_P1_2026051504.tar."""
    stem = Path(filename).name.removesuffix(".tar")
    run_str = stem.split("_")[-1]
    try:
        return pd.to_datetime(run_str, format="%Y%m%d%H", utc=True)
    except Exception as exc:
        raise ValueError(f"Cannot parse KNMI run timestamp from tar filename: {filename}") from exc


def mps_to_knots(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * KNOTS_PER_MPS


def wind_speed_direction(u: float, v: float) -> tuple[float, float]:
    """Return speed in m/s and meteorological direction in degrees."""
    speed = math.sqrt(u**2 + v**2)
    direction = (270.0 - math.degrees(math.atan2(v, u))) % 360.0
    return speed, direction


def circular_difference_degrees(a: float, b: float) -> float:
    """Smallest signed circular difference a - b in degrees."""
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(float(denominator)) < 1e-12:
        return None
    return float(numerator) / float(denominator)


def _base_url(dataset: str = DATASET, version: str = VERSION) -> str:
    return (
        "https://api.dataplatform.knmi.nl/open-data/v1/"
        f"datasets/{dataset}/versions/{version}/files"
    )


def _api_headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key or os.getenv("KNMI_API_KEY")
    if not key:
        raise KnmiApiError("KNMI_API_KEY is not set.")
    return {"Authorization": key}


def list_knmi_files(
    dataset: str = DATASET,
    version: str = VERSION,
    *,
    api_key: str | None = None,
    max_keys: int = 10,
) -> list[KnmiFileInfo]:
    """List KNMI files, newest first."""
    params = {"maxKeys": int(max_keys), "orderBy": "created", "sorting": "desc"}
    response = requests.get(_base_url(dataset, version), headers=_api_headers(api_key), params=params, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise KnmiApiError(f"KNMI file listing failed with HTTP {response.status_code}.") from exc

    files = response.json().get("files", [])
    out: list[KnmiFileInfo] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if not filename:
            continue
        out.append(
            KnmiFileInfo(
                filename=str(filename),
                last_modified=item.get("lastModified"),
                size=int(item["size"]) if item.get("size") is not None else None,
            )
        )
    return out


def select_latest_tar_file(files: Iterable[KnmiFileInfo]) -> KnmiFileInfo:
    tar_files = [item for item in files if item.filename.endswith(".tar")]
    if not tar_files:
        raise KnmiApiError("No KNMI HARMONIE tar files returned by the API.")
    return tar_files[0]


def get_download_url(
    filename: str,
    dataset: str = DATASET,
    version: str = VERSION,
    *,
    api_key: str | None = None,
) -> str:
    url = f"{_base_url(dataset, version)}/{filename}/url"
    response = requests.get(url, headers=_api_headers(api_key), timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise KnmiApiError(f"KNMI download URL request failed with HTTP {response.status_code}.") from exc
    download_url = response.json().get("temporaryDownloadUrl")
    if not download_url:
        raise KnmiApiError(f"KNMI download URL response did not include temporaryDownloadUrl for {filename}.")
    return str(download_url)


def download_file(download_url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with requests.get(download_url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp_path.replace(out_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return out_path


def ensure_downloaded_tar(
    filename: str,
    out_dir: Path,
    dataset: str = DATASET,
    version: str = VERSION,
    *,
    api_key: str | None = None,
) -> Path:
    out_path = out_dir / filename
    if out_path.exists():
        return out_path
    download_url = get_download_url(filename, dataset, version, api_key=api_key)
    return download_file(download_url, out_path)


def open_grib_parameter(grib_path: Path, parameter: int, level: int) -> "xr.Dataset":
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise KnmiExtractionError(
            "xarray is required for KNMI GRIB extraction. Install next_day_wind_model/requirements.txt."
        ) from exc

    try:
        return xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "indicatorOfParameter": parameter,
                    "level": level,
                },
                "indexpath": "",
            },
        )
    except Exception as exc:
        raise KnmiExtractionError(
            f"Could not open parameter={parameter} level={level} from {grib_path.name}."
        ) from exc


def nearest_value(ds: "xr.Dataset", site: SitePoint) -> tuple[float, float, float]:
    point = ds.sel(latitude=site.lat, longitude=site.lon, method="nearest")
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise KnmiExtractionError(f"Expected one GRIB data variable, found: {data_vars}")
    var = data_vars[0]
    return float(point[var].values), float(point.latitude), float(point.longitude)


def _level_feature_names(level: int) -> tuple[str, str, str, str, str, str]:
    return (
        f"u_{level}m_mps",
        f"v_{level}m_mps",
        f"wind_speed_{level}m_mps",
        f"wind_dir_{level}m",
        f"wind_dir_sin_{level}m",
        f"wind_dir_cos_{level}m",
    )


def extract_one_grib(
    grib_path: Path,
    site: SitePoint,
    *,
    source: str = SOURCE,
    dataset: str = DATASET,
    fetched_ts: str | None = None,
    levels: Iterable[int] = LEVELS,
) -> dict[str, Any]:
    run_ts, horizon_hr, target_ts = parse_run_and_horizon(grib_path.name)
    row: dict[str, Any] = {
        "source": source,
        "dataset": dataset,
        "run_ts": run_ts.isoformat(),
        "fetched_ts": fetched_ts or utc_now_iso(),
        "target_ts": target_ts.isoformat(),
        "horizon_hr": int(horizon_hr),
        "site": site.site,
        "site_lat": float(site.lat),
        "site_lon": float(site.lon),
    }

    grid_lat = None
    grid_lon = None
    for level in levels:
        u_ds = open_grib_parameter(grib_path, parameter=U_WIND_PARAMETER, level=int(level))
        v_ds = open_grib_parameter(grib_path, parameter=V_WIND_PARAMETER, level=int(level))
        try:
            u, lat, lon = nearest_value(u_ds, site)
            v, _, _ = nearest_value(v_ds, site)
        finally:
            u_ds.close()
            v_ds.close()

        speed, direction = wind_speed_direction(u, v)
        u_name, v_name, speed_name, dir_name, dir_sin_name, dir_cos_name = _level_feature_names(int(level))
        row[u_name] = u
        row[v_name] = v
        row[speed_name] = speed
        row[dir_name] = direction
        row[dir_sin_name] = math.sin(math.radians(direction))
        row[dir_cos_name] = math.cos(math.radians(direction))
        if int(level) == 10:
            row["wind_speed_10m_knots"] = mps_to_knots(speed)
        grid_lat = lat
        grid_lon = lon

    row["grid_lat"] = grid_lat
    row["grid_lon"] = grid_lon
    add_derived_features(row)
    return row


def add_derived_features(row: dict[str, Any]) -> None:
    for high_level in (50, 100, 200, 300):
        low = row.get("wind_speed_10m_mps")
        high = row.get(f"wind_speed_{high_level}m_mps")
        row[f"speed_shear_10m_{high_level}m"] = None if low is None or high is None else float(high) - float(low)
        row[f"speed_ratio_10m_{high_level}m"] = safe_ratio(low, high)

    for high_level in (100, 200, 300):
        low_dir = row.get("wind_dir_10m")
        high_dir = row.get(f"wind_dir_{high_level}m")
        row[f"dir_shear_10m_{high_level}m"] = (
            None if low_dir is None or high_dir is None else circular_difference_degrees(float(high_dir), float(low_dir))
        )


def extract_tar_features(
    tar_path: Path,
    site: SitePoint,
    *,
    source: str = SOURCE,
    dataset: str = DATASET,
    fetched_ts: str | None = None,
    levels: Iterable[int] = LEVELS,
    continue_on_error: bool = False,
) -> ExtractionResult:
    if not tar_path.exists():
        raise FileNotFoundError(f"KNMI tar file not found: {tar_path}")

    fetched = fetched_ts or utc_now_iso()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with tarfile.open(tar_path, "r") as tar:
        members = sorted((member for member in tar.getmembers() if member.isfile()), key=lambda member: member.name)
        if not members:
            raise KnmiExtractionError(f"No GRIB members found in {tar_path}.")

        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for member in members:
                try:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise KnmiExtractionError(f"Could not read tar member {member.name}.")
                    grib_path = tmp_dir / Path(member.name).name
                    with grib_path.open("wb") as handle:
                        handle.write(extracted.read())
                    rows.append(
                        extract_one_grib(
                            grib_path,
                            site,
                            source=source,
                            dataset=dataset,
                            fetched_ts=fetched,
                            levels=levels,
                        )
                    )
                except Exception as exc:
                    message = f"{member.name}: {exc}"
                    if not continue_on_error:
                        raise KnmiExtractionError(message) from exc
                    errors.append(message)
                finally:
                    try:
                        grib_path.unlink()
                    except (NameError, OSError):
                        pass

    if not rows:
        raise KnmiExtractionError(f"No KNMI feature rows extracted from {tar_path}.")
    frame = pd.DataFrame(rows).sort_values(["run_ts", "horizon_hr"]).reset_index(drop=True)
    return ExtractionResult(frame=frame, errors=tuple(errors))


def create_harmonie_knmi_features_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            run_ts TEXT NOT NULL,
            fetched_ts TEXT NOT NULL,
            target_ts TEXT NOT NULL,
            horizon_hr INTEGER NOT NULL,
            site TEXT NOT NULL,
            site_lat REAL,
            site_lon REAL,
            grid_lat REAL,
            grid_lon REAL,
            u_10m_mps REAL,
            v_10m_mps REAL,
            wind_speed_10m_mps REAL,
            wind_speed_10m_knots REAL,
            wind_dir_10m REAL,
            wind_dir_sin_10m REAL,
            wind_dir_cos_10m REAL,
            u_50m_mps REAL,
            v_50m_mps REAL,
            wind_speed_50m_mps REAL,
            wind_dir_50m REAL,
            wind_dir_sin_50m REAL,
            wind_dir_cos_50m REAL,
            u_100m_mps REAL,
            v_100m_mps REAL,
            wind_speed_100m_mps REAL,
            wind_dir_100m REAL,
            wind_dir_sin_100m REAL,
            wind_dir_cos_100m REAL,
            u_200m_mps REAL,
            v_200m_mps REAL,
            wind_speed_200m_mps REAL,
            wind_dir_200m REAL,
            wind_dir_sin_200m REAL,
            wind_dir_cos_200m REAL,
            u_300m_mps REAL,
            v_300m_mps REAL,
            wind_speed_300m_mps REAL,
            wind_dir_300m REAL,
            wind_dir_sin_300m REAL,
            wind_dir_cos_300m REAL,
            speed_shear_10m_50m REAL,
            speed_shear_10m_100m REAL,
            speed_shear_10m_200m REAL,
            speed_shear_10m_300m REAL,
            speed_ratio_10m_50m REAL,
            speed_ratio_10m_100m REAL,
            speed_ratio_10m_200m REAL,
            speed_ratio_10m_300m REAL,
            dir_shear_10m_100m REAL,
            dir_shear_10m_200m REAL,
            dir_shear_10m_300m REAL,
            created_at TEXT NOT NULL,
            UNIQUE(source, dataset, run_ts, target_ts, site)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_site_run ON {TABLE_NAME}(site, run_ts)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_site_target ON {TABLE_NAME}(site, target_ts)"
    )
    conn.commit()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]


def upsert_harmonie_knmi_features(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    create_harmonie_knmi_features_table(conn)
    if frame.empty:
        return 0

    columns = _table_columns(conn, TABLE_NAME)
    insert_columns = [col for col in columns if col in frame.columns or col == "created_at"]
    created_at = utc_now_iso()
    rows = []
    for record in frame.to_dict(orient="records"):
        row = []
        for column in insert_columns:
            value = created_at if column == "created_at" else record.get(column)
            if pd.isna(value):
                value = None
            row.append(value)
        rows.append(row)

    placeholders = ", ".join("?" for _ in insert_columns)
    column_sql = ", ".join(insert_columns)
    update_columns = [
        column
        for column in insert_columns
        if column not in {"source", "dataset", "run_ts", "target_ts", "site"}
    ]
    update_sql = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
    conn.executemany(
        f"""
        INSERT INTO {TABLE_NAME} ({column_sql})
        VALUES ({placeholders})
        ON CONFLICT(source, dataset, run_ts, target_ts, site)
        DO UPDATE SET {update_sql}
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def latest_harmonie_knmi_rows(conn: sqlite3.Connection, site: str, limit: int = 5) -> pd.DataFrame:
    create_harmonie_knmi_features_table(conn)
    return pd.read_sql_query(
        f"""
        SELECT
            site,
            run_ts,
            target_ts,
            horizon_hr,
            wind_speed_10m_mps,
            wind_speed_10m_knots,
            wind_dir_10m,
            grid_lat,
            grid_lon,
            created_at
        FROM {TABLE_NAME}
        WHERE site = ?
        ORDER BY run_ts DESC, horizon_hr ASC
        LIMIT ?
        """,
        conn,
        params=(site, int(limit)),
    )


def create_knmi_forecasts_shadow_table(conn: sqlite3.Connection) -> None:
    """Create a non-production forecast mirror for KNMI replacement checks."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SHADOW_TABLE_NAME} (
            site TEXT NOT NULL,
            model TEXT NOT NULL,
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            run_ts INTEGER NOT NULL,
            run_iso TEXT NOT NULL,
            fetched_ts INTEGER NOT NULL,
            fetched_iso TEXT NOT NULL,
            target_ts INTEGER NOT NULL,
            target_iso TEXT NOT NULL,
            horizon_hr INTEGER NOT NULL,
            wind_speed REAL,
            wind_gust REAL,
            wind_dir REAL,
            payload TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(source, dataset, site, run_ts, target_ts)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{SHADOW_TABLE_NAME}_site_run ON {SHADOW_TABLE_NAME}(site, run_ts)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{SHADOW_TABLE_NAME}_site_target ON {SHADOW_TABLE_NAME}(site, target_ts)"
    )
    conn.commit()


def feature_frame_to_shadow_forecasts(frame: pd.DataFrame, model: str = "HARMONIE") -> pd.DataFrame:
    """
    Convert canonical KNMI feature rows into the current forecast-table shape.

    The current Windsurfice-backed `forecasts.wind_speed` values are in the same
    unit as `WindForecastAvr`, which verification showed corresponds to knots.
    KNMI 10 m speed is extracted in m/s and converted here to knots so this
    shadow table can be compared against the existing fetch/logging path.
    """
    if frame.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "site": frame["site"],
            "model": model,
            "source": frame["source"],
            "dataset": frame["dataset"],
            "run_iso": frame["run_ts"].map(lambda value: parse_utc_timestamp(value).isoformat().replace("+00:00", "Z")),
            "fetched_iso": frame["fetched_ts"].map(
                lambda value: parse_utc_timestamp(value).isoformat().replace("+00:00", "Z")
            ),
            "target_iso": frame["target_ts"].map(lambda value: parse_utc_timestamp(value).isoformat().replace("+00:00", "Z")),
            "horizon_hr": pd.to_numeric(frame["horizon_hr"], errors="coerce").astype("Int64"),
            "wind_speed": pd.to_numeric(frame["wind_speed_10m_knots"], errors="coerce"),
            "wind_gust": None,
            "wind_dir": pd.to_numeric(frame["wind_dir_10m"], errors="coerce"),
        }
    )
    out["run_ts"] = out["run_iso"].map(timestamp_ms)
    out["fetched_ts"] = out["fetched_iso"].map(timestamp_ms)
    out["target_ts"] = out["target_iso"].map(timestamp_ms)
    out["payload"] = frame.apply(
        lambda row: json.dumps(
            {
                "source": row.get("source"),
                "dataset": row.get("dataset"),
                "wind_speed_10m_mps": json_scalar(row.get("wind_speed_10m_mps")),
                "wind_speed_10m_knots": json_scalar(row.get("wind_speed_10m_knots")),
                "wind_dir_10m": json_scalar(row.get("wind_dir_10m")),
                "grid_lat": json_scalar(row.get("grid_lat")),
                "grid_lon": json_scalar(row.get("grid_lon")),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        axis=1,
    )
    return out


def upsert_knmi_forecasts_shadow(conn: sqlite3.Connection, frame: pd.DataFrame, model: str = "HARMONIE") -> int:
    create_knmi_forecasts_shadow_table(conn)
    shadow = feature_frame_to_shadow_forecasts(frame, model=model)
    if shadow.empty:
        return 0

    columns = [
        "site",
        "model",
        "source",
        "dataset",
        "run_ts",
        "run_iso",
        "fetched_ts",
        "fetched_iso",
        "target_ts",
        "target_iso",
        "horizon_hr",
        "wind_speed",
        "wind_gust",
        "wind_dir",
        "payload",
        "created_at",
    ]
    created_at = utc_now_iso()
    rows = []
    for record in shadow.to_dict(orient="records"):
        row = []
        for column in columns:
            value = created_at if column == "created_at" else record.get(column)
            if value is not None and pd.isna(value):
                value = None
            row.append(value)
        rows.append(row)

    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    update_columns = [
        column for column in columns if column not in {"source", "dataset", "site", "run_ts", "target_ts"}
    ]
    update_sql = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
    conn.executemany(
        f"""
        INSERT INTO {SHADOW_TABLE_NAME} ({column_sql})
        VALUES ({placeholders})
        ON CONFLICT(source, dataset, site, run_ts, target_ts)
        DO UPDATE SET {update_sql}
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def write_knmi_rows_to_production_forecasts(conn: sqlite3.Connection, frame: pd.DataFrame, model: str = "HARMONIE") -> int:
    """
    Dangerous compatibility path for explicit manual tests only.

    This writes KNMI-derived forecast rows into the production `forecasts` table.
    Callers should expose this only behind a clearly named opt-in flag. Normal
    shadow verification must use `knmi_forecasts_shadow` instead.
    """
    shadow = feature_frame_to_shadow_forecasts(frame, model=model)
    if shadow.empty:
        return 0
    rows = [
        (
            row.site,
            row.model,
            int(row.run_ts),
            row.run_iso,
            int(row.fetched_ts),
            row.fetched_iso,
            int(row.target_ts),
            row.target_iso,
            int(row.horizon_hr),
            None if pd.isna(row.wind_speed) else float(row.wind_speed),
            None,
            None if pd.isna(row.wind_dir) else float(row.wind_dir),
            row.payload,
        )
        for row in shadow.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO forecasts(
            site, model, run_ts, run_iso, fetched_ts, fetched_iso,
            target_ts, target_iso, horizon_hr, wind_speed, wind_gust, wind_dir, payload
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,json(?))
        ON CONFLICT(site, model, run_ts, target_ts) DO UPDATE SET
            run_iso=excluded.run_iso,
            fetched_ts=excluded.fetched_ts,
            fetched_iso=excluded.fetched_iso,
            target_iso=excluded.target_iso,
            horizon_hr=excluded.horizon_hr,
            wind_speed=excluded.wind_speed,
            wind_gust=excluded.wind_gust,
            wind_dir=excluded.wind_dir,
            payload=excluded.payload
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def latest_shadow_rows(conn: sqlite3.Connection, site: str, limit: int = 5) -> pd.DataFrame:
    create_knmi_forecasts_shadow_table(conn)
    return pd.read_sql_query(
        f"""
        SELECT
            site,
            model,
            source,
            run_iso,
            fetched_iso,
            target_iso,
            horizon_hr,
            wind_speed,
            wind_dir,
            created_at
        FROM {SHADOW_TABLE_NAME}
        WHERE site = ?
        ORDER BY run_ts DESC, horizon_hr ASC
        LIMIT ?
        """,
        conn,
        params=(site, int(limit)),
    )


def recent_knmi_runs(conn: sqlite3.Connection, site: str, limit: int = 10) -> pd.DataFrame:
    create_harmonie_knmi_features_table(conn)
    return pd.read_sql_query(
        f"""
        SELECT
            site,
            run_ts,
            MIN(fetched_ts) AS min_fetched_ts,
            MAX(fetched_ts) AS max_fetched_ts,
            MIN(target_ts) AS min_target_ts,
            MAX(target_ts) AS max_target_ts,
            COUNT(DISTINCT horizon_hr) AS horizon_count,
            COUNT(*) AS row_count
        FROM {TABLE_NAME}
        WHERE site = ?
        GROUP BY site, run_ts
        ORDER BY run_ts DESC
        LIMIT ?
        """,
        conn,
        params=(site, int(limit)),
    )


def knmi_archive_diagnostic(conn: sqlite3.Connection, site: str, now_ts: str | None = None) -> dict[str, Any]:
    create_harmonie_knmi_features_table(conn)
    now_iso = now_ts or utc_now_iso()
    now_ms = timestamp_ms(now_iso)
    summary = pd.read_sql_query(
        f"""
        SELECT
            COUNT(DISTINCT run_ts) AS distinct_run_ts,
            COUNT(DISTINCT substr(target_ts, 1, 10)) AS distinct_target_dates,
            MIN(run_ts) AS min_run_ts,
            MAX(run_ts) AS max_run_ts,
            MIN(target_ts) AS min_target_ts,
            MAX(target_ts) AS max_target_ts,
            COUNT(*) AS row_count,
            SUM(CASE WHEN strftime('%s', target_ts) * 1000 <= ? THEN 1 ELSE 0 END) AS past_target_rows
        FROM {TABLE_NAME}
        WHERE site = ?
        """,
        conn,
        params=(now_ms, site),
    )
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    joinable_exact = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {TABLE_NAME} AS k
        JOIN observations AS o
          ON o.site = k.site
         AND o.ts = CAST(strftime('%s', k.target_ts) AS INTEGER) * 1000
        WHERE k.site = ?
        """,
        (site,),
    ).fetchone()[0]
    joinable_30min = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {TABLE_NAME} AS k
        WHERE k.site = ?
          AND EXISTS (
              SELECT 1
              FROM observations AS o
              WHERE o.site = k.site
                AND ABS(o.ts - CAST(strftime('%s', k.target_ts) AS INTEGER) * 1000) <= 30 * 60 * 1000
          )
        """,
        (site,),
    ).fetchone()[0]
    row["rows_joinable_to_observations_exact"] = int(joinable_exact or 0)
    row["rows_joinable_to_observations_30min"] = int(joinable_30min or 0)
    row["diagnostic_now_ts"] = now_iso
    return row
