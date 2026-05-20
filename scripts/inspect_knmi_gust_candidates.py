#!/usr/bin/env python3
"""Inspect KNMI HARMONIE P1 10 m fields that may correspond to Windsurfice max wind."""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.knmi_harmonie import KNOTS_PER_MPS, SitePoint, nearest_value, open_grib_parameter


DEFAULT_DB = Path("data/wind_data_all_sites.db")
DEFAULT_SITE_POINTS = {
    "valkenburgsemeer": SitePoint(site="valkenburgsemeer", lat=52.168, lon=4.437),
}
GRIB_NAME_RE = re.compile(r"^HA43_[^_]+_(\d{12})_(\d{3})00_GB$")


@dataclass(frozen=True)
class GribMember:
    name: str
    horizon: int


@dataclass(frozen=True)
class GribField:
    parameter: int
    level: int
    type_of_level: str
    step_range: str
    short_name: str
    name: str
    units: str


def parse_member(name: str) -> GribMember | None:
    match = GRIB_NAME_RE.match(Path(name).name)
    if not match:
        return None
    return GribMember(name=name, horizon=int(match.group(2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect KNMI P1 10 m GRIB fields and compare candidates to Windsurfice WindForecastMax."
    )
    parser.add_argument("--raw-tar", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--horizons", default="0,6,12,24,48")
    parser.add_argument("--max-vintage-delta-minutes", type=float, default=90.0)
    return parser.parse_args()


def parse_horizons(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return sorted(set(values))


def connect_read_only(path: Path) -> sqlite3.Connection:
    db_path = path.expanduser().resolve()
    errors = []
    for query in ("mode=ro", "immutable=1"):
        try:
            conn = sqlite3.connect(f"file:{db_path}?{query}", uri=True)
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            return conn
        except sqlite3.Error as exc:
            errors.append(f"{query}: {exc}")
    raise sqlite3.OperationalError("; ".join(errors))


def timestamp_ms(value: object) -> int:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot parse timestamp: {value!r}")
    return int(pd.Timestamp(ts).timestamp() * 1000)


def iso_z(value: object) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def run_and_target(member_name: str) -> tuple[pd.Timestamp, int, pd.Timestamp]:
    member = parse_member(member_name)
    if member is None:
        raise ValueError(f"Cannot parse HARMONIE member name: {member_name}")
    match = GRIB_NAME_RE.match(Path(member_name).name)
    if match is None:
        raise ValueError(f"Cannot parse HARMONIE member name: {member_name}")
    run = pd.to_datetime(match.group(1), format="%Y%m%d%H%M", utc=True)
    target = run + pd.Timedelta(hours=member.horizon)
    return run, member.horizon, target


def list_members(raw_tar: Path) -> list[GribMember]:
    with tarfile.open(raw_tar, "r") as tar:
        members = []
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parsed = parse_member(member.name)
            if parsed is not None:
                members.append(parsed)
    return sorted(members, key=lambda item: item.horizon)


def extract_selected(raw_tar: Path, horizons: list[int], temp_dir: Path) -> list[Path]:
    selected = {int(horizon) for horizon in horizons}
    extracted_paths = []
    with tarfile.open(raw_tar, "r") as tar:
        for member in tar.getmembers():
            parsed = parse_member(member.name)
            if parsed is None or parsed.horizon not in selected:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            out_path = temp_dir / Path(member.name).name
            with out_path.open("wb") as handle:
                handle.write(source.read())
            extracted_paths.append(out_path)
    return sorted(extracted_paths)


def grib_ls_fields(grib_path: Path) -> list[GribField]:
    fields = []
    completed = subprocess.run(
        [
            "grib_ls",
            "-p",
            "count,indicatorOfParameter,indicatorOfTypeOfLevel,level,typeOfLevel,stepRange,shortName,name,units",
            str(grib_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"grib_ls failed for {grib_path}: {completed.stderr.strip()}")
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            int(parts[0])
            parameter = int(parts[1])
            level = int(parts[3])
        except ValueError:
            continue
        fields.append(
            GribField(
                parameter=parameter,
                level=level,
                type_of_level=parts[4],
                step_range=parts[5],
                short_name=parts[6],
                name=" ".join(parts[7:-1]),
                units=parts[-1],
            )
        )
    return fields


def print_grib_inventory(paths: list[Path]) -> dict[int, GribField]:
    candidate_fields: dict[int, GribField] = {}
    print("\nGRIB 10 m inventory")
    print("===================")
    for path in paths:
        fields = grib_ls_fields(path)
        ten_meter = [field for field in fields if field.level == 10 and field.type_of_level == "heightAboveGround"]
        print(f"\n{path.name}: {len(fields)} messages, {len(ten_meter)} 10 m heightAboveGround fields")
        rows = []
        for field in ten_meter:
            candidate_fields[field.parameter] = field
            rows.append(
                {
                    "parameter": field.parameter,
                    "stepRange": field.step_range,
                    "shortName": field.short_name,
                    "name": field.name,
                    "units": field.units,
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))
    return candidate_fields


def site_point_from_db(conn: sqlite3.Connection, site: str) -> SitePoint:
    try:
        row = conn.execute(
            """
            SELECT site_lat, site_lon
            FROM harmonie_knmi_features
            WHERE site = ? AND site_lat IS NOT NULL AND site_lon IS NOT NULL
            ORDER BY run_ts DESC
            LIMIT 1
            """,
            (site,),
        ).fetchone()
    except sqlite3.Error:
        row = None
    if row is not None:
        return SitePoint(site=site, lat=float(row[0]), lon=float(row[1]))
    if site in DEFAULT_SITE_POINTS:
        return DEFAULT_SITE_POINTS[site]
    raise ValueError(f"No configured site coordinates for {site!r}.")


def point_value(grib_path: Path, parameter: int, site: SitePoint) -> tuple[float | None, str, str]:
    try:
        ds = open_grib_parameter(grib_path, parameter=parameter, level=10)
    except Exception as exc:
        return None, "", str(exc)
    try:
        value, _, _ = nearest_value(ds, site)
        variable = next(iter(ds.data_vars.values()))
        var_name = next(iter(ds.data_vars.keys()))
        units = str(variable.attrs.get("units", "unknown"))
        return float(value), units, var_name
    except Exception as exc:
        return None, "", str(exc)
    finally:
        ds.close()


def load_windsurfice_candidates(
    conn: sqlite3.Connection,
    site: str,
    target_ms_value: int,
    reference_ms: int,
    max_delta_minutes: float,
) -> tuple[float | None, float | None, float | None]:
    query = """
        SELECT fetched_ts, run_ts, wind_speed, wind_gust
        FROM forecasts
        WHERE site = ?
          AND model = 'HARMONIE'
          AND target_ts = ?
          AND wind_gust IS NOT NULL
    """
    rows = pd.read_sql_query(query, conn, params=(site, target_ms_value))
    if rows.empty:
        return None, None, None
    rows["vintage_ts"] = pd.to_numeric(rows["fetched_ts"], errors="coerce").fillna(
        pd.to_numeric(rows["run_ts"], errors="coerce")
    )
    rows["delta_min"] = (rows["vintage_ts"] - int(reference_ms)).abs() / 60000.0
    rows = rows[rows["delta_min"] <= float(max_delta_minutes)]
    if rows.empty:
        return None, None, None
    best = rows.sort_values(["delta_min", "vintage_ts"]).iloc[0]
    return float(best["wind_gust"]), float(best["wind_speed"]), float(best["delta_min"])


def classify_values(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "unavailable"
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if min_value >= 0 and max_value <= 40:
        return "plausible m/s wind-speed magnitude"
    if min_value >= 0 and max_value <= 80:
        return "plausible knots wind-speed magnitude"
    if min_value >= 0 and max_value <= 360:
        return "plausible direction degrees, but only weakly from range"
    if min_value < 0 and max_value > 0:
        return "signed component-like field"
    return "other/dimensionless"


def conversion_candidates(value: float) -> dict[str, float]:
    return {
        "none/raw": value,
        "m/s-to-knots": value * KNOTS_PER_MPS,
        "knots-to-m/s": value / KNOTS_PER_MPS,
    }


def build_records(paths: list[Path], site: SitePoint, conn: sqlite3.Connection, site_name: str, max_delta_minutes: float) -> pd.DataFrame:
    records = []
    for path in paths:
        run_dt, horizon, target_dt = run_and_target(path.name)
        run_ms = timestamp_ms(run_dt)
        target_ms_value = timestamp_ms(target_dt)
        wf_gust, wf_avg, delta_min = load_windsurfice_candidates(
            conn,
            site_name,
            target_ms_value=target_ms_value,
            reference_ms=run_ms,
            max_delta_minutes=max_delta_minutes,
        )
        values: dict[int, tuple[float | None, str, str]] = {}
        for parameter in sorted({33, 34, 162, 163}):
            values[parameter] = point_value(path, parameter, site)
        u10 = values[33][0]
        v10 = values[34][0]
        derived_speed_mps = None if u10 is None or v10 is None else math.sqrt(u10 * u10 + v10 * v10)
        if derived_speed_mps is not None:
            records.append(
                {
                    "candidate": "derived_speed_33_34",
                    "parameter": "33+34",
                    "level": "10 m",
                    "run_ts": iso_z(run_dt),
                    "target_ts": iso_z(target_dt),
                    "horizon_hr": horizon,
                    "raw_value": derived_speed_mps,
                    "raw_units": "m/s from u/v",
                    "candidate_value_kt": derived_speed_mps * KNOTS_PER_MPS,
                    "conversion": "m/s-to-knots",
                    "windsurfice_wind_gust_kt": wf_gust,
                    "windsurfice_wind_avg_kt": wf_avg,
                    "vintage_delta_minutes": delta_min,
                }
            )
        for parameter, (value, units, var_name) in values.items():
            if value is None:
                continue
            for conversion, converted in conversion_candidates(value).items():
                records.append(
                    {
                        "candidate": f"parameter_{parameter}",
                        "parameter": parameter,
                        "level": "10 m",
                        "run_ts": iso_z(run_dt),
                        "target_ts": iso_z(target_dt),
                        "horizon_hr": horizon,
                        "raw_value": value,
                        "raw_units": units or "unknown",
                        "cfgrib_name": var_name,
                        "candidate_value_kt": converted,
                        "conversion": conversion,
                        "windsurfice_wind_gust_kt": wf_gust,
                        "windsurfice_wind_avg_kt": wf_avg,
                        "vintage_delta_minutes": delta_min,
                    }
                )
        p162 = values.get(162, (None, "", ""))[0]
        p163 = values.get(163, (None, "", ""))[0]
        if p162 is not None and p163 is not None:
            magnitude = math.sqrt(p162 * p162 + p163 * p163)
            for conversion, converted in conversion_candidates(magnitude).items():
                records.append(
                    {
                        "candidate": "vector_magnitude_162_163",
                        "parameter": "162+163",
                        "level": "10 m",
                        "run_ts": iso_z(run_dt),
                        "target_ts": iso_z(target_dt),
                        "horizon_hr": horizon,
                        "raw_value": magnitude,
                        "raw_units": "derived magnitude",
                        "candidate_value_kt": converted,
                        "conversion": conversion,
                        "windsurfice_wind_gust_kt": wf_gust,
                        "windsurfice_wind_avg_kt": wf_avg,
                        "vintage_delta_minutes": delta_min,
                    }
                )
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame, target_col: str) -> pd.DataFrame:
    matched = records.dropna(subset=[target_col]).copy()
    if matched.empty:
        return pd.DataFrame()
    rows = []
    for (candidate, parameter, conversion), group in matched.groupby(["candidate", "parameter", "conversion"], dropna=False):
        diff = group["candidate_value_kt"] - group[target_col]
        abs_diff = diff.abs()
        rows.append(
            {
                "candidate": candidate,
                "parameter": parameter,
                "level": "10 m",
                "conversion": conversion,
                "matched_rows": int(len(group)),
                "value_min": group["raw_value"].min(),
                "value_max": group["raw_value"].max(),
                "classification": classify_values(group["raw_value"]),
                "mae": abs_diff.mean(),
                "median_abs_error": abs_diff.median(),
                "p95_abs_error": abs_diff.quantile(0.95) if len(group) >= 3 else pd.NA,
                "max_abs_error": abs_diff.max(),
                "correlation": group["candidate_value_kt"].corr(group[target_col]) if len(group) >= 3 else pd.NA,
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "matched_rows"], ascending=[True, False])


def print_summary(title: str, summary: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    if summary.empty:
        print("No matched rows.")
        return
    print(summary.round(3).to_string(index=False))


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    if not args.raw_tar.exists():
        raise SystemExit(f"Raw tar not found: {args.raw_tar}")

    members = list_members(args.raw_tar)
    print("KNMI gust candidate inspection")
    print("==============================")
    print(f"Raw tar: {args.raw_tar}")
    print(f"Tar members: {len(members)}")
    print(f"Available horizons: {members[0].horizon}-{members[-1].horizon}" if members else "Available horizons: none")
    print(f"Requested horizons: {horizons}")

    conn = connect_read_only(args.db)
    try:
        site_point = site_point_from_db(conn, args.site)
        print(f"Site: {site_point.site} ({site_point.lat}, {site_point.lon})")
        with TemporaryDirectory(prefix="knmi_gust_candidates_") as tmp:
            paths = extract_selected(args.raw_tar, horizons, Path(tmp))
            if not paths:
                raise SystemExit("No requested horizons were found in the raw tar.")
            print("Extracted horizons: " + ", ".join(str(parse_member(path.name).horizon) for path in paths if parse_member(path.name)))
            fields = print_grib_inventory(paths)
            print("\nCandidate 10 m parameters discovered: " + ", ".join(str(key) for key in sorted(fields)))
            records = build_records(paths, site_point, conn, args.site, args.max_vintage_delta_minutes)
    finally:
        conn.close()

    if records.empty:
        raise SystemExit("No candidate point values could be read from the selected GRIB files.")

    print("\nPoint values")
    print("============")
    display_cols = [
        "candidate",
        "parameter",
        "horizon_hr",
        "raw_value",
        "raw_units",
        "conversion",
        "candidate_value_kt",
        "windsurfice_wind_gust_kt",
        "windsurfice_wind_avg_kt",
        "vintage_delta_minutes",
    ]
    print(records[display_cols].round(3).to_string(index=False))

    gust_summary = summarize(records, "windsurfice_wind_gust_kt")
    avg_summary = summarize(records, "windsurfice_wind_avg_kt")
    print_summary("Comparison against Windsurfice wind_gust / WindForecastMax", gust_summary)
    print_summary("Sanity comparison against Windsurfice average wind", avg_summary)

    if gust_summary.empty:
        print("\nNo KNMI P1 candidate field clearly matches Windsurfice wind_gust from this diagnostic.")
        return

    best = gust_summary.iloc[0]
    interpretation = (
        f"Best candidate: {best['candidate']} parameter {best['parameter']} using {best['conversion']} "
        f"with MAE {best['mae']:.3f} kt over {int(best['matched_rows'])} matched rows."
    )
    print("\nInterpretation")
    print("==============")
    print(interpretation)
    clear = (
        int(best["matched_rows"]) >= max(4, min(5, len(parse_horizons(args.horizons))))
        and float(best["mae"]) <= 1.5
        and str(best["candidate"]) != "derived_speed_33_34"
    )
    if clear:
        print(
            f"Candidate parameter {best['parameter']} at level 10 m appears to match "
            f"Windsurfice wind_gust under conversion {best['conversion']}."
        )
    else:
        print("No KNMI P1 candidate field clearly matches Windsurfice wind_gust from this diagnostic.")


if __name__ == "__main__":
    main()
