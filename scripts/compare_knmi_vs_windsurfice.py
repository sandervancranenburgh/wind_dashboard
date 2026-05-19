#!/usr/bin/env python3
"""Compare extracted KNMI HARMONIE point winds with stored Windsurfice values."""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_KNMI_CSV = Path("data/processed/knmi/harmonie_p1_wind_features_2026051504.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/knmi/knmi_vs_windsurfice_comparison_2026051504.csv")
KNOTS_PER_MPS = 1.9438444924406


@dataclass(frozen=True)
class CandidateSource:
    kind: str
    path: Path
    detail: str


@dataclass(frozen=True)
class ExistingData:
    frame: pd.DataFrame
    source: CandidateSource
    loaded_rows: int


def parse_timestamp(value: object) -> pd.Timestamp:
    """Parse ISO strings or epoch seconds/milliseconds as a UTC timestamp."""
    if value is None or pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    if isinstance(value, (int, float)):
        number = float(value)
        unit = "ms" if abs(number) >= 10_000_000_000 else "s"
        return pd.to_datetime(number, unit=unit, utc=True, errors="coerce")

    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        number = float(text)
    except ValueError:
        return pd.to_datetime(text, utc=True, errors="coerce")

    unit = "ms" if abs(number) >= 10_000_000_000 else "s"
    return pd.to_datetime(number, unit=unit, utc=True, errors="coerce")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    return series.map(parse_timestamp)


def timestamp_ms(ts: pd.Timestamp) -> int:
    ts = parse_timestamp(ts)
    if pd.isna(ts):
        raise ValueError("Cannot convert NaT to epoch milliseconds.")
    return int(ts.timestamp() * 1000)


def iso_z(ts: pd.Timestamp | object) -> str | None:
    ts = parse_timestamp(ts)
    if pd.isna(ts):
        return None
    return ts.isoformat().replace("+00:00", "Z")


def circular_direction_difference_degrees(a: pd.Series, b: pd.Series) -> pd.Series:
    """Smallest absolute circular difference between two direction series."""
    return ((a.astype(float) - b.astype(float) + 180.0) % 360.0 - 180.0).abs()


def discover_candidate_data_sources(data_dir: Path, site: str) -> list[CandidateSource]:
    sources: list[CandidateSource] = []
    for db_name in ("wind_data_all_sites.db", "wind_data.db"):
        db_path = data_dir / db_name
        if db_path.exists():
            sources.append(CandidateSource("sqlite", db_path, "SQLite forecasts table"))

    site_dir = data_dir / site
    if site_dir.exists():
        csv_count = len(list(site_dir.glob("*_forecast.csv")))
        if csv_count:
            sources.append(CandidateSource("csv", site_dir, f"{csv_count} forecast CSV snapshots"))
    return sources


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def load_knmi_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"KNMI CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"target_ts", "horizon_hr", "wind_speed_10m", "wind_dir_10m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"KNMI CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["target_dt"] = parse_timestamp_series(df["target_ts"])
    if "run_ts" in df.columns:
        df["run_dt"] = parse_timestamp_series(df["run_ts"])
    else:
        df["run_dt"] = pd.NaT
    df["wind_speed_10m"] = pd.to_numeric(df["wind_speed_10m"], errors="coerce")
    df["wind_dir_10m"] = pd.to_numeric(df["wind_dir_10m"], errors="coerce")
    df["horizon_hr"] = pd.to_numeric(df["horizon_hr"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["target_dt", "wind_speed_10m", "wind_dir_10m"]).sort_values("target_dt")
    if df.empty:
        raise ValueError(f"KNMI CSV has no parseable target/speed/direction rows: {path}")
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} KNMI rows with invalid target/speed/direction values.")
    return df


def load_existing_from_sqlite(
    source: CandidateSource,
    site: str,
    model: str,
    target_min: pd.Timestamp,
    target_max: pd.Timestamp,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    conn = sqlite3.connect(source.path)
    try:
        columns = table_columns(conn, "forecasts")
        required = {"site", "model", "target_ts", "wind_speed", "wind_dir"}
        if not required.issubset(columns):
            missing = sorted(required - columns)
            raise ValueError(f"{source.path} forecasts table is missing columns: {missing}")

        start_ms = timestamp_ms(target_min - tolerance)
        end_ms = timestamp_ms(target_max + tolerance)
        query = """
            SELECT
                site,
                model,
                run_ts,
                run_iso,
                fetched_ts,
                fetched_iso,
                target_ts,
                target_iso,
                horizon_hr,
                wind_speed,
                wind_dir
            FROM forecasts
            WHERE site = ?
              AND model = ?
              AND target_ts BETWEEN ? AND ?
            ORDER BY run_ts, target_ts
        """
        return pd.read_sql_query(query, conn, params=(site, model, start_ms, end_ms))
    finally:
        conn.close()


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_real = {col.lower(): col for col in columns}
    for candidate in candidates:
        real = lower_to_real.get(candidate.lower())
        if real:
            return real
    return None


def load_existing_from_csv_snapshots(
    source: CandidateSource,
    site: str,
    model: str,
    target_min: pd.Timestamp,
    target_max: pd.Timestamp,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(source.path.glob("*_forecast.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping unreadable forecast CSV {path}: {exc}")
            continue

        target_col = first_present(df.columns, ["timestamp", "time", "UnixTime", "ts", "dt", "iso_time", "target_ts"])
        speed_col = first_present(df.columns, ["WindForecastAvr", "forecast_avg", "wind_speed", "windspeed", "speed"])
        dir_col = first_present(df.columns, ["WindDirection", "forecast_dir", "wind_dir", "winddirection", "direction"])
        if not target_col or not speed_col or not dir_col:
            continue

        run_col = first_present(df.columns, ["forecast_run_ts", "run_ts", "issued_ts"])
        fetched_col = first_present(df.columns, ["forecast_fetched_ts", "fetched_ts"])

        part = pd.DataFrame(
            {
                "site": site,
                "model": model,
                "target_dt": parse_timestamp_series(df[target_col]),
                "wind_speed": pd.to_numeric(df[speed_col], errors="coerce"),
                "wind_dir": pd.to_numeric(df[dir_col], errors="coerce"),
                "source_file": str(path),
            }
        )
        part["run_dt"] = parse_timestamp_series(df[run_col]) if run_col else pd.NaT
        part["fetched_dt"] = parse_timestamp_series(df[fetched_col]) if fetched_col else pd.NaT
        part["run_ts"] = part["run_dt"].map(lambda ts: timestamp_ms(ts) if not pd.isna(ts) else pd.NA)
        part["fetched_ts"] = part["fetched_dt"].map(lambda ts: timestamp_ms(ts) if not pd.isna(ts) else pd.NA)
        part["run_iso"] = part["run_dt"].map(iso_z)
        part["fetched_iso"] = part["fetched_dt"].map(iso_z)
        part["target_ts"] = part["target_dt"].map(lambda ts: timestamp_ms(ts) if not pd.isna(ts) else pd.NA)
        part["target_iso"] = part["target_dt"].map(iso_z)
        part["horizon_hr"] = pd.NA
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    start = target_min - tolerance
    end = target_max + tolerance
    return out[(out["target_dt"] >= start) & (out["target_dt"] <= end)].copy()


def normalize_existing_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if "target_dt" not in out.columns:
        if "target_ts" in out.columns:
            out["target_dt"] = parse_timestamp_series(out["target_ts"])
        else:
            out["target_dt"] = parse_timestamp_series(out["target_iso"])
    if "run_dt" not in out.columns:
        out["run_dt"] = parse_timestamp_series(out["run_ts"] if "run_ts" in out.columns else out.get("run_iso"))
    if "fetched_dt" not in out.columns:
        out["fetched_dt"] = parse_timestamp_series(out["fetched_ts"] if "fetched_ts" in out.columns else out.get("fetched_iso"))

    out["wind_speed"] = pd.to_numeric(out["wind_speed"], errors="coerce")
    out["wind_dir"] = pd.to_numeric(out["wind_dir"], errors="coerce")
    out["horizon_hr"] = pd.to_numeric(out.get("horizon_hr"), errors="coerce")
    out = out.dropna(subset=["target_dt", "wind_speed", "wind_dir"])
    return out.sort_values(["run_dt", "fetched_dt", "target_dt"], na_position="first")


def load_existing_data(
    data_dir: Path,
    site: str,
    model: str,
    target_min: pd.Timestamp,
    target_max: pd.Timestamp,
    tolerance: pd.Timedelta,
    source_kind: str,
) -> ExistingData:
    candidates = discover_candidate_data_sources(data_dir, site)
    if source_kind != "auto":
        candidates = [candidate for candidate in candidates if candidate.kind == source_kind]

    if not candidates:
        raise FileNotFoundError(f"No candidate Windsurfice data sources found under {data_dir} for site={site!r}.")

    errors: list[str] = []
    for source in candidates:
        try:
            if source.kind == "sqlite":
                raw = load_existing_from_sqlite(source, site, model, target_min, target_max, tolerance)
            elif source.kind == "csv":
                raw = load_existing_from_csv_snapshots(source, site, model, target_min, target_max, tolerance)
            else:
                continue
            frame = normalize_existing_frame(raw)
        except Exception as exc:
            errors.append(f"{source.kind}:{source.path}: {exc}")
            continue
        if not frame.empty:
            return ExistingData(frame=frame, source=source, loaded_rows=len(frame))
        errors.append(f"{source.kind}:{source.path}: no rows in KNMI target window")

    detail = "\n  - ".join(errors)
    raise RuntimeError(f"Could not load existing Windsurfice/HARMONIE rows.\n  - {detail}")


def select_existing_records(df: pd.DataFrame, policy: str, knmi_run_dt: pd.Timestamp | None) -> tuple[pd.DataFrame, str]:
    if df.empty:
        return df.copy(), "no existing rows"

    sort_cols = ["target_dt", "run_dt", "fetched_dt"]
    sorted_df = df.sort_values(sort_cols, na_position="first").copy()

    if policy == "latest-per-target":
        selected = sorted_df.drop_duplicates("target_dt", keep="last")
        latest_run = selected["run_dt"].dropna().max()
        note = f"selected latest available Windsurfice row per target; latest run/fetch timestamp={iso_z(latest_run)}"
        return selected.sort_values("target_dt"), note

    if policy == "nearest-run" and knmi_run_dt is not None and not pd.isna(knmi_run_dt):
        run_key = sorted_df["run_dt"].fillna(sorted_df["fetched_dt"])
        candidates = []
        for run_dt, group in sorted_df.assign(_run_key=run_key).groupby("_run_key", dropna=False):
            if pd.isna(run_dt):
                continue
            coverage = group["target_dt"].nunique()
            delta_seconds = abs((pd.Timestamp(run_dt) - knmi_run_dt).total_seconds())
            after_penalty = 0 if pd.Timestamp(run_dt) >= knmi_run_dt else 1
            candidates.append((coverage, after_penalty, delta_seconds, pd.Timestamp(run_dt), group))
        if candidates:
            coverage, _, delta_seconds, selected_run, group = sorted(
                candidates,
                key=lambda item: (-item[0], item[1], item[2], -item[3].timestamp()),
            )[0]
            selected = group.sort_values(["target_dt", "fetched_dt"]).drop_duplicates("target_dt", keep="last")
            note = (
                f"selected nearest complete-ish Windsurfice run/fetch={iso_z(selected_run)} "
                f"({delta_seconds / 3600.0:.2f} h from KNMI run, {coverage} target timestamps)"
            )
            return selected.sort_values("target_dt"), note

    selected = sorted_df.drop_duplicates("target_dt", keep="last")
    return selected.sort_values("target_dt"), "fell back to latest available row per target"


def choose_speed_factor(matches: pd.DataFrame, mode: str) -> tuple[float, str, pd.DataFrame]:
    candidates = {
        "none": 1.0,
        "mps-to-knots": KNOTS_PER_MPS,
        "knots-to-mps": 1.0 / KNOTS_PER_MPS,
    }
    if mode != "auto":
        factor = candidates[mode]
        return factor, mode, pd.DataFrame()

    rows = []
    paired = matches.dropna(subset=["wind_speed_10m", "windsurfice_wind_speed"])
    if paired.empty:
        return 1.0, "none", pd.DataFrame()
    for name, factor in candidates.items():
        diff = paired["wind_speed_10m"].astype(float) * factor - paired["windsurfice_wind_speed"].astype(float)
        rows.append({"conversion": name, "factor": factor, "mae": float(diff.abs().mean())})
    scores = pd.DataFrame(rows).sort_values("mae")
    best = scores.iloc[0]
    return float(best["factor"]), str(best["conversion"]), scores


def match_and_compare(
    knmi: pd.DataFrame,
    existing: pd.DataFrame,
    tolerance: pd.Timedelta,
    speed_conversion: str,
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    left = knmi[
        ["target_dt", "horizon_hr", "run_dt", "wind_speed_10m", "wind_dir_10m"]
    ].rename(columns={"run_dt": "knmi_run_dt"})
    right = existing[
        ["target_dt", "run_dt", "fetched_dt", "horizon_hr", "wind_speed", "wind_dir"]
    ].rename(
        columns={
            "target_dt": "windsurfice_target_dt",
            "run_dt": "windsurfice_run_dt",
            "fetched_dt": "windsurfice_fetched_dt",
            "horizon_hr": "windsurfice_horizon_hr",
            "wind_speed": "windsurfice_wind_speed",
            "wind_dir": "windsurfice_wind_dir",
        }
    )

    if tolerance > pd.Timedelta(0):
        merged = pd.merge_asof(
            left.sort_values("target_dt"),
            right.sort_values("windsurfice_target_dt"),
            left_on="target_dt",
            right_on="windsurfice_target_dt",
            direction="nearest",
            tolerance=tolerance,
        )
    else:
        merged = left.merge(
            right,
            left_on="target_dt",
            right_on="windsurfice_target_dt",
            how="left",
        )

    merged["target_time_delta_seconds"] = (
        merged["windsurfice_target_dt"] - merged["target_dt"]
    ).dt.total_seconds()

    speed_factor, speed_conversion_used, conversion_scores = choose_speed_factor(merged, speed_conversion)
    merged["knmi_wind_speed_compared"] = merged["wind_speed_10m"].astype(float) * speed_factor
    merged["wind_speed_diff"] = merged["knmi_wind_speed_compared"] - merged["windsurfice_wind_speed"]
    merged["wind_dir_circular_diff"] = circular_direction_difference_degrees(
        merged["wind_dir_10m"],
        merged["windsurfice_wind_dir"],
    )

    out = pd.DataFrame(
        {
            "target_ts": merged["target_dt"].map(iso_z),
            "horizon_hr": merged["horizon_hr"],
            "knmi_run_ts": merged["knmi_run_dt"].map(iso_z),
            "windsurfice_target_ts": merged["windsurfice_target_dt"].map(iso_z),
            "target_time_delta_seconds": merged["target_time_delta_seconds"],
            "knmi_wind_speed_10m": merged["wind_speed_10m"],
            "knmi_wind_speed_compared": merged["knmi_wind_speed_compared"],
            "speed_conversion": speed_conversion_used,
            "windsurfice_wind_speed": merged["windsurfice_wind_speed"],
            "wind_speed_diff": merged["wind_speed_diff"],
            "knmi_wind_dir_10m": merged["wind_dir_10m"],
            "windsurfice_wind_dir": merged["windsurfice_wind_dir"],
            "wind_dir_circular_diff": merged["wind_dir_circular_diff"],
            "windsurfice_run_ts": merged["windsurfice_run_dt"].map(iso_z),
            "windsurfice_fetched_ts": merged["windsurfice_fetched_dt"].map(iso_z),
            "windsurfice_horizon_hr": merged["windsurfice_horizon_hr"],
        }
    )
    return out, speed_conversion_used, conversion_scores


def print_summary(
    knmi_rows: int,
    existing_loaded_rows: int,
    existing_selected_rows: int,
    comparison: pd.DataFrame,
    existing_data: ExistingData,
    selection_note: str,
    speed_conversion: str,
    conversion_scores: pd.DataFrame,
    tolerance: pd.Timedelta,
    output_csv: Path,
) -> None:
    matched = comparison["windsurfice_wind_speed"].notna()
    matched_rows = int(matched.sum())
    unmatched_rows = int((~matched).sum())
    speed_abs = comparison.loc[matched, "wind_speed_diff"].abs()
    dir_abs = comparison.loc[matched, "wind_dir_circular_diff"]

    print("\nKNMI versus Windsurfice/HARMONIE comparison")
    print("===========================================")
    print(f"Source: {existing_data.source.kind} {existing_data.source.path} ({existing_data.source.detail})")
    print(f"Selection: {selection_note}")
    print(f"Timestamp matching: {'exact target_ts' if tolerance == pd.Timedelta(0) else f'nearest within {tolerance}'}")
    print(f"Speed conversion used for differences: {speed_conversion}")
    if not conversion_scores.empty:
        scores = ", ".join(
            f"{row.conversion} MAE={row.mae:.3f}"
            for row in conversion_scores.itertuples(index=False)
        )
        print(f"Speed conversion candidates: {scores}")
    print(f"KNMI rows: {knmi_rows}")
    print(f"Existing/Windsurfice rows loaded: {existing_loaded_rows}")
    print(f"Existing/Windsurfice rows selected: {existing_selected_rows}")
    print(f"Matched rows: {matched_rows}")
    print(f"Unmatched KNMI rows: {unmatched_rows}")
    if matched_rows:
        print(f"Mean absolute wind speed difference: {speed_abs.mean():.3f}")
        print(f"Max absolute wind speed difference: {speed_abs.max():.3f}")
        print(f"Mean circular absolute direction difference: {dir_abs.mean():.3f} deg")
        print(f"Max circular absolute direction difference: {dir_abs.max():.3f} deg")
    else:
        print("No matched rows, so difference metrics are unavailable.")
    print(f"Comparison CSV: {output_csv}")

    run_values = comparison.loc[matched, "windsurfice_run_ts"].dropna().unique()
    if len(run_values) == 1:
        print(
            "Note: Windsurfice run_ts currently comes from forecast metadata when available; "
            "for fallback_fetch_time snapshots it is the fetch time, not an authoritative model run."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare KNMI-extracted HARMONIE 10 m winds with stored Windsurfice/HARMONIE forecasts."
    )
    parser.add_argument("--knmi-csv", type=Path, default=DEFAULT_KNMI_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--model", default="HARMONIE")
    parser.add_argument("--source", choices=("auto", "sqlite", "csv"), default="auto")
    parser.add_argument(
        "--run-policy",
        choices=("latest-per-target", "nearest-run"),
        default="latest-per-target",
        help="How to collapse many Windsurfice snapshots for the same target timestamp.",
    )
    parser.add_argument(
        "--tolerance-minutes",
        type=float,
        default=0.0,
        help="Allow nearest timestamp matching within this many minutes. Default requires exact target_ts.",
    )
    parser.add_argument(
        "--speed-conversion",
        choices=("auto", "none", "mps-to-knots", "knots-to-mps"),
        default="auto",
        help="Conversion applied to KNMI speed before computing speed differences.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tolerance = pd.Timedelta(minutes=float(args.tolerance_minutes))

    knmi = load_knmi_csv(args.knmi_csv)
    target_min = knmi["target_dt"].min()
    target_max = knmi["target_dt"].max()
    knmi_run_dt = knmi["run_dt"].dropna().iloc[0] if knmi["run_dt"].notna().any() else None

    candidates = discover_candidate_data_sources(args.data_dir, args.site)
    print("Discovered candidate Windsurfice data sources:")
    if candidates:
        for candidate in candidates:
            print(f"- {candidate.kind}: {candidate.path} ({candidate.detail})")
    else:
        print("- none")

    existing_data = load_existing_data(
        data_dir=args.data_dir,
        site=args.site,
        model=args.model,
        target_min=target_min,
        target_max=target_max,
        tolerance=tolerance,
        source_kind=args.source,
    )
    selected_existing, selection_note = select_existing_records(
        existing_data.frame,
        policy=args.run_policy,
        knmi_run_dt=knmi_run_dt,
    )

    comparison, speed_conversion_used, conversion_scores = match_and_compare(
        knmi,
        selected_existing,
        tolerance=tolerance,
        speed_conversion=args.speed_conversion,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False)

    print_summary(
        knmi_rows=len(knmi),
        existing_loaded_rows=existing_data.loaded_rows,
        existing_selected_rows=len(selected_existing),
        comparison=comparison,
        existing_data=existing_data,
        selection_note=selection_note,
        speed_conversion=speed_conversion_used,
        conversion_scores=conversion_scores,
        tolerance=tolerance,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
