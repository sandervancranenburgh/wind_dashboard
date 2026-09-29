"""Offline, read-only evaluation of HARMONIE P1 gustiness proxies.

The experiment intentionally does not train or publish a production model. It
matches rider-reported session gustiness to the latest P1 forecast that was
available before the session, evaluates a small set of predeclared feature
families with date-grouped cross-validation, and writes audit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABEL_TO_BINARY = {
    "very_steady": 0,
    "steady": 0,
    "moderate": 0,
    "gusty": 1,
    "very_gusty": 1,
}

FEATURE_SETS: dict[str, list[str]] = {
    "mean_wind": ["wind_speed_10m_mean"],
    "harmonie_gust": ["wind_gust_10m_mean"],
    "gust_excess": ["gust_excess_mean"],
    "tke_proxy": ["tke_proxy_mean"],
    "gust_factor_ti": ["ti_proxy_mean"],
    "shear_100_10": ["shear_exponent_100_10_mean"],
    "compact_gust_shear": ["gust_excess_mean", "shear_exponent_100_10_mean"],
}

PRIMARY_MODEL = "gust_excess"
BASELINE_MODEL = "mean_wind"


@dataclass(frozen=True)
class ExperimentConfig:
    site: str = "valkenburgsemeer"
    spot: str = "Valkenburgse meer"
    facraf: float = 3.8
    minimum_mean_wind_mps: float = 2.0
    bootstrap_repeats: int = 2_000
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate archived HARMONIE P1 gustiness proxies against rider session labels.",
    )
    parser.add_argument("--db", default="data/wind_data_all_sites.db", help="Production SQLite DB; opened immutable/read-only.")
    parser.add_argument("--site", default="valkenburgsemeer", help="P1 site identifier.")
    parser.add_argument("--spot", default="Valkenburgse meer", help="Session spot label.")
    parser.add_argument("--facraf", type=float, default=3.8, help="FACRAF scaling used only for the inferred-TKE proxy.")
    parser.add_argument(
        "--minimum-mean-wind-mps",
        type=float,
        default=2.0,
        help="Minimum U10 for normalized gust and shear ratios.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000, help="Date-cluster bootstrap repetitions.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic estimator/bootstrap seed.")
    parser.add_argument(
        "--out-dir",
        default="next_day_wind_model/artifacts/gustiness_experiment",
        help="Output directory for CSV, JSON, and PNG artifacts.",
    )
    return parser.parse_args()


def connect_immutable(db_path: Path | str) -> sqlite3.Connection:
    """Open SQLite with both URI-level and connection-level write protection."""
    resolved = Path(db_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Database not found: {resolved}")
    conn = sqlite3.connect(f"file:{resolved}?mode=ro&immutable=1", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _validate_schema(conn: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    required_tables = {"surf_experiences", "harmonie_knmi_features"}
    missing_tables = sorted(required_tables - tables)
    if missing_tables:
        raise ValueError(f"Database is missing required tables: {', '.join(missing_tables)}")

    required_columns = {
        "surf_experiences": {
            "id", "user_id", "spot", "date", "start_ts", "end_ts", "perceived_wind_variability",
        },
        "harmonie_knmi_features": {
            "site", "run_ts", "fetched_ts", "target_ts",
            "u_10m_mps", "v_10m_mps", "wind_speed_10m_mps", "wind_gust_10m_mps",
            "u_50m_mps", "v_50m_mps", "wind_speed_50m_mps",
            "u_100m_mps", "v_100m_mps", "wind_speed_100m_mps",
            "dir_shear_10m_100m",
        },
    }
    for table, expected in required_columns.items():
        present = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
        missing = sorted(expected - present)
        if missing:
            raise ValueError(f"Table {table} is missing required columns: {', '.join(missing)}")


def load_point_in_time_rows(
    db_path: Path | str,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Return the latest pre-session P1 vintage for each session/target hour."""
    conn = connect_immutable(db_path)
    try:
        _validate_schema(conn)
        frame = pd.read_sql_query(
            """
            WITH candidates AS (
                SELECT
                    e.id AS session_id,
                    e.user_id,
                    e.date AS session_date,
                    e.start_ts,
                    e.end_ts,
                    e.perceived_wind_variability,
                    h.run_ts,
                    h.fetched_ts,
                    h.target_ts,
                    h.u_10m_mps,
                    h.v_10m_mps,
                    h.wind_speed_10m_mps,
                    h.wind_gust_10m_mps,
                    h.u_50m_mps,
                    h.v_50m_mps,
                    h.wind_speed_50m_mps,
                    h.u_100m_mps,
                    h.v_100m_mps,
                    h.wind_speed_100m_mps,
                    h.dir_shear_10m_100m,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.id, h.target_ts
                        ORDER BY h.fetched_ts DESC, h.run_ts DESC
                    ) AS vintage_rank
                FROM surf_experiences AS e
                JOIN harmonie_knmi_features AS h
                  ON h.site = ?
                 AND CAST(strftime('%s', h.target_ts) AS INTEGER) * 1000
                     BETWEEN e.start_ts - 1800000 AND e.end_ts + 1800000
                 AND CAST(strftime('%s', h.fetched_ts) AS INTEGER) * 1000 <= e.start_ts
                WHERE e.spot = ?
                  AND e.perceived_wind_variability IN (
                      'very_steady', 'steady', 'moderate', 'gusty', 'very_gusty'
                  )
            )
            SELECT *
            FROM candidates
            WHERE vintage_rank = 1
            ORDER BY session_date, start_ts, target_ts
            """,
            conn,
            params=(cfg.site, cfg.spot),
        )
    finally:
        conn.close()
    return frame


def add_hourly_proxy_features(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Calculate physical proxies and centered-hour overlap weights."""
    if frame.empty:
        return frame.copy()
    if cfg.facraf <= 0:
        raise ValueError("facraf must be positive")
    if cfg.minimum_mean_wind_mps <= 0:
        raise ValueError("minimum_mean_wind_mps must be positive")

    out = frame.copy()
    numeric = [
        "u_10m_mps", "v_10m_mps", "wind_speed_10m_mps", "wind_gust_10m_mps",
        "u_50m_mps", "v_50m_mps", "wind_speed_50m_mps",
        "u_100m_mps", "v_100m_mps", "wind_speed_100m_mps", "dir_shear_10m_100m",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    u10 = out["wind_speed_10m_mps"]
    gust = out["wind_gust_10m_mps"]
    delta = (gust - u10).clip(lower=0.0)
    normalized_ok = u10 >= float(cfg.minimum_mean_wind_mps)
    out["gust_excess"] = delta
    out["tke_proxy"] = np.square(delta / float(cfg.facraf))
    out["gust_factor"] = np.where(normalized_ok, gust / u10, np.nan)
    out["ti_proxy"] = np.where(
        normalized_ok,
        math.sqrt(2.0 / 3.0) * np.sqrt(out["tke_proxy"]) / u10,
        np.nan,
    )
    out["force_excess_abs"] = np.square(gust) - np.square(u10)
    out["force_excess_relative"] = np.where(
        normalized_ok,
        out["force_excess_abs"] / np.square(u10),
        np.nan,
    )
    ratio50_ok = normalized_ok & (out["wind_speed_50m_mps"] > 0)
    ratio100_ok = normalized_ok & (out["wind_speed_100m_mps"] > 0)
    out["shear_ratio_50_10"] = np.where(ratio50_ok, out["wind_speed_50m_mps"] / u10, np.nan)
    out["shear_ratio_100_10"] = np.where(ratio100_ok, out["wind_speed_100m_mps"] / u10, np.nan)
    out["shear_exponent_50_10"] = np.where(
        ratio50_ok,
        np.log(out["wind_speed_50m_mps"] / u10) / math.log(5.0),
        np.nan,
    )
    out["shear_exponent_100_10"] = np.where(
        ratio100_ok,
        np.log(out["wind_speed_100m_mps"] / u10) / math.log(10.0),
        np.nan,
    )
    out["vector_shear_50_10"] = np.hypot(
        out["u_50m_mps"] - out["u_10m_mps"],
        out["v_50m_mps"] - out["v_10m_mps"],
    )
    out["vector_shear_100_10"] = np.hypot(
        out["u_100m_mps"] - out["u_10m_mps"],
        out["v_100m_mps"] - out["v_10m_mps"],
    )

    target_datetimes = pd.to_datetime(out["target_ts"], utc=True, errors="coerce")
    target_ms = target_datetimes.map(
        lambda value: np.nan if pd.isna(value) else int(value.timestamp() * 1000)
    )
    interval_start = target_ms - 30 * 60 * 1000
    interval_end = target_ms + 30 * 60 * 1000
    overlap_start = np.maximum(interval_start, out["start_ts"].astype(np.int64))
    overlap_end = np.minimum(interval_end, out["end_ts"].astype(np.int64))
    out["overlap_seconds"] = np.maximum(overlap_end - overlap_start, 0) / 1000.0
    out["gust_below_mean_flag"] = (gust < u10).astype(np.int8)
    return out[out["overlap_seconds"] > 0].copy()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))


def _anonymous_groups(values: Iterable[object], prefix: str) -> dict[object, str]:
    unique = sorted(set(values), key=lambda value: str(value))
    return {value: f"{prefix}{idx:03d}" for idx, value in enumerate(unique, start=1)}


def build_session_dataset(hourly: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if hourly.empty:
        raise ValueError("No point-in-time P1 rows overlap the labeled sessions.")

    feature_columns = [
        "wind_speed_10m_mps", "wind_gust_10m_mps", "gust_excess", "tke_proxy",
        "gust_factor", "ti_proxy", "force_excess_abs", "force_excess_relative",
        "shear_ratio_50_10", "shear_ratio_100_10",
        "shear_exponent_50_10", "shear_exponent_100_10",
        "vector_shear_50_10", "vector_shear_100_10", "dir_shear_10m_100m",
    ]
    session_rows: list[dict[str, object]] = []
    for session_id, part in hourly.groupby("session_id", sort=False):
        first = part.iloc[0]
        row: dict[str, object] = {
            "session_id_internal": int(session_id),
            "user_id_internal": int(first["user_id"]),
            "session_date_internal": str(first["session_date"]),
            "perceived_wind_variability": str(first["perceived_wind_variability"]),
            "target_hour_count": int(len(part)),
            "forecast_run_count": int(part["run_ts"].nunique()),
            "gust_below_mean_count": int(part["gust_below_mean_flag"].sum()),
        }
        weights = part["overlap_seconds"]
        for feature in feature_columns:
            row[f"{feature.removesuffix('_mps')}_mean"] = _weighted_mean(part[feature], weights)
            row[f"{feature.removesuffix('_mps')}_max"] = float(part[feature].max(skipna=True))
        session_rows.append(row)

    sessions = pd.DataFrame.from_records(session_rows)
    sessions["target"] = sessions["perceived_wind_variability"].map(LABEL_TO_BINARY)
    sessions = sessions.dropna(subset=["target"]).copy()
    sessions["target"] = sessions["target"].astype(np.int8)

    date_map = _anonymous_groups(sessions["session_date_internal"], "D")
    rider_map = _anonymous_groups(sessions["user_id_internal"], "R")
    sessions["date_group"] = sessions["session_date_internal"].map(date_map)
    sessions["rider_group"] = sessions["user_id_internal"].map(rider_map)
    sessions["session_key"] = sessions["session_id_internal"].map(
        lambda value: "S" + hashlib.sha256(f"gustiness-session:{value}".encode()).hexdigest()[:10]
    )
    return sessions.sort_values(["session_date_internal", "session_id_internal"]).reset_index(drop=True)


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, probabilities))


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    prediction = (probabilities >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n_sessions": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "roc_auc": _safe_auc(y_true, probabilities),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def grouped_oof_predictions(
    sessions: pd.DataFrame,
    feature_sets: dict[str, Sequence[str]] = FEATURE_SETS,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    y = sessions["target"].to_numpy(dtype=np.int8)
    groups = sessions["date_group"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Both gusty and non-gusty sessions are required.")
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three distinct session dates are required for grouped validation.")

    output = sessions[["session_key", "date_group", "rider_group", "target"]].copy()
    splitter = LeaveOneGroupOut()
    for model_name, columns in feature_sets.items():
        missing = [column for column in columns if column not in sessions]
        if missing:
            raise ValueError(f"Feature set {model_name} is missing columns: {', '.join(missing)}")
        x = sessions[list(columns)].replace([np.inf, -np.inf], np.nan)
        if x.isna().any().any():
            missing_rows = int(x.isna().any(axis=1).sum())
            raise ValueError(f"Feature set {model_name} has missing values in {missing_rows} sessions.")

        probabilities = np.full(len(sessions), np.nan, dtype=float)
        for train_idx, test_idx in splitter.split(x, y, groups):
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                probabilities[test_idx] = float(y_train.mean())
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=int(seed),
                ),
            )
            model.fit(x.iloc[train_idx], y_train)
            probabilities[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
        output[f"prob_{model_name}"] = probabilities
    return output


def _cluster_bootstrap(
    oof: pd.DataFrame,
    model_names: Sequence[str],
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    if repeats < 1:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    groups = np.array(sorted(oof["date_group"].unique()), dtype=object)
    by_group = {group: oof.index[oof["date_group"] == group].to_numpy() for group in groups}
    records: list[dict[str, float | int]] = []
    attempts = 0
    max_attempts = max(repeats * 10, 100)
    while len(records) < repeats and attempts < max_attempts:
        attempts += 1
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled_groups])
        sample = oof.loc[indices]
        y = sample["target"].to_numpy(dtype=np.int8)
        if len(np.unique(y)) < 2:
            continue
        record: dict[str, float | int] = {"replicate": len(records) + 1}
        for model_name in model_names:
            probabilities = sample[f"prob_{model_name}"].to_numpy(dtype=float)
            record[f"auc_{model_name}"] = float(roc_auc_score(y, probabilities))
            record[f"balanced_accuracy_{model_name}"] = float(
                balanced_accuracy_score(y, probabilities >= 0.5)
            )
        record["auc_gain_primary_vs_baseline"] = (
            float(record[f"auc_{PRIMARY_MODEL}"]) - float(record[f"auc_{BASELINE_MODEL}"])
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _interval(values: pd.Series) -> tuple[float | None, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None, None
    low, high = np.quantile(clean.to_numpy(dtype=float), [0.025, 0.975])
    return float(low), float(high)


def summarize_evaluation(
    sessions: pd.DataFrame,
    oof: pd.DataFrame,
    bootstrap: pd.DataFrame,
    feature_sets: dict[str, Sequence[str]] = FEATURE_SETS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metric_rows: list[dict[str, object]] = []
    y = oof["target"].to_numpy(dtype=np.int8)
    for model_name, columns in feature_sets.items():
        row: dict[str, object] = {
            "model": model_name,
            "features": ",".join(columns),
            **_classification_metrics(y, oof[f"prob_{model_name}"].to_numpy(dtype=float)),
        }
        if not bootstrap.empty:
            row["roc_auc_ci_low"], row["roc_auc_ci_high"] = _interval(bootstrap[f"auc_{model_name}"])
            row["balanced_accuracy_ci_low"], row["balanced_accuracy_ci_high"] = _interval(
                bootstrap[f"balanced_accuracy_{model_name}"]
            )
        metric_rows.append(row)
    metrics = pd.DataFrame.from_records(metric_rows)

    metric_index = metrics.set_index("model")
    primary_auc = float(metric_index.loc[PRIMARY_MODEL, "roc_auc"])
    baseline_auc = float(metric_index.loc[BASELINE_MODEL, "roc_auc"])
    primary_balanced_accuracy = float(metric_index.loc[PRIMARY_MODEL, "balanced_accuracy"])
    auc_gain = primary_auc - baseline_auc

    rider_results: list[dict[str, object]] = []
    for rider_group, part in oof.groupby("rider_group"):
        rider_y = part["target"].to_numpy(dtype=np.int8)
        if len(part) < 8 or len(np.unique(rider_y)) < 2:
            continue
        auc = _safe_auc(rider_y, part[f"prob_{PRIMARY_MODEL}"].to_numpy(dtype=float))
        rider_results.append({"rider_group": rider_group, "n": int(len(part)), "roc_auc": auc})
    rider_direction_consistent = bool(rider_results) and all(float(row["roc_auc"]) >= 0.5 for row in rider_results)

    ci_gain_low, ci_gain_high = (None, None)
    if not bootstrap.empty:
        ci_gain_low, ci_gain_high = _interval(bootstrap["auc_gain_primary_vs_baseline"])

    criteria = {
        "primary_auc_at_least_0_70": primary_auc >= 0.70,
        "auc_gain_vs_mean_wind_at_least_0_05": auc_gain >= 0.05,
        "primary_balanced_accuracy_at_least_0_65": primary_balanced_accuracy >= 0.65,
        "direction_consistent_for_riders_with_8_sessions": rider_direction_consistent,
    }
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline_shadow_evaluation_only",
        "target_definition": {
            "negative": ["very_steady", "steady", "moderate"],
            "positive": ["gusty", "very_gusty"],
        },
        "n_sessions": int(len(sessions)),
        "n_dates": int(sessions["date_group"].nunique()),
        "n_riders": int(sessions["rider_group"].nunique()),
        "n_positive": int(sessions["target"].sum()),
        "label_counts": {
            str(key): int(value)
            for key, value in sessions["perceived_wind_variability"].value_counts().sort_index().items()
        },
        "primary_model": PRIMARY_MODEL,
        "baseline_model": BASELINE_MODEL,
        "primary_auc": primary_auc,
        "baseline_auc": baseline_auc,
        "auc_gain_primary_vs_baseline": auc_gain,
        "auc_gain_ci_low": ci_gain_low,
        "auc_gain_ci_high": ci_gain_high,
        "primary_balanced_accuracy": primary_balanced_accuracy,
        "rider_sensitivity": rider_results,
        "advancement_criteria": criteria,
        "passes_point_estimate_criteria": bool(all(criteria.values())),
        "production_decision": "do_not_deploy; collect more labels and run prospective shadow validation",
        "interpretation_notes": [
            "TKE proxy is algebraically equivalent in ranking to positive absolute gust excess.",
            "TI proxy is a rescaled gust factor and does not independently measure turbulence.",
            "Hourly P1 gust output supports gust-severity evaluation, not lull-frequency claims.",
        ],
    }
    return metrics, summary


def _public_session_columns(sessions: pd.DataFrame) -> list[str]:
    excluded = {"session_id_internal", "user_id_internal", "session_date_internal"}
    return [column for column in sessions.columns if column not in excluded]


def save_diagnostic_plot(sessions: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ordered = metrics.sort_values("roc_auc", ascending=True)
    axes[0].barh(ordered["model"], ordered["roc_auc"], color="#247ba0")
    axes[0].axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].axvline(0.7, color="#2a9d8f", linestyle=":", linewidth=1.3)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Leave-one-date-out ROC AUC")
    axes[0].set_title("Candidate decision rules")

    negative = sessions.loc[sessions["target"] == 0, "gust_excess_mean"].to_numpy(dtype=float)
    positive = sessions.loc[sessions["target"] == 1, "gust_excess_mean"].to_numpy(dtype=float)
    axes[1].boxplot([negative, positive], tick_labels=["Steady/moderate", "Gusty/very gusty"])
    rng = np.random.default_rng(7)
    for idx, values in enumerate((negative, positive), start=1):
        jitter = rng.normal(0.0, 0.035, size=len(values))
        axes[1].scatter(np.full(len(values), idx) + jitter, values, alpha=0.7, s=28, color="#f25f5c")
    axes[1].set_ylabel("Session-weighted HARMONIE gust excess (m/s)")
    axes[1].set_title("Primary proxy by rider report")
    fig.suptitle("Offline HARMONIE P1 gustiness experiment")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    db_path: Path | str,
    out_dir: Path | str,
    cfg: ExperimentConfig,
) -> dict[str, object]:
    raw = load_point_in_time_rows(db_path, cfg)
    hourly = add_hourly_proxy_features(raw, cfg)
    sessions = build_session_dataset(hourly, cfg)
    oof = grouped_oof_predictions(sessions, seed=cfg.seed)
    bootstrap = _cluster_bootstrap(
        oof,
        list(FEATURE_SETS),
        repeats=int(cfg.bootstrap_repeats),
        seed=int(cfg.seed),
    )
    metrics, summary = summarize_evaluation(sessions, oof, bootstrap)
    summary["configuration"] = {
        "site": cfg.site,
        "spot": cfg.spot,
        "facraf": cfg.facraf,
        "minimum_mean_wind_mps": cfg.minimum_mean_wind_mps,
        "bootstrap_repeats": cfg.bootstrap_repeats,
        "seed": cfg.seed,
        "database_open_mode": "mode=ro&immutable=1; PRAGMA query_only=ON",
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    sessions[_public_session_columns(sessions)].to_csv(output / "gustiness_session_features.csv", index=False)
    oof.to_csv(output / "gustiness_oof_predictions.csv", index=False)
    metrics.to_csv(output / "gustiness_model_metrics.csv", index=False)
    if not bootstrap.empty:
        bootstrap.to_csv(output / "gustiness_cluster_bootstrap.csv", index=False)
    with (output / "gustiness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    save_diagnostic_plot(sessions, metrics, output / "gustiness_diagnostics.png")
    return summary


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        site=args.site,
        spot=args.spot,
        facraf=float(args.facraf),
        minimum_mean_wind_mps=float(args.minimum_mean_wind_mps),
        bootstrap_repeats=int(args.bootstrap_repeats),
        seed=int(args.seed),
    )
    summary = run_experiment(Path(args.db), Path(args.out_dir), cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
