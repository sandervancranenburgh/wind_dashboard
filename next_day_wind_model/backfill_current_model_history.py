from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_pipeline import DatasetConfig, _apply_standardizer, build_all_training_arrays
from db_store import init_db, log_prediction_batch, materialize_prediction_log_evaluation
from train_lstm import NextDayLSTM
from update_model_and_predict import _build_speed_calibration_context, _predict_speed_batch


LSTM_HIGHLIGHT_COLOR = "#d7191c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical day-level MAE using the current trained next-day model.",
    )
    parser.add_argument("--db", default="data/wind_data.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument(
        "--metadata",
        default="next_day_wind_model/artifacts/metadata_update.json",
        help="Metadata JSON from latest model update (used for window/target defaults).",
    )
    parser.add_argument(
        "--model-path",
        default="next_day_wind_model/artifacts/next_day_lstm_speed_residual.pt",
        help="Path to trained speed model checkpoint.",
    )
    parser.add_argument("--x-mean", default="next_day_wind_model/artifacts/x_mean_speed.npy", help="Path to X mean scaler.")
    parser.add_argument("--x-std", default="next_day_wind_model/artifacts/x_std_speed.npy", help="Path to X std scaler.")
    parser.add_argument("--y-mean", default="next_day_wind_model/artifacts/y_mean_speed.npy", help="Path to y mean scaler.")
    parser.add_argument("--y-std", default="next_day_wind_model/artifacts/y_std_speed.npy", help="Path to y std scaler.")
    parser.add_argument("--window-hours", type=int, default=None, help="Override input window size.")
    parser.add_argument("--target-hours", type=int, default=None, help="Override prediction horizon.")
    parser.add_argument("--local-timezone", default="Europe/Amsterdam", help="Timezone for daily grouping/plot.")
    parser.add_argument("--out-csv", default="next_day_wind_model/artifacts/dayahead_backfill_history.csv", help="Output CSV.")
    parser.add_argument("--out-png", default="next_day_wind_model/artifacts/dayahead_backfill_history.png", help="Output PNG.")
    return parser.parse_args()


def _load_model(path: Path) -> tuple[NextDayLSTM, dict]:
    device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NextDayLSTM(
        n_features=int(ckpt["n_features"]),
        target_hours=int(ckpt["target_hours"]),
        output_activation=str(ckpt.get("output_activation", "linear")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _plot_daily_mae(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot(df["day_local"], df["mae_lstm"], color=LSTM_HIGHLIGHT_COLOR, linewidth=2.2, label="Super local vs measured wind")
    ax.plot(df["day_local"], df["mae_forecast"], color="gray", linewidth=1.8, label="Harmonie vs measured wind")
    ax.set_title("Backfilled Frozen Day-ahead MAE (Current Model on Past Data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("MAE (kts)")
    ax.set_ylim(0.0, max(4.0, float(np.nanmax([df["mae_lstm"].max(), df["mae_forecast"].max()])) * 1.06))
    ax.margins(x=0, y=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    avg_lstm = float(df["mae_lstm"].mean())
    avg_fc = float(df["mae_forecast"].mean())
    ax.text(
        0.985,
        1.02,
        f"Average MAE in shown period\nSuper local: {avg_lstm:.2f} kts\nHarmonie: {avg_fc:.2f} kts",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="black",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _inverse_standardizer(X_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return X_scaled * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def _build_frozen_backfill_prediction_detail_frame(
    db_path: Path,
    cfg: DatasetConfig,
    model: NextDayLSTM,
    ckpt: dict,
    x_mean_saved: np.ndarray,
    x_std_saved: np.ndarray,
    y_mean_saved: float,
    y_std_saved: float,
    local_tz: str,
) -> pd.DataFrame:
    speed_arrays = build_all_training_arrays(db_path, cfg, target_mode="residual")
    if int(speed_arrays["X_all"].shape[0]) == 0:
        raise ValueError("No historical canonical samples available for backfill.")

    X_raw = _inverse_standardizer(
        np.asarray(speed_arrays["X_all"], dtype=np.float32),
        np.asarray(speed_arrays["x_mean"], dtype=np.float32),
        np.asarray(speed_arrays["x_std"], dtype=np.float32),
    ).astype(np.float32)
    X_scaled = _apply_standardizer(X_raw, x_mean_saved, x_std_saved).astype(np.float32)

    y_actual_raw = np.asarray(speed_arrays["y_actual_all_raw"], dtype=np.float32)
    y_forecast_raw = np.asarray(speed_arrays["y_forecast_all_raw"], dtype=np.float32)
    anchor_times_utc = pd.to_datetime(speed_arrays["timestamps"], utc=True, errors="coerce")
    target_times_all = np.asarray(speed_arrays["target_times_all"], dtype=object)
    target_run_ts_all = np.asarray(speed_arrays["target_run_ts_all"], dtype=np.int64)
    target_fetched_ts_all = np.asarray(speed_arrays["target_fetched_ts_all"], dtype=np.int64)
    target_horizon_hr_all = np.asarray(speed_arrays["target_horizon_hr_all"], dtype=np.float32)
    if anchor_times_utc.isna().any():
        raise ValueError("Canonical backfill samples contain invalid anchor timestamps.")

    target_start_utc = anchor_times_utc + pd.Timedelta(hours=1)
    frozen_mask = np.asarray(target_start_utc.hour, dtype=np.int16) == 0
    if not frozen_mask.any():
        raise ValueError("No day-start canonical samples available for frozen day-ahead backfill.")

    target_mode = str(ckpt.get("target_mode", "residual")).strip().lower()
    constraint_eps = ckpt.get("constraint_eps", None)
    speed_calibration = ckpt.get("speed_regime_calibration")
    speed_calibration_context = _build_speed_calibration_context(
        anchor_dir_deg=X_raw[:, -1, 2],
        target_times_utc=target_start_utc,
    )
    pred_speed = _predict_speed_batch(
        model=model,
        X_input=X_scaled,
        forecast_speed=y_forecast_raw,
        y_mean=float(y_mean_saved),
        y_std=float(y_std_saved),
        target_mode=target_mode,
        constraint_eps=constraint_eps,
        speed_calibration=speed_calibration,
        speed_calibration_context=speed_calibration_context,
        device=torch.device("cpu"),
    )

    tz = ZoneInfo(local_tz)
    detail_rows: list[dict[str, object]] = []
    frozen_indices = np.flatnonzero(frozen_mask)
    if frozen_indices.size == 0:
        raise ValueError("No frozen day-ahead samples remained after canonical filtering.")

    for idx in frozen_indices.tolist():
        anchor_time = anchor_times_utc[idx]
        target_times = pd.to_datetime(target_times_all[idx], utc=True, errors="coerce")
        if target_times.isna().any():
            raise ValueError("Canonical backfill sample contains invalid target timestamps.")
        issue_local_time = anchor_time.tz_convert(tz)
        for step, target_time in enumerate(target_times):
            target_local = target_time.tz_convert(tz)
            detail_rows.append(
                {
                    "anchor_time_utc": anchor_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "anchor_time_local": issue_local_time.isoformat(),
                    "issued_time_utc": anchor_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "issued_time_local": issue_local_time.isoformat(),
                    "target_time_utc": target_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "target_time_local": target_local.isoformat(),
                    "target_day_local": target_local.strftime("%Y-%m-%d"),
                    "horizon_hr": float(target_horizon_hr_all[idx, step]),
                    "actual_value": float(y_actual_raw[idx, step]),
                    "harmonie_value": float(y_forecast_raw[idx, step]),
                    "prediction_value": float(pred_speed[idx, step]),
                    "harmonie_run_ts": int(target_run_ts_all[idx, step]),
                    "harmonie_fetched_ts": int(target_fetched_ts_all[idx, step]),
                    "evaluation_type": "day_ahead_frozen_backfill",
                }
            )

    detail_df = pd.DataFrame.from_records(detail_rows)
    if detail_df.empty:
        raise ValueError("No frozen day-ahead target rows were built for canonical backfill.")
    return detail_df


def _build_frozen_backfill_daily_frame(
    db_path: Path,
    cfg: DatasetConfig,
    model: NextDayLSTM,
    ckpt: dict,
    x_mean_saved: np.ndarray,
    x_std_saved: np.ndarray,
    y_mean_saved: float,
    y_std_saved: float,
    local_tz: str,
) -> pd.DataFrame:
    detail_df = _build_frozen_backfill_prediction_detail_frame(
        db_path=db_path,
        cfg=cfg,
        model=model,
        ckpt=ckpt,
        x_mean_saved=x_mean_saved,
        x_std_saved=x_std_saved,
        y_mean_saved=y_mean_saved,
        y_std_saved=y_std_saved,
        local_tz=local_tz,
    )
    daily_df = (
        detail_df.groupby("target_day_local", sort=True)
        .agg(
            anchor_time_utc=("anchor_time_utc", "first"),
            issue_local_time=("issued_time_local", "first"),
            target_start_utc=("target_time_utc", "first"),
            target_start_local=("target_time_local", "first"),
            day_local=("target_day_local", "first"),
            mae_forecast=("harmonie_value", lambda s: float(np.mean(np.abs(np.asarray(s, dtype=float) - detail_df.loc[s.index, "actual_value"].to_numpy(dtype=float))))),
            mae_lstm=("prediction_value", lambda s: float(np.mean(np.abs(np.asarray(s, dtype=float) - detail_df.loc[s.index, "actual_value"].to_numpy(dtype=float))))),
            avg_actual_wind_speed=("actual_value", lambda s: float(np.mean(np.asarray(s, dtype=float)))),
            avg_forecast_wind_speed=("harmonie_value", lambda s: float(np.mean(np.asarray(s, dtype=float)))),
            avg_lstm_wind_speed=("prediction_value", lambda s: float(np.mean(np.asarray(s, dtype=float)))),
            n_points=("actual_value", "count"),
        )
        .reset_index(drop=True)
    )
    daily_df["improvement_vs_forecast"] = daily_df["mae_forecast"] - daily_df["mae_lstm"]
    daily_df["evaluation_type"] = "day_ahead_frozen_backfill"
    daily_df["day_local"] = pd.to_datetime(daily_df["day_local"], errors="coerce")
    daily_df["date"] = daily_df["day_local"].dt.strftime("%Y-%m-%d")
    daily_df = daily_df.dropna(subset=["day_local"]).reset_index(drop=True)
    return daily_df


def backfill_next_day_prediction_log(
    db_path: Path,
    cfg: DatasetConfig,
    model: NextDayLSTM,
    ckpt: dict,
    x_mean_saved: np.ndarray,
    x_std_saved: np.ndarray,
    y_mean_saved: float,
    y_std_saved: float,
    local_tz: str,
    *,
    model_type: str = "next_day_backfill",
    model_artifact: str,
    model_version: str | None,
    model_name: str = "NextDayLSTM",
    prediction_kind: str = "wind_speed",
    run_context_prefix: str = "backfill_current_champion_report",
) -> tuple[int, int]:
    """
    Materialize historical frozen next-day champion rows into prediction_log.

    Each row represents one target timestamp from one canonical historical
    day-ahead issue context, with issued_ts set to that historical anchor time
    so reporting by issued day reflects the frozen operational context rather
    than the time this backfill was executed.
    """
    detail_df = _build_frozen_backfill_prediction_detail_frame(
        db_path=db_path,
        cfg=cfg,
        model=model,
        ckpt=ckpt,
        x_mean_saved=x_mean_saved,
        x_std_saved=x_std_saved,
        y_mean_saved=y_mean_saved,
        y_std_saved=y_std_saved,
        local_tz=local_tz,
    )
    rows: list[dict[str, object]] = []
    for rec in detail_df.itertuples(index=False):
        rows.append(
            {
                "site": cfg.site,
                "model_type": model_type,
                "prediction_kind": prediction_kind,
                "model_name": model_name,
                "model_version": model_version,
                "model_artifact": model_artifact,
                "issued_ts": int(pd.Timestamp(rec.issued_time_utc).timestamp() * 1000),
                "anchor_ts": int(pd.Timestamp(rec.anchor_time_utc).timestamp() * 1000),
                "target_ts": int(pd.Timestamp(rec.target_time_utc).timestamp() * 1000),
                "horizon_hr": float(rec.horizon_hr),
                "prediction_value": float(rec.prediction_value),
                "harmonie_value": float(rec.harmonie_value),
                "harmonie_run_ts": int(rec.harmonie_run_ts),
                "harmonie_fetched_ts": int(rec.harmonie_fetched_ts),
                "actual_value": float(rec.actual_value),
                "run_context": f"{run_context_prefix}:{str(rec.target_day_local)}",
                "metadata_json": {
                    "source": "backfill_current_model_history",
                    "forecast_model": cfg.model,
                    "evaluation_type": "day_ahead_frozen_backfill",
                },
            }
        )

    conn = sqlite3.connect(str(db_path))
    try:
        init_db(conn)
        inserted = log_prediction_batch(conn, rows)
        materialized = materialize_prediction_log_evaluation(
            conn,
            site=cfg.site,
            model_type=model_type,
            prediction_kind=prediction_kind,
        )
        return int(inserted), int(materialized)
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    window_hours = int(args.window_hours if args.window_hours is not None else metadata.get("window_hours", 72))
    target_hours = int(args.target_hours if args.target_hours is not None else metadata.get("target_hours", 24))

    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=window_hours,
        target_hours=target_hours,
    )

    model, ckpt = _load_model(Path(args.model_path))
    model_target_hours = int(ckpt.get("target_hours", cfg.target_hours))
    if model_target_hours != int(cfg.target_hours):
        raise ValueError(
            f"Configured target_hours={cfg.target_hours} does not match checkpoint target_hours={model_target_hours}."
        )

    x_mean = np.load(args.x_mean)
    x_std = np.load(args.x_std)
    y_mean = float(np.load(args.y_mean)[0])
    y_std = float(np.load(args.y_std)[0])
    daily_df = _build_frozen_backfill_daily_frame(
        db_path=Path(args.db),
        cfg=cfg,
        model=model,
        ckpt=ckpt,
        x_mean_saved=x_mean,
        x_std_saved=x_std,
        y_mean_saved=y_mean,
        y_std_saved=y_std,
        local_tz=args.local_timezone,
    )

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_csv, index=False, date_format="%Y-%m-%d")
    _plot_daily_mae(daily_df, out_png)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print(f"Rows (days): {len(daily_df)}")
    print(f"Average MAE Harmonie: {daily_df['mae_forecast'].mean():.3f} kts")
    print(f"Average MAE Super local: {daily_df['mae_lstm'].mean():.3f} kts")
    print(f"Average gain: {(daily_df['mae_forecast'].mean() - daily_df['mae_lstm'].mean()):.3f} kts")
    print(f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
