from __future__ import annotations

import copy
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import DatasetConfig, _load_observations, build_anchor_forecast_context


FEATURE_COLS = [
    "forecast_avg_t",
    "forecast_max_t",
    "forecast_avg_t1",
    "forecast_max_t1",
    "forecast_dir_t_sin",
    "forecast_dir_t_cos",
    "forecast_dir_t1_sin",
    "forecast_dir_t1_cos",
    "actual_t",
    "actual_t_1",
    "actual_t_2",
    "res_t",
    "res_t_1",
    "res_t_2",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

INTRADAY_CALIBRATION_HORIZON = 12


class IntradayResidualMLP(nn.Module):
    def __init__(self, n_features: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, int(hidden1)),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden1), int(hidden2)),
            nn.ReLU(),
            nn.Linear(int(hidden2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class IntradayBundle:
    model: IntradayResidualMLP
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    rollout_calibration: dict | None = None
    continuity_calibration: dict | None = None


@dataclass
class IntradayTrainParams:
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    hidden1: int = 128
    hidden2: int = 64
    dropout: float = 0.1
    learning_rate: float = 1e-3
    recency_power: float = 1.0


def _fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _load_hourly_observations(db_path: Path, cfg: DatasetConfig) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        obs = _load_observations(conn, cfg.site)
    finally:
        conn.close()
    return obs.sort_index()


def _build_intraday_anchor_contexts(
    db_path: Path,
    cfg: DatasetConfig,
    max_horizon: int = INTRADAY_CALIBRATION_HORIZON,
) -> list[dict]:
    obs = _load_hourly_observations(db_path, cfg)
    if obs.empty:
        raise ValueError("No rows available for intraday model training.")

    contexts: list[dict] = []
    total = len(obs)
    horizon = int(max_horizon)
    for i in range(2, total - horizon):
        anchor_time = obs.index[i]
        history_times = obs.index[i - 2 : i + 1]
        target_times = obs.index[i + 1 : i + 1 + horizon]
        context = build_anchor_forecast_context(
            db_path=db_path,
            cfg=cfg,
            anchor_time=anchor_time,
            history_times=history_times,
            target_times=target_times,
        )
        history_frame = context["history_frame"]
        target_frame = context["target_frame"]
        if history_frame is None or target_frame is None:
            continue

        actual_series = np.concatenate(
            [
                obs.iloc[i - 2 : i + 1]["actual_avg"].to_numpy(dtype=np.float32),
                obs.iloc[i + 1 : i + 1 + horizon]["actual_avg"].to_numpy(dtype=np.float32),
            ]
        ).astype(np.float32)
        forecast_avg_series = np.concatenate(
            [
                history_frame["forecast_avg"].to_numpy(dtype=np.float32),
                target_frame["forecast_avg"].to_numpy(dtype=np.float32),
            ]
        ).astype(np.float32)
        forecast_max_series = np.concatenate(
            [
                history_frame["forecast_max"].to_numpy(dtype=np.float32),
                target_frame["forecast_max"].to_numpy(dtype=np.float32),
            ]
        ).astype(np.float32)
        forecast_dir_series = np.concatenate(
            [
                history_frame["forecast_dir"].to_numpy(dtype=np.float32),
                target_frame["forecast_dir"].to_numpy(dtype=np.float32),
            ]
        ).astype(np.float32)
        if (
            np.isnan(actual_series).any()
            or np.isnan(forecast_avg_series).any()
            or np.isnan(forecast_max_series).any()
            or np.isnan(forecast_dir_series).any()
        ):
            continue
        contexts.append(
            {
                "anchor_time": anchor_time,
                "actual_series": actual_series,
                "forecast_avg_series": forecast_avg_series,
                "forecast_max_series": forecast_max_series,
                "forecast_dir_series": forecast_dir_series,
            }
        )

    if len(contexts) < 50:
        raise ValueError("Not enough rows to train intraday model.")
    return contexts


def _month_cycle(ts: pd.Timestamp) -> tuple[float, float]:
    angle = (2.0 * np.pi * (float(ts.month) - 1.0)) / 12.0
    return float(np.sin(angle)), float(np.cos(angle))


def _build_intraday_feature_row_from_context(
    context: dict,
    realized_series: np.ndarray,
    step: int,
) -> np.ndarray:
    forecast_avg = np.asarray(context["forecast_avg_series"], dtype=np.float32)
    forecast_max = np.asarray(context["forecast_max_series"], dtype=np.float32)
    forecast_dir = np.asarray(context["forecast_dir_series"], dtype=np.float32)
    actual = np.asarray(realized_series, dtype=np.float32)

    curr_idx = int(step + 2)
    next_idx = int(step + 3)
    current_time = pd.Timestamp(context["anchor_time"]) + pd.Timedelta(hours=int(step))
    month_sin, month_cos = _month_cycle(current_time)
    hour_ang = 2.0 * np.pi * (float(current_time.hour) / 24.0)

    fc_avg_t = float(forecast_avg[curr_idx])
    fc_avg_t1 = float(forecast_avg[next_idx])
    fc_max_t = float(forecast_max[curr_idx])
    fc_max_t1 = float(forecast_max[next_idx])
    fc_dir_t = float(forecast_dir[curr_idx])
    fc_dir_t1 = float(forecast_dir[next_idx])
    a_t = float(actual[curr_idx])
    a_t_1 = float(actual[curr_idx - 1])
    a_t_2 = float(actual[curr_idx - 2])
    res_t = a_t - fc_avg_t
    res_t_1 = a_t_1 - float(forecast_avg[curr_idx - 1])
    res_t_2 = a_t_2 - float(forecast_avg[curr_idx - 2])

    return np.array(
        [
            fc_avg_t,
            fc_max_t,
            fc_avg_t1,
            fc_max_t1,
            np.sin(np.deg2rad(fc_dir_t)),
            np.cos(np.deg2rad(fc_dir_t)),
            np.sin(np.deg2rad(fc_dir_t1)),
            np.cos(np.deg2rad(fc_dir_t1)),
            a_t,
            a_t_1,
            a_t_2,
            res_t,
            res_t_1,
            res_t_2,
            np.sin(hour_ang),
            np.cos(hour_ang),
            month_sin,
            month_cos,
        ],
        dtype=np.float32,
    )


def build_intraday_training_xy(
    db_path: Path,
    cfg: DatasetConfig,
    contexts: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    contexts = _build_intraday_anchor_contexts(db_path, cfg) if contexts is None else contexts

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for context in contexts:
        actual_series = np.asarray(context["actual_series"], dtype=np.float32)
        forecast_avg = np.asarray(context["forecast_avg_series"], dtype=np.float32)
        x = _build_intraday_feature_row_from_context(context, actual_series[:3], step=0)
        y = float(actual_series[3] - forecast_avg[3])
        if np.isnan(x).any() or np.isnan(y):
            continue
        X_list.append(x)
        y_list.append(y)

    if not X_list:
        raise ValueError("No intraday training samples after filtering.")
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def _fit_intraday_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: IntradayTrainParams,
    device: torch.device,
) -> tuple[IntradayBundle, float]:
    x_mean, x_std = _fit_standardizer(X_train)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) if float(y_train.std()) > 0 else 1.0

    X_train_s = (X_train - x_mean) / x_std
    X_val_s = (X_val - x_mean) / x_std
    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std

    n_train = len(X_train_s)
    if n_train <= 1:
        raise ValueError("Not enough training samples for intraday fit.")
    rec_idx = np.linspace(0.0, 1.0, n_train, dtype=np.float32)
    sample_w = np.power(1.0 + rec_idx, float(params.recency_power)).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_s).float(),
            torch.from_numpy(y_train_s).float(),
            torch.from_numpy(sample_w).float(),
        ),
        batch_size=int(params.batch_size),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_s).float(), torch.from_numpy(y_val_s).float()),
        batch_size=int(params.batch_size),
        shuffle=False,
    )

    model = IntradayResidualMLP(
        n_features=X_train.shape[1],
        hidden1=int(params.hidden1),
        hidden2=int(params.hidden2),
        dropout=float(params.dropout),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(params.learning_rate))
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    no_improve = 0
    crit = nn.MSELoss(reduction="none")

    for _ in range(int(params.epochs)):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss_vec = crit(pred, yb)
            loss = (loss_vec * wb).sum() / (wb.sum() + 1e-8)
            loss.backward()
            opt.step()

        model.eval()
        v_sum = 0.0
        v_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                l = nn.functional.mse_loss(model(xb), yb, reduction="mean")
                v_sum += float(l.item()) * xb.size(0)
                v_n += xb.size(0)
        val_loss = v_sum / max(v_n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 8:
            break

    model.load_state_dict(best_state)
    bundle = IntradayBundle(model=model, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    return bundle, float(best_val)


def _predict_intraday_step(
    bundle: IntradayBundle,
    x: np.ndarray,
    forecast_next: float,
    device: torch.device,
) -> float:
    x_s = (x - bundle.x_mean) / bundle.x_std
    with torch.no_grad():
        pred_res_s = float(bundle.model(torch.from_numpy(x_s).float().unsqueeze(0).to(device)).cpu().numpy()[0])
    pred_res = pred_res_s * bundle.y_std + bundle.y_mean
    return max(0.0, float(forecast_next + pred_res))


def _build_intraday_feature_row(
    frame: pd.DataFrame,
    actual_series: np.ndarray,
    i: int,
) -> np.ndarray:
    fc_avg = frame["forecast_avg"].to_numpy(dtype=np.float32)
    fc_max = frame["forecast_max"].to_numpy(dtype=np.float32)
    fc_dir = frame["forecast_dir"].to_numpy(dtype=np.float32)
    month_sin = frame["month_sin"].to_numpy(dtype=np.float32)
    month_cos = frame["month_cos"].to_numpy(dtype=np.float32)
    timestamp = frame.index[i]
    res_t = actual_series[i] - fc_avg[i]
    res_t_1 = actual_series[i - 1] - fc_avg[i - 1]
    res_t_2 = actual_series[i - 2] - fc_avg[i - 2]
    hour_ang = 2.0 * np.pi * (float(timestamp.hour) / 24.0)
    return np.array(
        [
            fc_avg[i],
            fc_max[i],
            fc_avg[i + 1],
            fc_max[i + 1],
            np.sin(np.deg2rad(fc_dir[i])),
            np.cos(np.deg2rad(fc_dir[i])),
            np.sin(np.deg2rad(fc_dir[i + 1])),
            np.cos(np.deg2rad(fc_dir[i + 1])),
            actual_series[i],
            actual_series[i - 1],
            actual_series[i - 2],
            res_t,
            res_t_1,
            res_t_2,
            np.sin(hour_ang),
            np.cos(hour_ang),
            month_sin[i],
            month_cos[i],
        ],
        dtype=np.float32,
    )


def _simulate_intraday_recursive_forecast_from_context(
    bundle: IntradayBundle,
    context: dict,
    device: torch.device,
    max_horizon: int,
) -> np.ndarray:
    if max_horizon <= 0:
        return np.array([], dtype=np.float32)

    forecast_avg = np.asarray(context["forecast_avg_series"], dtype=np.float32)
    actual_series = np.asarray(context["actual_series"], dtype=np.float32)
    usable = min(int(max_horizon), len(forecast_avg) - 3, len(actual_series) - 3)
    if usable <= 0:
        return np.array([], dtype=np.float32)

    recursive_actual = actual_series[:3].astype(np.float32).copy()
    preds: list[float] = []
    for step in range(usable):
        x = _build_intraday_feature_row_from_context(context, recursive_actual, step)
        pred_speed = _predict_intraday_step(bundle, x, forecast_next=float(forecast_avg[step + 3]), device=device)
        pred_speed = _apply_intraday_rollout_calibration(
            pred_speed=pred_speed,
            forecast_speed=float(forecast_avg[step + 3]),
            future_horizon=step + 1,
            rollout_calibration=bundle.rollout_calibration,
        )
        recursive_actual = np.append(recursive_actual, np.float32(pred_speed))
        preds.append(float(pred_speed))
    return np.asarray(preds, dtype=np.float32)


def _fit_intraday_rollout_calibration(
    bundle: IntradayBundle,
    contexts: list[dict],
    sample_split_idx: int,
    device: torch.device,
    max_horizon: int = 12,
) -> dict | None:
    if sample_split_idx < 20:
        return None

    eval_contexts = contexts[sample_split_idx:]
    if not eval_contexts:
        return None

    rows: list[dict] = []
    for context in eval_contexts:
        forecast_avg = np.asarray(context["forecast_avg_series"], dtype=np.float32)
        actual_series = np.asarray(context["actual_series"], dtype=np.float32)
        usable = min(int(max_horizon), len(forecast_avg) - 3, len(actual_series) - 3)
        if usable <= 0:
            continue

        recursive_actual = actual_series[:3].astype(np.float32).copy()
        for step in range(usable):
            x = _build_intraday_feature_row_from_context(context, recursive_actual, step)
            pred_speed = _predict_intraday_step(bundle, x, forecast_next=float(forecast_avg[step + 3]), device=device)
            recursive_actual = np.append(recursive_actual, np.float32(pred_speed))
            rows.append(
                {
                    "horizon": step + 1,
                    "forecast": float(forecast_avg[step + 3]),
                    "pred_recursive": float(pred_speed),
                    "actual": float(actual_series[step + 3]),
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    weights: list[float] = []
    prev_weight = 1.0
    for horizon in range(1, max_horizon + 1):
        subset = df[df["horizon"] == horizon]
        if subset.empty or horizon == 1:
            weights.append(prev_weight)
            continue
        forecast = subset["forecast"].to_numpy(dtype=np.float32)
        pred = subset["pred_recursive"].to_numpy(dtype=np.float32)
        actual_vals = subset["actual"].to_numpy(dtype=np.float32)
        best_weight = prev_weight
        best_mae = float(np.mean(np.abs(np.maximum(0.0, forecast + prev_weight * (pred - forecast)) - actual_vals)))
        for weight in np.arange(0.8, prev_weight + 0.001, 0.05):
            cand = np.maximum(0.0, forecast + float(weight) * (pred - forecast))
            mae = float(np.mean(np.abs(cand - actual_vals)))
            if mae + 1e-9 < best_mae:
                best_mae = mae
                best_weight = float(weight)
        weights.append(best_weight)
        prev_weight = best_weight

    weights_arr = np.asarray(weights, dtype=np.float32)
    horizon_idx = df["horizon"].to_numpy(dtype=int) - 1
    calibrated = np.maximum(
        0.0,
        df["forecast"].to_numpy(dtype=np.float32)
        + weights_arr[horizon_idx] * (df["pred_recursive"].to_numpy(dtype=np.float32) - df["forecast"].to_numpy(dtype=np.float32)),
    )
    baseline_mae = float(np.mean(np.abs(df["pred_recursive"].to_numpy(dtype=np.float32) - df["actual"].to_numpy(dtype=np.float32))))
    calibrated_mae = float(np.mean(np.abs(calibrated - df["actual"].to_numpy(dtype=np.float32))))
    improvement_abs = baseline_mae - calibrated_mae
    if improvement_abs <= 1e-4:
        return None
    return {
        "enabled": True,
        "max_horizon": int(max_horizon),
        "weights": [float(w) for w in weights_arr.tolist()],
        "baseline_mae": float(baseline_mae),
        "calibrated_mae": float(calibrated_mae),
        "improvement_abs": float(improvement_abs),
        "improvement_pct": float(improvement_abs / max(baseline_mae, 1e-6)),
        "n_rows": int(len(df)),
    }


def _apply_intraday_rollout_calibration(
    pred_speed: float,
    forecast_speed: float,
    future_horizon: int,
    rollout_calibration: dict | None,
) -> float:
    if future_horizon <= 0 or not rollout_calibration or not bool(rollout_calibration.get("enabled", False)):
        return float(pred_speed)
    weights = rollout_calibration.get("weights") or []
    if not weights:
        return float(pred_speed)
    horizon_idx = min(int(future_horizon) - 1, len(weights) - 1)
    weight = float(weights[horizon_idx])
    return max(0.0, float(forecast_speed) + weight * (float(pred_speed) - float(forecast_speed)))


def _fit_intraday_continuity_calibration(
    bundle: IntradayBundle,
    contexts: list[dict],
    sample_split_idx: int,
    device: torch.device,
    max_horizon: int = 12,
) -> dict | None:
    if sample_split_idx < 24:
        return None

    if sample_split_idx >= len(contexts):
        return None

    rows: list[dict] = []
    for idx in range(max(int(sample_split_idx), 1), len(contexts)):
        prev_context = contexts[idx - 1]
        curr_context = contexts[idx]
        prev_anchor = pd.Timestamp(prev_context["anchor_time"])
        curr_anchor = pd.Timestamp(curr_context["anchor_time"])
        if curr_anchor - prev_anchor != pd.Timedelta(hours=1):
            continue

        prev_preds = _simulate_intraday_recursive_forecast_from_context(
            bundle=bundle,
            context=prev_context,
            device=device,
            max_horizon=max_horizon + 1,
        )
        curr_preds = _simulate_intraday_recursive_forecast_from_context(
            bundle=bundle,
            context=curr_context,
            device=device,
            max_horizon=max_horizon,
        )
        if len(prev_preds) < 2 or len(curr_preds) < 1:
            continue
        actual_series = np.asarray(curr_context["actual_series"], dtype=np.float32)
        usable_horizon = min(max_horizon, len(curr_preds), len(prev_preds) - 1, len(actual_series) - 3)
        for horizon in range(1, usable_horizon + 1):
            rows.append(
                {
                    "horizon": horizon,
                    "prev_pred": float(prev_preds[horizon]),
                    "curr_pred": float(curr_preds[horizon - 1]),
                    "actual": float(actual_series[horizon + 2]),
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    weights: list[float] = []
    prev_weight = 0.35
    for horizon in range(1, max_horizon + 1):
        subset = df[df["horizon"] == horizon]
        if subset.empty:
            weights.append(prev_weight)
            continue
        prev_pred = subset["prev_pred"].to_numpy(dtype=np.float32)
        curr_pred = subset["curr_pred"].to_numpy(dtype=np.float32)
        actual_vals = subset["actual"].to_numpy(dtype=np.float32)
        search_start = 0.2 if horizon == 1 else prev_weight
        best_weight = float(prev_weight if horizon > 1 else 1.0)
        best_mae = float(np.mean(np.abs(prev_pred + best_weight * (curr_pred - prev_pred) - actual_vals)))
        for weight in np.arange(search_start, 1.001, 0.05):
            cand = np.maximum(0.0, prev_pred + float(weight) * (curr_pred - prev_pred))
            mae = float(np.mean(np.abs(cand - actual_vals)))
            if mae + 1e-9 < best_mae:
                best_mae = mae
                best_weight = float(weight)
        weights.append(best_weight)
        prev_weight = best_weight

    weights_arr = np.asarray(weights, dtype=np.float32)
    horizon_idx = df["horizon"].to_numpy(dtype=int) - 1
    calibrated = np.maximum(
        0.0,
        df["prev_pred"].to_numpy(dtype=np.float32)
        + weights_arr[horizon_idx] * (df["curr_pred"].to_numpy(dtype=np.float32) - df["prev_pred"].to_numpy(dtype=np.float32)),
    )
    baseline_mae = float(np.mean(np.abs(df["curr_pred"].to_numpy(dtype=np.float32) - df["actual"].to_numpy(dtype=np.float32))))
    calibrated_mae = float(np.mean(np.abs(calibrated - df["actual"].to_numpy(dtype=np.float32))))
    improvement_abs = baseline_mae - calibrated_mae
    if improvement_abs <= 1e-4:
        return None
    return {
        "enabled": True,
        "max_horizon": int(max_horizon),
        "weights": [float(w) for w in weights_arr.tolist()],
        "baseline_mae": float(baseline_mae),
        "calibrated_mae": float(calibrated_mae),
        "improvement_abs": float(improvement_abs),
        "improvement_pct": float(improvement_abs / max(baseline_mae, 1e-6)),
        "n_rows": int(len(df)),
    }


def _apply_intraday_continuity_calibration(
    pred_speed: float,
    previous_issue_speed: float | None,
    future_horizon: int,
    continuity_calibration: dict | None,
) -> float:
    if future_horizon <= 0 or previous_issue_speed is None:
        return float(pred_speed)
    if pd.isna(previous_issue_speed):
        return float(pred_speed)
    if not continuity_calibration or not bool(continuity_calibration.get("enabled", False)):
        return float(pred_speed)
    weights = continuity_calibration.get("weights") or []
    if not weights:
        return float(pred_speed)
    horizon_idx = min(int(future_horizon) - 1, len(weights) - 1)
    weight = float(weights[horizon_idx])
    return max(0.0, float(previous_issue_speed) + weight * (float(pred_speed) - float(previous_issue_speed)))


def _default_intraday_continuity_calibration(max_horizon: int = 12) -> dict:
    base_weights = [0.35, 0.50, 0.65, 0.78, 0.88, 0.94]
    weights = [float(base_weights[min(h - 1, len(base_weights) - 1)]) for h in range(1, int(max_horizon) + 1)]
    return {
        "enabled": True,
        "type": "default_v1",
        "max_horizon": int(max_horizon),
        "weights": weights,
        "baseline_mae": None,
        "calibrated_mae": None,
        "improvement_abs": None,
        "improvement_pct": None,
        "n_rows": 0,
    }


def split_intraday_contexts_for_holdout(
    contexts: list[dict],
    holdout_eval_split: float,
    holdout_min_contexts: int,
    min_training_contexts: int = 40,
) -> tuple[list[dict], list[dict]]:
    """
    Split real intraday issue contexts into an earlier training chunk and later holdout.

    Each context represents one operational issue time with its remaining hourly
    targets. Keeping the split chronological preserves the fair forecasting
    setup: challengers only train on earlier contexts and are compared on later
    issue times that the champion also sees.
    """
    n_contexts = int(len(contexts))
    if n_contexts <= int(min_training_contexts):
        raise ValueError("Not enough intraday contexts to reserve a later holdout.")

    holdout_n = max(int(round(n_contexts * float(holdout_eval_split))), int(holdout_min_contexts))
    holdout_n = min(holdout_n, n_contexts - int(min_training_contexts))
    if holdout_n < 1:
        raise ValueError("Not enough intraday contexts for challenger evaluation holdout.")

    split_idx = int(n_contexts - holdout_n)
    return contexts[:split_idx], contexts[split_idx:]


def build_intraday_holdout_context_split(
    db_path: Path,
    cfg: DatasetConfig,
    holdout_eval_split: float,
    holdout_min_contexts: int,
    max_horizon: int = INTRADAY_CALIBRATION_HORIZON,
) -> tuple[list[dict], list[dict]]:
    """
    Build the canonical chronological intraday train/holdout context split.

    The holdout unit is an issued intraday forecast context: one anchor/issue
    time with its remaining hourly targets. This mirrors the real current-day
    forecasting task more faithfully than a random split.
    """
    contexts = _build_intraday_anchor_contexts(
        db_path=db_path,
        cfg=cfg,
        max_horizon=max_horizon,
    )
    return split_intraday_contexts_for_holdout(
        contexts=contexts,
        holdout_eval_split=holdout_eval_split,
        holdout_min_contexts=holdout_min_contexts,
    )


def _intraday_metric_summary(pred: np.ndarray, actual: np.ndarray) -> dict[str, float | int | None]:
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    actual_arr = np.asarray(actual, dtype=float).reshape(-1)
    valid = (~np.isnan(pred_arr)) & (~np.isnan(actual_arr))
    count = int(np.sum(valid))
    if count <= 0:
        return {"count": 0, "mae": None, "rmse": None}
    errors = pred_arr[valid] - actual_arr[valid]
    return {
        "count": count,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def build_intraday_holdout_evaluation_frame(
    bundle: IntradayBundle,
    contexts: list[dict],
    device: torch.device,
    max_horizon: int = INTRADAY_CALIBRATION_HORIZON,
) -> pd.DataFrame:
    """
    Flatten intraday issue contexts into canonical hourly realised evaluation rows.

    One row corresponds to one target timestamp from one issued intraday
    context. The predictions are built recursively from the bundle exactly as
    the operational current-day model would issue them, and the realised
    observations remain the hourly targets after that issue time.
    """
    rows: list[dict] = []
    for context in contexts:
        anchor_time = pd.Timestamp(context["anchor_time"])
        if anchor_time.tzinfo is None:
            anchor_time = anchor_time.tz_localize("UTC")
        else:
            anchor_time = anchor_time.tz_convert("UTC")

        forecast_avg = np.asarray(context["forecast_avg_series"], dtype=np.float32)
        actual_series = np.asarray(context["actual_series"], dtype=np.float32)
        usable = min(int(max_horizon), len(forecast_avg) - 3, len(actual_series) - 3)
        if usable <= 0:
            continue

        pred = _simulate_intraday_recursive_forecast_from_context(
            bundle=bundle,
            context=context,
            device=device,
            max_horizon=usable,
        )
        usable = min(usable, int(len(pred)))
        if usable <= 0:
            continue

        forecast_target = forecast_avg[3 : 3 + usable]
        actual_target = actual_series[3 : 3 + usable]
        for step in range(usable):
            target_time = anchor_time + pd.Timedelta(hours=int(step + 1))
            rows.append(
                {
                    "anchor_time_utc": anchor_time,
                    "target_time_utc": target_time,
                    "horizon_hr": float(step + 1),
                    "prediction_value": float(pred[step]),
                    "harmonie_value": float(forecast_target[step]),
                    "actual_value": float(actual_target[step]),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "anchor_time_utc",
            "target_time_utc",
            "horizon_hr",
            "prediction_value",
            "harmonie_value",
            "actual_value",
        ],
    )


def align_intraday_holdout_frames(
    challenger_eval_frame: pd.DataFrame,
    champion_eval_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Align intraday challenger/champion holdout rows on the same issue/target pairs.
    """
    base = challenger_eval_frame[
        [
            "anchor_time_utc",
            "target_time_utc",
            "horizon_hr",
            "actual_value",
            "harmonie_value",
            "prediction_value",
        ]
    ].rename(columns={"prediction_value": "challenger_prediction_value"})

    if champion_eval_frame is None:
        aligned = base.copy()
        aligned["champion_prediction_value"] = np.nan
        return aligned

    champion = champion_eval_frame[
        [
            "anchor_time_utc",
            "target_time_utc",
            "prediction_value",
        ]
    ].rename(columns={"prediction_value": "champion_prediction_value"})

    return base.merge(
        champion,
        on=["anchor_time_utc", "target_time_utc"],
        how="inner",
    )


def summarize_intraday_champion_vs_challenger(
    *,
    challenger_eval_frame: pd.DataFrame,
    champion_eval_frame: pd.DataFrame | None,
    promotion_margin_pct: float,
    holdout_eval_split: float,
    holdout_eval_min_contexts: int,
    challenger_model_id: str,
    champion_model_id: str | None,
) -> dict:
    """
    Summarize the fair intraday promotion holdout for challenger vs champion.

    The holdout rows are built from later issue contexts only. Champion and
    challenger are aligned on the same anchor/target timestamps and scored
    against the same hourly realised observations, with Harmonie reported on
    that exact same holdout for baseline context.
    """
    aligned = align_intraday_holdout_frames(
        challenger_eval_frame=challenger_eval_frame,
        champion_eval_frame=champion_eval_frame,
    )
    if aligned.empty:
        raise ValueError("Intraday champion/challenger holdout comparison has no aligned realised rows.")

    actual = aligned["actual_value"].to_numpy(dtype=float)
    harmonie = aligned["harmonie_value"].to_numpy(dtype=float)
    challenger = aligned["challenger_prediction_value"].to_numpy(dtype=float)
    common_contexts = int(aligned["anchor_time_utc"].nunique())

    harmonie_metrics = _intraday_metric_summary(harmonie, actual)
    challenger_metrics = _intraday_metric_summary(challenger, actual)
    summary: dict[str, object] = {
        "comparison_unit": "intraday_issue_context_remaining_hourly_targets",
        "holdout_eval_split": float(holdout_eval_split),
        "holdout_eval_min_contexts": int(holdout_eval_min_contexts),
        "promotion_margin_pct": float(promotion_margin_pct),
        "intraday_model_id_challenger": challenger_model_id,
        "intraday_model_id_champion": champion_model_id or "none",
        "intraday_eval_contexts": common_contexts,
        "intraday_eval_rows": int(challenger_metrics["count"]),
        "intraday_mae_harmonie": harmonie_metrics["mae"],
        "intraday_rmse_harmonie": harmonie_metrics["rmse"],
        "intraday_mae_challenger": challenger_metrics["mae"],
        "intraday_rmse_challenger": challenger_metrics["rmse"],
    }

    if champion_eval_frame is None:
        summary.update(
            {
                "comparison_scope": "aligned_holdout_rows_no_existing_champion",
                "intraday_mae_champion": None,
                "intraday_rmse_champion": None,
                "intraday_mae_improvement_challenger_vs_champion": None,
                "intraday_rmse_improvement_challenger_vs_champion": None,
                "intraday_mae_improvement_challenger_vs_harmonie": None
                if challenger_metrics["mae"] is None or harmonie_metrics["mae"] is None
                else float(harmonie_metrics["mae"] - challenger_metrics["mae"]),
                "intraday_rmse_improvement_challenger_vs_harmonie": None
                if challenger_metrics["rmse"] is None or harmonie_metrics["rmse"] is None
                else float(harmonie_metrics["rmse"] - challenger_metrics["rmse"]),
                "promote_intraday": True,
                "reason": "no_existing_champion",
            }
        )
        return summary

    champion = aligned["champion_prediction_value"].to_numpy(dtype=float)
    champion_metrics = _intraday_metric_summary(champion, actual)
    champion_mae = float(champion_metrics["mae"])
    challenger_mae = float(challenger_metrics["mae"])
    champion_rmse = float(champion_metrics["rmse"])
    challenger_rmse = float(challenger_metrics["rmse"])
    promote_intraday = challenger_mae <= champion_mae * (1.0 - max(0.0, float(promotion_margin_pct)) / 100.0)

    summary.update(
        {
            "comparison_scope": "aligned_holdout_rows_common_to_champion_and_challenger",
            "intraday_mae_champion": champion_mae,
            "intraday_rmse_champion": champion_rmse,
            "intraday_mae_improvement_challenger_vs_champion": float(champion_mae - challenger_mae),
            "intraday_rmse_improvement_challenger_vs_champion": float(champion_rmse - challenger_rmse),
            "intraday_mae_improvement_challenger_vs_harmonie": None
            if harmonie_metrics["mae"] is None
            else float(harmonie_metrics["mae"] - challenger_mae),
            "intraday_rmse_improvement_challenger_vs_harmonie": None
            if harmonie_metrics["rmse"] is None
            else float(harmonie_metrics["rmse"] - challenger_rmse),
            "promote_intraday": bool(promote_intraday),
        }
    )
    return summary


def train_intraday_model(
    db_path: Path,
    cfg: DatasetConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    validation_split: float,
    hidden1: int = 128,
    hidden2: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    recency_power: float = 1.0,
    contexts: list[dict] | None = None,
) -> tuple[IntradayBundle, dict]:
    training_contexts = (
        _build_intraday_anchor_contexts(
            db_path=db_path,
            cfg=cfg,
            max_horizon=INTRADAY_CALIBRATION_HORIZON,
        )
        if contexts is None
        else contexts
    )
    X_all, y_all = build_intraday_training_xy(db_path, cfg, contexts=training_contexts)
    n = len(X_all)
    split_idx = int(n * (1.0 - validation_split))
    split_idx = max(20, min(split_idx, n - 20))
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    params = IntradayTrainParams(
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_split=float(validation_split),
        hidden1=int(hidden1),
        hidden2=int(hidden2),
        dropout=float(dropout),
        learning_rate=float(learning_rate),
        recency_power=float(recency_power),
    )
    bundle, best_val = _fit_intraday_from_arrays(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        params=params,
        device=device,
    )
    bundle.rollout_calibration = _fit_intraday_rollout_calibration(
        bundle=bundle,
        contexts=training_contexts,
        sample_split_idx=split_idx,
        device=device,
        max_horizon=INTRADAY_CALIBRATION_HORIZON,
    )
    bundle.continuity_calibration = _fit_intraday_continuity_calibration(
        bundle=bundle,
        contexts=training_contexts,
        sample_split_idx=split_idx,
        device=device,
        max_horizon=INTRADAY_CALIBRATION_HORIZON,
    )
    if bundle.continuity_calibration is None:
        bundle.continuity_calibration = _default_intraday_continuity_calibration()
    stats = {
        "n_samples": int(n),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "best_val_loss": float(best_val),
        "feature_cols": FEATURE_COLS,
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "recency_power": float(recency_power),
        "rollout_calibration": bundle.rollout_calibration,
        "continuity_calibration": bundle.continuity_calibration,
    }
    return bundle, stats


def save_intraday_model(path: Path, bundle: IntradayBundle, extra: dict | None = None) -> None:
    payload = {
        "model_state_dict": bundle.model.state_dict(),
        "model_class": "IntradayResidualMLP",
        "n_features": int(len(FEATURE_COLS)),
        "feature_cols": FEATURE_COLS,
        "x_mean": bundle.x_mean.astype(np.float32),
        "x_std": bundle.x_std.astype(np.float32),
        "y_mean": float(bundle.y_mean),
        "y_std": float(bundle.y_std),
        "hidden1": int(getattr(bundle.model.net[0], "out_features", 128)),
        "hidden2": int(getattr(bundle.model.net[3], "out_features", 64)),
        "dropout": float(getattr(bundle.model.net[2], "p", 0.1)),
        "rollout_calibration": bundle.rollout_calibration,
        "continuity_calibration": bundle.continuity_calibration,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_intraday_model(path: Path, device: torch.device) -> tuple[IntradayBundle, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = IntradayResidualMLP(
        n_features=int(ckpt["n_features"]),
        hidden1=int(ckpt.get("hidden1", 128)),
        hidden2=int(ckpt.get("hidden2", 64)),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    bundle = IntradayBundle(
        model=model,
        x_mean=np.asarray(ckpt["x_mean"], dtype=np.float32),
        x_std=np.asarray(ckpt["x_std"], dtype=np.float32),
        y_mean=float(ckpt["y_mean"]),
        y_std=float(ckpt["y_std"]),
        rollout_calibration=ckpt.get("rollout_calibration"),
        continuity_calibration=ckpt.get("continuity_calibration") or _default_intraday_continuity_calibration(),
    )
    return bundle, ckpt


def _get_series_value(series: pd.Series, ts: pd.Timestamp, fallback: float) -> float:
    val = series.get(ts, np.nan)
    if pd.isna(val):
        return float(fallback)
    return float(val)


def predict_intraday_day_speed(
    bundle: IntradayBundle,
    forecast_frame_local: pd.DataFrame,
    actual_hourly_local: pd.Series,
    previous_issue_forecast: pd.Series | None,
    day_start_local: pd.Timestamp,
    day_end_local: pd.Timestamp,
    now_hour_local: pd.Timestamp,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    hours = pd.date_range(start=day_start_local, end=day_end_local, freq="1h", tz=day_start_local.tzinfo)
    series = actual_hourly_local.copy()
    pred_full = np.full(len(hours), np.nan, dtype=np.float32)

    for h_idx, h in enumerate(hours):
        t = h - pd.Timedelta(hours=1)
        t1 = h

        fc_t = forecast_frame_local.reindex([t]).iloc[0] if t in forecast_frame_local.index else None
        fc_t1 = forecast_frame_local.reindex([t1]).iloc[0] if t1 in forecast_frame_local.index else None
        if fc_t is None or fc_t1 is None or pd.isna(fc_t1["forecast_avg"]):
            continue

        fc_avg_t = float(fc_t["forecast_avg"]) if not pd.isna(fc_t["forecast_avg"]) else float(fc_t1["forecast_avg"])
        fc_max_t = float(fc_t["forecast_max"]) if not pd.isna(fc_t["forecast_max"]) else fc_avg_t
        fc_avg_t1 = float(fc_t1["forecast_avg"])
        fc_max_t1 = float(fc_t1["forecast_max"]) if not pd.isna(fc_t1["forecast_max"]) else fc_avg_t1
        dir_t = float(fc_t["forecast_dir"]) if not pd.isna(fc_t["forecast_dir"]) else 0.0
        dir_t1 = float(fc_t1["forecast_dir"]) if not pd.isna(fc_t1["forecast_dir"]) else dir_t

        a_t = _get_series_value(series, t, fc_avg_t)
        a_t_1 = _get_series_value(series, t - pd.Timedelta(hours=1), fc_avg_t)
        a_t_2 = _get_series_value(series, t - pd.Timedelta(hours=2), fc_avg_t)
        fc_t_1_row = forecast_frame_local.reindex([t - pd.Timedelta(hours=1)]).iloc[0] if (t - pd.Timedelta(hours=1)) in forecast_frame_local.index else None
        fc_t_2_row = forecast_frame_local.reindex([t - pd.Timedelta(hours=2)]).iloc[0] if (t - pd.Timedelta(hours=2)) in forecast_frame_local.index else None
        fc_avg_t_1 = float(fc_t_1_row["forecast_avg"]) if fc_t_1_row is not None and not pd.isna(fc_t_1_row["forecast_avg"]) else fc_avg_t
        fc_avg_t_2 = float(fc_t_2_row["forecast_avg"]) if fc_t_2_row is not None and not pd.isna(fc_t_2_row["forecast_avg"]) else fc_avg_t
        res_t = a_t - fc_avg_t
        res_t_1 = a_t_1 - fc_avg_t_1
        res_t_2 = a_t_2 - fc_avg_t_2

        hour_ang = 2.0 * np.pi * (float(t.hour) / 24.0)
        month_sin = float(fc_t1["month_sin"]) if "month_sin" in fc_t1.index and not pd.isna(fc_t1["month_sin"]) else 0.0
        month_cos = float(fc_t1["month_cos"]) if "month_cos" in fc_t1.index and not pd.isna(fc_t1["month_cos"]) else 1.0

        x = np.array(
            [
                fc_avg_t,
                fc_max_t,
                fc_avg_t1,
                fc_max_t1,
                np.sin(np.deg2rad(dir_t)),
                np.cos(np.deg2rad(dir_t)),
                np.sin(np.deg2rad(dir_t1)),
                np.cos(np.deg2rad(dir_t1)),
                a_t,
                a_t_1,
                a_t_2,
                res_t,
                res_t_1,
                res_t_2,
                np.sin(hour_ang),
                np.cos(hour_ang),
                month_sin,
                month_cos,
            ],
            dtype=np.float32,
        )
        pred_speed = _predict_intraday_step(bundle, x, forecast_next=fc_avg_t1, device=device)
        if h > now_hour_local:
            future_horizon = int((h - now_hour_local).total_seconds() / 3600.0)
            pred_speed = _apply_intraday_rollout_calibration(
                pred_speed=pred_speed,
                forecast_speed=fc_avg_t1,
                future_horizon=future_horizon,
                rollout_calibration=bundle.rollout_calibration,
            )
            previous_issue_speed = None
            if previous_issue_forecast is not None:
                previous_issue_speed = previous_issue_forecast.get(h, np.nan)
            pred_speed = _apply_intraday_continuity_calibration(
                pred_speed=pred_speed,
                previous_issue_speed=previous_issue_speed,
                future_horizon=future_horizon,
                continuity_calibration=bundle.continuity_calibration,
            )
        pred_full[h_idx] = float(pred_speed)

        # Recursive update only for unknown/future points.
        if h > now_hour_local or pd.isna(series.get(h, np.nan)):
            series.loc[h] = float(pred_speed)

    pred_rem = np.where(hours > now_hour_local, pred_full, np.nan).astype(np.float32)
    return pred_full.astype(np.float32), pred_rem
