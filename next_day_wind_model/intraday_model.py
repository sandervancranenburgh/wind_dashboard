from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import DatasetConfig, _build_training_frame


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


def build_intraday_training_xy(db_path: Path, cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    frame = _build_training_frame(db_path, cfg).copy()
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError("No rows available for intraday model training.")

    # Keep only rows where actuals + required forecasts exist.
    need_cols = ["forecast_avg", "forecast_max", "forecast_dir", "actual_avg", "month_sin", "month_cos"]
    frame = frame.dropna(subset=need_cols)
    if len(frame) < 50:
        raise ValueError("Not enough rows to train intraday model.")

    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    fc_avg = frame["forecast_avg"].to_numpy(dtype=np.float32)
    fc_max = frame["forecast_max"].to_numpy(dtype=np.float32)
    fc_dir = frame["forecast_dir"].to_numpy(dtype=np.float32)
    actual = frame["actual_avg"].to_numpy(dtype=np.float32)
    month_sin = frame["month_sin"].to_numpy(dtype=np.float32)
    month_cos = frame["month_cos"].to_numpy(dtype=np.float32)
    idx = frame.index

    # Feature time is t=i, target is residual at t+1.
    for i in range(2, len(frame) - 1):
        if np.isnan([fc_avg[i], fc_avg[i + 1], fc_max[i], fc_max[i + 1], fc_dir[i], fc_dir[i + 1], actual[i], actual[i - 1], actual[i - 2]]).any():
            continue
        res_t = actual[i] - fc_avg[i]
        res_t_1 = actual[i - 1] - fc_avg[i - 1]
        res_t_2 = actual[i - 2] - fc_avg[i - 2]
        hour = idx[i].hour
        hour_ang = 2.0 * np.pi * (float(hour) / 24.0)
        x = np.array(
            [
                fc_avg[i],
                fc_max[i],
                fc_avg[i + 1],
                fc_max[i + 1],
                np.sin(np.deg2rad(fc_dir[i])),
                np.cos(np.deg2rad(fc_dir[i])),
                np.sin(np.deg2rad(fc_dir[i + 1])),
                np.cos(np.deg2rad(fc_dir[i + 1])),
                actual[i],
                actual[i - 1],
                actual[i - 2],
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
        y = float(actual[i + 1] - fc_avg[i + 1])  # next-hour residual
        if np.isnan(x).any() or np.isnan(y):
            continue
        X_list.append(x)
        y_list.append(y)

    if not X_list:
        raise ValueError("No intraday training samples after filtering.")
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def _build_intraday_training_frame(db_path: Path, cfg: DatasetConfig) -> pd.DataFrame:
    frame = _build_training_frame(db_path, cfg).copy()
    frame = frame.sort_index()
    need_cols = ["forecast_avg", "forecast_max", "forecast_dir", "actual_avg", "month_sin", "month_cos"]
    frame = frame.dropna(subset=need_cols)
    if len(frame) < 50:
        raise ValueError("Not enough rows to train intraday model.")
    return frame


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


def _fit_intraday_rollout_calibration(
    bundle: IntradayBundle,
    frame: pd.DataFrame,
    sample_split_idx: int,
    device: torch.device,
    max_horizon: int = 12,
) -> dict | None:
    if sample_split_idx < 20:
        return None

    fc_avg = frame["forecast_avg"].to_numpy(dtype=np.float32)
    actual = frame["actual_avg"].to_numpy(dtype=np.float32)
    first_anchor = int(sample_split_idx + 2)
    last_anchor = int(len(frame) - max_horizon - 1)
    if first_anchor >= last_anchor:
        return None

    rows: list[dict] = []
    for anchor in range(first_anchor, last_anchor):
        recursive_actual = actual.copy()
        recursive_actual[anchor + 1 :] = np.nan
        for horizon in range(1, max_horizon + 1):
            i = anchor + horizon - 1
            x = _build_intraday_feature_row(frame, recursive_actual, i)
            pred_speed = _predict_intraday_step(bundle, x, forecast_next=float(fc_avg[i + 1]), device=device)
            recursive_actual[i + 1] = pred_speed
            rows.append(
                {
                    "horizon": horizon,
                    "forecast": float(fc_avg[i + 1]),
                    "pred_recursive": float(pred_speed),
                    "actual": float(actual[i + 1]),
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


def _simulate_intraday_recursive_forecast(
    bundle: IntradayBundle,
    frame: pd.DataFrame,
    anchor: int,
    device: torch.device,
    max_horizon: int,
) -> np.ndarray:
    if anchor < 2 or anchor >= len(frame) - 1 or max_horizon <= 0:
        return np.array([], dtype=np.float32)

    fc_avg = frame["forecast_avg"].to_numpy(dtype=np.float32)
    actual = frame["actual_avg"].to_numpy(dtype=np.float32)
    recursive_actual = actual.copy()
    recursive_actual[anchor + 1 :] = np.nan

    preds: list[float] = []
    for horizon in range(1, int(max_horizon) + 1):
        i = anchor + horizon - 1
        if i + 1 >= len(frame):
            break
        x = _build_intraday_feature_row(frame, recursive_actual, i)
        pred_speed = _predict_intraday_step(bundle, x, forecast_next=float(fc_avg[i + 1]), device=device)
        pred_speed = _apply_intraday_rollout_calibration(
            pred_speed=pred_speed,
            forecast_speed=float(fc_avg[i + 1]),
            future_horizon=horizon,
            rollout_calibration=bundle.rollout_calibration,
        )
        recursive_actual[i + 1] = pred_speed
        preds.append(float(pred_speed))
    return np.asarray(preds, dtype=np.float32)


def _fit_intraday_continuity_calibration(
    bundle: IntradayBundle,
    frame: pd.DataFrame,
    sample_split_idx: int,
    device: torch.device,
    max_horizon: int = 12,
) -> dict | None:
    if sample_split_idx < 24:
        return None

    actual = frame["actual_avg"].to_numpy(dtype=np.float32)
    first_anchor = int(max(sample_split_idx + 3, 3))
    last_anchor = int(len(frame) - max_horizon - 1)
    if first_anchor >= last_anchor:
        return None

    rows: list[dict] = []
    for anchor in range(first_anchor, last_anchor):
        prev_preds = _simulate_intraday_recursive_forecast(
            bundle=bundle,
            frame=frame,
            anchor=anchor - 1,
            device=device,
            max_horizon=max_horizon + 1,
        )
        curr_preds = _simulate_intraday_recursive_forecast(
            bundle=bundle,
            frame=frame,
            anchor=anchor,
            device=device,
            max_horizon=max_horizon,
        )
        if len(prev_preds) < 2 or len(curr_preds) < 1:
            continue
        usable_horizon = min(max_horizon, len(curr_preds), len(prev_preds) - 1)
        for horizon in range(1, usable_horizon + 1):
            target_idx = anchor + horizon
            if target_idx >= len(actual):
                break
            rows.append(
                {
                    "horizon": horizon,
                    "prev_pred": float(prev_preds[horizon]),
                    "curr_pred": float(curr_preds[horizon - 1]),
                    "actual": float(actual[target_idx]),
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
) -> tuple[IntradayBundle, dict]:
    training_frame = _build_intraday_training_frame(db_path, cfg)
    X_all, y_all = build_intraday_training_xy(db_path, cfg)
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
        frame=training_frame,
        sample_split_idx=split_idx,
        device=device,
    )
    bundle.continuity_calibration = _fit_intraday_continuity_calibration(
        bundle=bundle,
        frame=training_frame,
        sample_split_idx=split_idx,
        device=device,
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
