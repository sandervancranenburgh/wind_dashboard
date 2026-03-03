from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import DatasetConfig, build_training_arrays


class NextDayLSTM(nn.Module):
    def __init__(self, n_features: int, target_hours: int, output_activation: str = "linear") -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, target_hours)
        self.output_activation = str(output_activation).strip().lower()
        if self.output_activation not in {"linear", "softplus"}:
            raise ValueError("output_activation must be 'linear' or 'softplus'.")
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        if self.output_activation == "softplus":
            x = self.softplus(x)
        return x


def evaluate_denormalized(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_actual_val_raw: np.ndarray,
    y_forecast_val_raw: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> dict:
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_val).float().to(device)
        pred_scaled = model(X_t).cpu().numpy()

    pred_target = pred_scaled * y_std + y_mean
    pred = pred_target + y_forecast_val_raw
    truth = y_actual_val_raw

    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    return {"mae": mae, "rmse": rmse}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM model for next-day wind prediction.")
    parser.add_argument("--db", default="data/wind_data.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument("--window-hours", type=int, default=72, help="Input history length for X.")
    parser.add_argument("--target-hours", type=int, default=24, help="Prediction horizon in hours for Y.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--out-dir",
        default="next_day_wind_model/artifacts",
        help="Directory where model and metadata are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=args.window_hours,
        target_hours=args.target_hours,
    )

    arrays = build_training_arrays(db_path=db_path, cfg=cfg, target_mode="residual")

    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    X_val = arrays["X_val"]
    y_val = arrays["y_val"]

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu")
    model = NextDayLSTM(n_features=X_train.shape[2], target_hours=y_train.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0
    early_stopping_patience = 8

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)

                batch_size = X_batch.size(0)
                val_loss_sum += float(loss.item()) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(val_count, 1)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)

    y_mean = float(arrays["y_mean"][0])
    y_std = float(arrays["y_std"][0])
    metrics = evaluate_denormalized(
        model,
        X_val,
        y_val,
        arrays["y_actual_val_raw"],
        arrays["y_forecast_val_raw"],
        y_mean=y_mean,
        y_std=y_std,
        device=device,
    )

    model_path = out_dir / "next_day_lstm_speed_residual.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "NextDayLSTM",
        "n_features": int(X_train.shape[2]),
        "target_hours": int(y_train.shape[1]),
    }
    torch.save(checkpoint, model_path)

    np.save(out_dir / "x_mean_speed.npy", arrays["x_mean"])
    np.save(out_dir / "x_std_speed.npy", arrays["x_std"])
    np.save(out_dir / "y_mean_speed.npy", arrays["y_mean"])
    np.save(out_dir / "y_std_speed.npy", arrays["y_std"])

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "site": args.site,
        "forecast_model": args.model,
        "window_hours": args.window_hours,
        "target_hours": args.target_hours,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "feature_cols": arrays["feature_cols"],
        "target_col": arrays["target_col"],
        "target_mode": "residual",
        "best_val_loss": float(best_val_loss),
        "val_mae_denormalized": metrics["mae"],
        "val_rmse_denormalized": metrics["rmse"],
        "device": str(device),
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Validation MAE (original units): {metrics['mae']:.3f}")
    print(f"Validation RMSE (original units): {metrics['rmse']:.3f}")


if __name__ == "__main__":
    main()
