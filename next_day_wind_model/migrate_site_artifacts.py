"""Stage, validate, and atomically install a site-isolated artifact copy.

The default is a non-mutating dry run. Source files are never changed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.intraday_model import load_intraday_model
from next_day_wind_model.site_paths import build_artifact_manifest, validate_artifact_manifest, write_artifact_manifest
from next_day_wind_model.update_model_and_predict import _load_model


MODEL_FILES = (
    "next_day_lstm_speed_residual.pt",
    "next_day_lstm_direction_residual.pt",
    "intraday_speed_residual.pt",
)
SCALER_FILES = (
    "x_mean_speed.npy",
    "x_std_speed.npy",
    "y_mean_speed.npy",
    "y_std_speed.npy",
    "x_mean_direction.npy",
    "x_std_direction.npy",
    "y_mean_direction.npy",
    "y_std_direction.npy",
)
CORE_FILES = MODEL_FILES + SCALER_FILES


def _assert_disjoint(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination must be disjoint directories")


def _validate_required(directory: Path) -> None:
    missing = [name for name in CORE_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts in {directory}: {', '.join(missing)}")


def _model_prediction(model: torch.nn.Module, checkpoint: dict) -> np.ndarray:
    sequence_length = int(checkpoint.get("history_hours", 72)) + int(checkpoint["target_hours"])
    if str(checkpoint.get("model_class", "NextDayLSTM")) == "NextDayLSTM":
        sequence_length = int(checkpoint.get("history_hours", 72))
    generator = torch.Generator().manual_seed(3407)
    values = torch.randn((2, sequence_length, int(checkpoint["n_features"])), generator=generator)
    with torch.no_grad():
        return model(values).cpu().numpy()


def validate_identical_models(source: Path, destination: Path, *, site_id: str, forecast_model: str) -> None:
    device = torch.device("cpu")
    for filename in MODEL_FILES[:2]:
        source_model, source_checkpoint = _load_model(source / filename, device)
        copied_model, copied_checkpoint = _load_model(
            destination / filename,
            device,
            expected_site_id=site_id,
            expected_forecast_model=forecast_model,
            allow_legacy_identity=True,
        )
        if not np.array_equal(
            _model_prediction(source_model, source_checkpoint),
            _model_prediction(copied_model, copied_checkpoint),
        ):
            raise ValueError(f"prediction mismatch after copying {filename}")
    source_intraday, _ = load_intraday_model(source / MODEL_FILES[2], device)
    copied_intraday, _ = load_intraday_model(
        destination / MODEL_FILES[2],
        device,
        expected_site_id=site_id,
        expected_forecast_model=forecast_model,
        allow_legacy_identity=True,
    )
    generator = torch.Generator().manual_seed(3407)
    features = torch.randn((4, len(source_intraday.x_mean)), generator=generator)
    with torch.no_grad():
        source_values = source_intraday.model(features).cpu().numpy()
        copied_values = copied_intraday.model(features).cpu().numpy()
    if not np.array_equal(source_values, copied_values):
        raise ValueError("prediction mismatch after copying intraday model")
    for filename in SCALER_FILES:
        if not np.array_equal(np.load(source / filename), np.load(destination / filename)):
            raise ValueError(f"scaler mismatch after copying {filename}")


def migrate(source: Path, destination: Path, *, site_id: str, forecast_model: str) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    _assert_disjoint(source, destination)
    _validate_required(source)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    try:
        shutil.copytree(source, staging, symlinks=True)
        manifest = build_artifact_manifest(
            artifact_dir=staging,
            site_id=site_id,
            forecast_model=forecast_model,
            status="champion",
            production_eligible=True,
            files=[staging / name for name in CORE_FILES],
            extra={"migration_source": str(source), "legacy_checkpoint_identity_allowed": True},
        )
        write_artifact_manifest(staging, manifest)
        validate_artifact_manifest(
            staging,
            expected_site_id=site_id,
            expected_forecast_model=forecast_model,
        )
        validate_identical_models(source, staging, site_id=site_id, forecast_model=forecast_model)
        staging.replace(destination)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--model", default="HARMONIE")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _assert_disjoint(args.source, args.destination)
    _validate_required(args.source)
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "source": str(args.source.resolve()),
        "destination": str(args.destination.resolve()),
        "site_id": args.site,
        "forecast_model": args.model,
        "required_files": list(CORE_FILES),
    }
    print(json.dumps(plan, indent=2))
    if args.execute:
        manifest = migrate(
            args.source,
            args.destination,
            site_id=args.site,
            forecast_model=args.model,
        )
        print(json.dumps({"validated": True, "manifest_files": len(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
