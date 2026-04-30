from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backfill_current_model_history import _load_model, backfill_next_day_prediction_log
from data_pipeline import DatasetConfig
from db_store import (
    init_db,
    materialize_prediction_log_evaluation,
    summarize_prediction_log_vs_harmonie,
    summarize_prediction_log_vs_harmonie_by_horizon,
    summarize_prediction_log_vs_harmonie_by_issued_day,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final deployed-champion vs Harmonie performance report from canonical SQLite data.",
    )
    parser.add_argument("--db", default="data/wind_data_all_sites.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument(
        "--metadata",
        default="next_day_wind_model/artifacts/metadata_update.json",
        help="Metadata JSON from latest model update.",
    )
    parser.add_argument(
        "--out-dir",
        default="next_day_wind_model/artifacts/champion_vs_harmonie_report",
        help="Directory for report outputs.",
    )
    parser.add_argument("--local-timezone", default="Europe/Amsterdam", help="Timezone for descriptive labels.")
    parser.add_argument(
        "--skip-next-day-backfill",
        action="store_true",
        help="Do not backfill canonical next-day champion rows into prediction_log before reporting.",
    )
    return parser.parse_args()


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _champion_info(metadata: dict) -> dict:
    refresh = metadata.get("champion_refresh_summary") or {}
    post_run = refresh.get("post_run") or {}
    next_day_speed = post_run.get("next_day_speed") or {}
    next_day_direction = post_run.get("next_day_direction") or {}
    intraday_speed = post_run.get("intraday_speed") or {}
    return {
        "next_day_speed": {
            "path": next_day_speed.get("path") or "next_day_wind_model/artifacts/next_day_lstm_speed_residual.pt",
            "model_id": refresh.get("active_next_day_speed_model_id"),
            "model_version": next_day_speed.get("trained_at_utc"),
        },
        "next_day_direction": {
            "path": next_day_direction.get("path") or "next_day_wind_model/artifacts/next_day_lstm_direction_residual.pt",
            "model_id": refresh.get("active_next_day_direction_model_id"),
            "model_version": next_day_direction.get("trained_at_utc"),
        },
        "intraday_speed": {
            "path": intraday_speed.get("path") or "next_day_wind_model/artifacts/intraday_speed_residual.pt",
            "model_id": refresh.get("active_intraday_model_id"),
            "model_version": intraday_speed.get("trained_at_utc"),
        },
    }


def _window_and_target_hours(metadata: dict) -> tuple[int, int]:
    return int(metadata.get("window_hours", 72)), int(metadata.get("target_hours", 24))


def _format_metric(value: object, digits: int = 3, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if pct:
        return f"{value_f:.{digits}f}%"
    return f"{value_f:.{digits}f}"


def _write_csv(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _rows_to_markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for _label, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append("" if value is None else str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def _overall_summary_lines(title: str, summary: dict) -> list[str]:
    return [
        f"### {title}",
        f"- Count: {summary.get('count', 0)}",
        f"- Model MAE: {_format_metric(summary.get('model_mae'))}",
        f"- Harmonie MAE: {_format_metric(summary.get('harmonie_mae'))}",
        f"- MAE improvement: {_format_metric(summary.get('mae_improvement'))} "
        f"({_format_metric(summary.get('mae_improvement_pct'), pct=True)})",
        f"- Model RMSE: {_format_metric(summary.get('model_rmse'))}",
        f"- Harmonie RMSE: {_format_metric(summary.get('harmonie_rmse'))}",
        f"- RMSE improvement: {_format_metric(summary.get('rmse_improvement'))} "
        f"({_format_metric(summary.get('rmse_improvement_pct'), pct=True)})",
        f"- Model bias: {_format_metric(summary.get('model_bias'))}",
        f"- Harmonie bias: {_format_metric(summary.get('harmonie_bias'))}",
        f"- Model win rate vs Harmonie: {_format_metric(summary.get('model_win_rate_vs_harmonie'), pct=True)}",
    ]


def _interpretation_block(intraday_summary: dict, next_day_summary: dict) -> list[str]:
    lines = [
        "## Interpretation",
        "This is a production comparison for the currently deployed champion models under the improved canonical evaluation pipeline.",
        "It is not a full methodological re-baseline from newly reset champions.",
    ]
    for label, summary in [("Intraday", intraday_summary), ("Next-day", next_day_summary)]:
        count = int(summary.get("count") or 0)
        improvement = summary.get("mae_improvement")
        if count <= 0 or improvement is None:
            lines.append(f"- {label}: no realised canonical rows were available for a fair overall conclusion.")
            continue
        direction = "beats" if float(improvement) > 0 else ("trails" if float(improvement) < 0 else "matches")
        lines.append(
            f"- {label}: the deployed champion {direction} Harmonie overall on MAE by "
            f"{_format_metric(abs(float(improvement)))}."
        )
    return lines


def _build_report_payload(
    *,
    db_path: Path,
    metadata_path: Path,
    champions: dict,
    next_day_backfill: dict,
    intraday_overall: dict,
    intraday_by_horizon: list[dict],
    intraday_by_issued_day: list[dict],
    next_day_overall: dict,
    next_day_by_horizon: list[dict],
    next_day_by_issued_day: list[dict],
) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "report_scope": "currently deployed champion models vs Harmonie",
        "framing_note": (
            "Production comparison of the active champions under the improved canonical evaluation pipeline; "
            "not a full methodological re-baseline from reset/retrained champions."
        ),
        "champions": champions,
        "next_day_backfill": next_day_backfill,
        "intraday": {
            "overall": intraday_overall,
            "by_horizon": intraday_by_horizon,
            "by_issued_day": intraday_by_issued_day,
        },
        "next_day": {
            "overall": next_day_overall,
            "by_horizon": next_day_by_horizon,
            "by_issued_day": next_day_by_issued_day,
        },
        "limitations": [
            "Next-day results are derived from canonical historical champion backfill rows written into prediction_log for the currently deployed next-day speed champion.",
            "Intraday results come from realised issued prediction_log rows for the currently deployed intraday champion.",
            "Wind-speed performance is the focus of this report; the next-day direction champion is identified for completeness but not scored against Harmonie here.",
        ],
    }


def _write_markdown_report(
    path: Path,
    *,
    generated_at_utc: str,
    champions: dict,
    intraday_overall: dict,
    intraday_by_horizon: list[dict],
    intraday_by_issued_day: list[dict],
    next_day_overall: dict,
    next_day_by_horizon: list[dict],
    next_day_by_issued_day: list[dict],
    next_day_backfill: dict,
) -> None:
    lines = [
        "# Deployed Champion vs Harmonie Report",
        "",
        f"Generated at UTC: `{generated_at_utc}`",
        "",
        "## Champion Scope",
        f"- Next-day speed champion ID: `{champions['next_day_speed'].get('model_id')}`",
        f"- Next-day speed champion version: `{champions['next_day_speed'].get('model_version')}`",
        f"- Next-day direction champion ID: `{champions['next_day_direction'].get('model_id')}`",
        f"- Intraday champion ID: `{champions['intraday_speed'].get('model_id')}`",
        f"- Intraday champion version: `{champions['intraday_speed'].get('model_version')}`",
        "",
        "## Backfill Note",
        "- The next-day section below uses canonical realised `prediction_log` rows.",
        "- Because the deployed next-day champion had not yet accumulated realised durable rows in the live log, "
        "this report first materialised historical frozen next-day champion rows into `prediction_log` "
        "under a dedicated `next_day_backfill` model type using the current deployed champion.",
        f"- Next-day backfill rows inserted this run: `{next_day_backfill['inserted_rows']}`",
        f"- Next-day evaluation rows materialised this run: `{next_day_backfill['materialized_rows']}`",
        "",
        *_overall_summary_lines("Intraday / Current-day Champion vs Harmonie", intraday_overall),
        "",
        _rows_to_markdown_table(
            intraday_by_horizon,
            [
                ("Horizon", "horizon_hr"),
                ("Count", "count"),
                ("Model MAE", "model_mae"),
                ("Harmonie MAE", "harmonie_mae"),
                ("MAE Gain", "mae_improvement"),
                ("Win Rate", "model_win_rate_vs_harmonie"),
            ],
        ),
        "",
        "Full intraday issued-day history is exported to `intraday_by_issued_day.csv`.",
        "",
        *_overall_summary_lines("Next-day Champion vs Harmonie", next_day_overall),
        "",
        _rows_to_markdown_table(
            next_day_by_horizon,
            [
                ("Horizon", "horizon_hr"),
                ("Count", "count"),
                ("Model MAE", "model_mae"),
                ("Harmonie MAE", "harmonie_mae"),
                ("MAE Gain", "mae_improvement"),
                ("Win Rate", "model_win_rate_vs_harmonie"),
            ],
        ),
        "",
        "Full next-day issued-day history is exported to `next_day_by_issued_day.csv`.",
        "",
        *_interpretation_block(intraday_overall, next_day_overall),
        "",
        "## Limitations",
        "- This report is intentionally about the currently deployed champions, not a full reset/re-baseline of model history.",
        "- Intraday currently spans only the realised canonical rows present in the durable log for the active champion.",
        "- Next-day evaluation here is wind-speed only.",
    ]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    metadata_path = Path(args.metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(metadata_path)
    champions = _champion_info(metadata)
    window_hours, target_hours = _window_and_target_hours(metadata)

    next_day_speed_path = Path(champions["next_day_speed"]["path"])
    intraday_speed_path = Path(champions["intraday_speed"]["path"])
    next_day_version = champions["next_day_speed"]["model_version"]
    intraday_version = champions["intraday_speed"]["model_version"]

    next_day_backfill = {"inserted_rows": 0, "materialized_rows": 0, "skipped": bool(args.skip_next_day_backfill)}
    if not args.skip_next_day_backfill:
        cfg = DatasetConfig(
            site=args.site,
            model=args.model,
            window_hours=window_hours,
            target_hours=target_hours,
        )
        next_day_model, next_day_ckpt = _load_model(next_day_speed_path)
        x_mean = np.load(REPO_ROOT / "next_day_wind_model/artifacts/x_mean_speed.npy")
        x_std = np.load(REPO_ROOT / "next_day_wind_model/artifacts/x_std_speed.npy")
        y_mean = float(np.load(REPO_ROOT / "next_day_wind_model/artifacts/y_mean_speed.npy")[0])
        y_std = float(np.load(REPO_ROOT / "next_day_wind_model/artifacts/y_std_speed.npy")[0])
        inserted_rows, materialized_rows = backfill_next_day_prediction_log(
            db_path=db_path,
            cfg=cfg,
            model=next_day_model,
            ckpt=next_day_ckpt,
            x_mean_saved=x_mean,
            x_std_saved=x_std,
            y_mean_saved=y_mean,
            y_std_saved=y_std,
            local_tz=args.local_timezone,
            model_type="next_day_backfill",
            model_artifact=next_day_speed_path.name,
            model_version=next_day_version,
            model_name=type(next_day_model).__name__,
        )
        next_day_backfill = {
            "inserted_rows": int(inserted_rows),
            "materialized_rows": int(materialized_rows),
            "skipped": False,
        }

    conn = sqlite3.connect(str(db_path))
    try:
        init_db(conn)
        materialize_prediction_log_evaluation(conn, site=args.site, prediction_kind="wind_speed")

        intraday_overall = summarize_prediction_log_vs_harmonie(
            conn,
            site=args.site,
            model_type="intraday",
            prediction_kind="wind_speed",
            model_artifact=intraday_speed_path.name,
            model_version=intraday_version,
        )
        intraday_by_horizon = summarize_prediction_log_vs_harmonie_by_horizon(
            conn,
            site=args.site,
            model_type="intraday",
            prediction_kind="wind_speed",
            model_artifact=intraday_speed_path.name,
            model_version=intraday_version,
        )
        intraday_by_issued_day = summarize_prediction_log_vs_harmonie_by_issued_day(
            conn,
            site=args.site,
            model_type="intraday",
            prediction_kind="wind_speed",
            model_artifact=intraday_speed_path.name,
            model_version=intraday_version,
        )

        next_day_overall = summarize_prediction_log_vs_harmonie(
            conn,
            site=args.site,
            model_type="next_day_backfill",
            prediction_kind="wind_speed",
            model_artifact=next_day_speed_path.name,
            model_version=next_day_version,
            frozen_next_day=True,
        )
        next_day_by_horizon = summarize_prediction_log_vs_harmonie_by_horizon(
            conn,
            site=args.site,
            model_type="next_day_backfill",
            prediction_kind="wind_speed",
            model_artifact=next_day_speed_path.name,
            model_version=next_day_version,
            frozen_next_day=True,
        )
        next_day_by_issued_day = summarize_prediction_log_vs_harmonie_by_issued_day(
            conn,
            site=args.site,
            model_type="next_day_backfill",
            prediction_kind="wind_speed",
            model_artifact=next_day_speed_path.name,
            model_version=next_day_version,
            frozen_next_day=True,
        )
    finally:
        conn.close()

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    report_payload = _build_report_payload(
        db_path=db_path,
        metadata_path=metadata_path,
        champions=champions,
        next_day_backfill=next_day_backfill,
        intraday_overall=intraday_overall,
        intraday_by_horizon=intraday_by_horizon,
        intraday_by_issued_day=intraday_by_issued_day,
        next_day_overall=next_day_overall,
        next_day_by_horizon=next_day_by_horizon,
        next_day_by_issued_day=next_day_by_issued_day,
    )

    json_path = out_dir / "champion_vs_harmonie_report.json"
    md_path = out_dir / "champion_vs_harmonie_report.md"
    intraday_horizon_csv = out_dir / "intraday_by_horizon.csv"
    intraday_day_csv = out_dir / "intraday_by_issued_day.csv"
    next_day_horizon_csv = out_dir / "next_day_by_horizon.csv"
    next_day_day_csv = out_dir / "next_day_by_issued_day.csv"

    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    _write_csv(intraday_by_horizon, intraday_horizon_csv)
    _write_csv(intraday_by_issued_day, intraday_day_csv)
    _write_csv(next_day_by_horizon, next_day_horizon_csv)
    _write_csv(next_day_by_issued_day, next_day_day_csv)
    _write_markdown_report(
        md_path,
        generated_at_utc=generated_at_utc,
        champions=champions,
        intraday_overall=intraday_overall,
        intraday_by_horizon=intraday_by_horizon,
        intraday_by_issued_day=intraday_by_issued_day,
        next_day_overall=next_day_overall,
        next_day_by_horizon=next_day_by_horizon,
        next_day_by_issued_day=next_day_by_issued_day,
        next_day_backfill=next_day_backfill,
    )

    print(f"Report JSON: {json_path}")
    print(f"Report Markdown: {md_path}")
    print(f"Intraday horizon CSV: {intraday_horizon_csv}")
    print(f"Intraday issued-day CSV: {intraday_day_csv}")
    print(f"Next-day horizon CSV: {next_day_horizon_csv}")
    print(f"Next-day issued-day CSV: {next_day_day_csv}")
    print(
        "Top line | "
        f"intraday count={intraday_overall.get('count')} "
        f"mae_model={_format_metric(intraday_overall.get('model_mae'))} "
        f"mae_harmonie={_format_metric(intraday_overall.get('harmonie_mae'))} | "
        f"next_day count={next_day_overall.get('count')} "
        f"mae_model={_format_metric(next_day_overall.get('model_mae'))} "
        f"mae_harmonie={_format_metric(next_day_overall.get('harmonie_mae'))}"
    )


if __name__ == "__main__":
    main()
