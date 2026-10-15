"""Shared compass plot for the three forecast evaluation horizons."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from next_day_wind_model.ecmwf_dashboard import ECMWF_FORECAST_COLOR
from next_day_wind_model.ecmwf_evaluation import SECTORS


def save_direction_plot(direction_csv: Path, plot_png: Path, *, model_column: str,
                        model_label: str, title: str) -> None:
    if not direction_csv.exists():
        return
    frame = pd.read_csv(direction_csv)
    required = {"sector", "forecast_mae", model_column}
    if not required.issubset(frame):
        return
    frame = frame.set_index("sector").reindex(SECTORS)
    if frame[["forecast_mae", model_column]].dropna().shape[0] < 3:
        plot_png.unlink(missing_ok=True)
        return
    angles = np.arange(8) * np.pi / 4
    closed_angles = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(7.4, 7.0), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    curves = [("forecast_mae", "n_points", "Harmonie", "#777777"),
              (model_column, "n_points", model_label, "#f28e2b"),
              ("ecmwf_mae", "ecmwf_n_points", "ECMWF", ECMWF_FORECAST_COLOR)]
    for column, weight_column, label, color in curves:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float, copy=True)
        weights = (pd.to_numeric(frame[weight_column], errors="coerce").to_numpy(float)
                   if weight_column in frame else np.ones(8))
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            continue
        values[~valid] = np.nan
        mean = np.average(values[valid], weights=weights[valid])
        # All eight compass positions are retained. NaNs break the line rather
        # than connecting across an unobserved direction sector.
        ax.plot(closed_angles, np.r_[values, values[0]], color=color,
                linewidth=2.0, marker="o", markersize=4,
                label=f"{label} ({mean:.2f} kts)")
        for angle, value in zip(angles, values):
            if value > 3.5:
                ax.scatter([angle], [3.5], marker="^", color=color, s=35, clip_on=False)
                ax.annotate(f"{value:.2f}", (angle, 3.5), xytext=(0, 9),
                            textcoords="offset points", color=color, fontsize=9,
                            ha="center", clip_on=False)
    ax.set_xticks(angles)
    ax.set_xticklabels(SECTORS, fontsize=11)
    ax.set_ylim(0, 3.5)
    ax.set_rlabel_position(225)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(color="#d7d7d7", linewidth=.8)
    ax.spines["polar"].set_color("#cfcfcf")
    ax.set_title(title, pad=22, fontsize=14, fontweight="bold")
    ax.text(.5, -.08, "Lower radial value means lower mean absolute error (kts).",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#444444")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=1,
              frameon=False, fontsize=10)
    fig.subplots_adjust(left=.08, right=.92, top=.90, bottom=.22)
    plot_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_png, dpi=150, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)
