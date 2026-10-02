"""D+2 website fragments; no database or model work."""
import html
import pandas as pd
from next_day_wind_model.day_after_tomorrow import PREFIX, status_text


def forecast_card(state, assets, version):
    if state.get("status") == "disabled":
        return ""
    description = html.escape(status_text(state))
    plot = ""
    if f"{PREFIX}_predictions.png" in assets and state.get("status") in ["available", "stale"]:
        plot = f'''<div class="interactive-block" data-interactive-wind-block="true"
             data-plot-id="day-after-tomorrow-interactive-plot" data-controls-id="day-after-tomorrow-interactive-controls"
             data-details-id="day-after-tomorrow-interactive-details" data-fallback-id="day-after-tomorrow-fallback"
             data-json-url="{PREFIX}_interactive_data.json?v={version}">
          <div class="interactive-controls" id="day-after-tomorrow-interactive-controls"></div>
          <div class="interactive-plot" id="day-after-tomorrow-interactive-plot" aria-label="Interactive day-after-tomorrow wind plot"></div>
          <div class="interactive-point-details" id="day-after-tomorrow-interactive-details">Click a plotted point to view exact values.</div>
        </div>
        <picture id="day-after-tomorrow-fallback">
          <source media="(max-width:768px)" srcset="{PREFIX}_predictions_mobile.png?v={version}">
          <img src="{PREFIX}_predictions.png?v={version}" alt="Experimental day-after-tomorrow forecast">
        </picture><p><a href="{PREFIX}_predictions.csv">Download forecast CSV</a></p>'''
    return f'''<section class="card"><h2>Day-after-tomorrow prediction · experimental</h2>
        <p class="desc" id="day-after-tomorrow-status">{description}</p>{plot}</section>'''


def evaluation_content(state, assets, version, kind=None):
    if state.get("status") == "disabled":
        return ""
    gate = state.get("gate", {})
    caption = "No completed operational D+2 gate evaluation is available yet."
    gate_caption = caption
    if gate:
        dates = f"Evaluation targets: {gate['evaluation_start']}–{gate['evaluation_end']}. "
        caption = ("The spider compares speed MAE by HARMONIE forecast direction; lower values are better. "
            "Overlapping forecasts are averaged per target hour before scoring. " + dates)
        gate_caption = ("Aligned holdout wind speeds above; absolute errors below. Grey: HARMONIE; blue: challenger; "
            "orange: prior champion; magenta: observations. " + dates +
            "Promotion requires a 1% MAE improvement over the dedicated champion on identical available targets. ")
        decision = gate.get("decisions", {}).get("speed", {})
        if decision:
            candidate = decision["candidate"]["mae"]
            baseline = decision["baseline"]["mae"]
            gate_caption += (f"Challenger speed MAE: {candidate:.2f} knots; HARMONIE: {baseline:.2f} knots. "
                        + ("The challenger is worse than HARMONIE on this holdout." if candidate > baseline
                           else "The challenger improves on HARMONIE on this holdout."))
            if decision.get("reason") == "no_existing_champion":
                gate_caption += " This is the initial champion; no previous D+2 champion existed for comparison."
        gate_caption += (f" Champion: {gate.get('speed_model_id_champion', 'unknown')}; "
            f"challenger: {gate.get('speed_model_id_challenger', 'unknown')}.")
    sector_path = assets.get(f"{PREFIX}_speed_by_direction.csv")
    if sector_path:
        sectors = pd.read_csv(sector_path).dropna(subset=["champion_mae_gain_vs_forecast"])
        if not sectors.empty and sectors.n_points.sum() > 0:
            weights = sectors.n_points / sectors.n_points.sum()
            model_mae = float((weights * sectors.champion_mae).sum())
            baseline_mae = float((weights * sectors.forecast_mae).sum())
            caption += f"Target-hour average MAE: D+2 {model_mae:.2f} knots; HARMONIE {baseline_mae:.2f} knots. "
        positive = sectors[sectors.champion_mae_gain_vs_forecast > 0].sort_values("champion_mae_gain_vs_forecast", ascending=False)
        worse = sectors[sectors.champion_mae_gain_vs_forecast < 0]
        if not positive.empty:
            caption += " Largest gains: " + ", ".join(positive.sector.head(3)) + "."
        if not worse.empty:
            caption += " HARMONIE performs better for " + ", ".join(worse.sector) + "."
    if gate:
        caption += f" Evaluated active model: {gate['spider_model_id']}."
    if gate.get("calibration_policy") == "none":
        note = " Speed calibration disabled. These historical results informed the choice; confirmation on 20 new dates is pending."
        caption += note
        gate_caption += note
    pieces = []
    for name, title in [(f"{PREFIX}_direction_spider", "Day-after-tomorrow performance by wind direction · experimental"),
                         (f"{PREFIX}_model_gate_eval_history", "Day-after-tomorrow model-gate evaluation history")]:
        is_spider = "direction" in name
        if kind == "spider" and not is_spider or kind == "gate" and is_spider:
            continue
        image = f'<img src="{name}.png?v={version}" alt="{title}">' if f"{name}.png" in assets else ""
        text = caption if is_spider else gate_caption
        if is_spider:
            row = (f'<div class="direction-card">'
                f'<div class="direction-copy"><h3>{title}</h3><p class="desc">{html.escape(text)}</p></div>'
                f'<div class="direction-plot">{image}</div></div>')
            pieces.append(row if kind == "spider" else f'<section class="card performance-section">{row}</section>')
        else:
            pieces.append(f'<section class="card"><h2>{title}</h2><p class="desc">{html.escape(text)}</p>{image}</section>')
    return ''.join(pieces)
