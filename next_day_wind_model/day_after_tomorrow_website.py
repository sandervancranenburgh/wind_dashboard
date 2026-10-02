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
        plot = f'''<div id="day-after-tomorrow-interactive" class="interactive-plot" hidden
                  data-json-url="{PREFIX}_interactive_data.json?v={version}" aria-label="Interactive experimental day-after-tomorrow forecast"></div>
        <picture id="day-after-tomorrow-fallback">
          <source media="(max-width:768px)" srcset="{PREFIX}_predictions_mobile.png?v={version}">
          <img src="{PREFIX}_predictions.png?v={version}" alt="Experimental day-after-tomorrow forecast">
        </picture><p><a href="{PREFIX}_predictions.csv">Download forecast CSV</a></p>'''
    return f'''<section class="card"><h2>Day-after-tomorrow prediction · experimental</h2>
        <p class="desc" id="day-after-tomorrow-status">{description}</p>{plot}</section>'''


def evaluation_content(state, assets, version):
    if state.get("status") == "disabled":
        return ""
    gate = state.get("gate", {})
    caption = "No completed operational D+2 gate evaluation is available yet."
    if gate:
        caption = (f"Holdout targets: {gate['evaluation_start']}–{gate['evaluation_end']}. "
            f"Evaluated active speed model: {gate['spider_model_id']}. "
            "The spider compares speed MAE by HARMONIE forecast direction; overlapping issues are averaged per target hour. "
            "The gate compares challenger and prior champion on identical available targets. "
            "Promotion requires a 1% MAE improvement; missing targets are excluded.")
        decision = gate.get("decisions", {}).get("speed", {})
        if decision:
            candidate = decision["candidate"]["mae"]
            baseline = decision["baseline"]["mae"]
            caption += (f" Latest gate challenger speed MAE: {candidate:.2f} knots; HARMONIE: {baseline:.2f} knots. "
                        + ("The challenger is worse than HARMONIE on this holdout." if candidate > baseline
                           else "The challenger improves on HARMONIE on this holdout."))
            if decision.get("reason") == "no_existing_champion":
                caption += " This is the initial champion; no previous D+2 champion existed for comparison."
    sector_path = assets.get(f"{PREFIX}_speed_by_direction.csv")
    if sector_path:
        sectors = pd.read_csv(sector_path).dropna(subset=["champion_mae_gain_vs_forecast"])
        positive = sectors[sectors.champion_mae_gain_vs_forecast > 0].sort_values("champion_mae_gain_vs_forecast", ascending=False)
        worse = sectors[sectors.champion_mae_gain_vs_forecast < 0]
        if not positive.empty:
            caption += " Largest gains: " + ", ".join(positive.sector.head(3)) + "."
        if not worse.empty:
            caption += " HARMONIE performs better for " + ", ".join(worse.sector) + "."
    pieces = []
    for name, title in [(f"{PREFIX}_direction_spider", "Day-after-tomorrow performance by wind direction · experimental"),
                         (f"{PREFIX}_model_gate_eval_history", "Day-after-tomorrow model-gate evaluation history")]:
        image = f'<img src="{name}.png?v={version}" alt="{title}">' if f"{name}.png" in assets else ""
        text = caption if "direction" in name else ("Aligned holdout wind speeds above; absolute errors below. Grey: HARMONIE; blue: challenger; orange: prior champion; magenta: observations. " + caption)
        if "direction" in name:
            pieces.append(f'<section class="card performance-section"><div class="direction-card">'
                f'<div class="direction-copy"><h3>{title}</h3><p class="desc">{html.escape(text)}</p></div>'
                f'<div class="direction-plot">{image}</div></div></section>')
        else:
            pieces.append(f'<section class="card"><h2>{title}</h2><p class="desc">{html.escape(text)}</p>{image}</section>')
    return ''.join(pieces)
