/* Uses the next-day forecast table schema. Static images remain on any failure. */
(function () {
    "use strict";
    async function initialize() {
        const container = document.getElementById("day-after-tomorrow-interactive");
        const fallback = document.getElementById("day-after-tomorrow-fallback");
        if (!container || !fallback || !window.Plotly) return;
        try {
            const response = await fetch(container.dataset.jsonUrl, {cache: "no-store"});
            if (!response.ok) return;
            const payload = await response.json();
            const rows = payload.rows;
            if (!Array.isArray(rows) || !rows.length) return;
            const x = rows.map(row => row.target_time_local.slice(0, 16));
            const trace = (field, name, color) => ({
                x, y: rows.map(row => row[field]), name, type: "scatter", mode: "lines+markers",
                connectgaps: false, line: {color, width: 2}, marker: {size: 4},
                customdata: rows.map(row => [row.forecast_wind_dir_deg, row.lstm_pred_wind_dir_deg,
                    row.forecast_temperature_c, row.weather_description]),
                hovertemplate: "%{x}<br>%{y:.2f} knots<br>HARMONIE direction: %{customdata[0]:.0f}°<br>Local direction: %{customdata[1]:.0f}°<br>Temperature: %{customdata[2]:.1f}°C<br>%{customdata[3]}<extra>%{fullData.name}</extra>"
            });
            const data = [trace("forecast_wind_speed", "HARMONIE", "#777777"),
                          trace("lstm_pred_wind_speed", "Super local D+2", "#ff7f0e")];
            if (payload.ecmwf && payload.ecmwf.length) {
                data.push({x: payload.ecmwf.map(row => row.time_local.slice(0, 16)),
                    y: payload.ecmwf.map(row => row.wind_speed_knots),
                    name: "ECMWF", mode: "lines", type: "scatter", connectgaps: false,
                    line: {color: "#007f7f", width: 1.9}});
            }
            // Keep the full local 08–22 window, including masked forecast hours.
            const start = x[0], end = x[x.length - 1];
            container.hidden = false;
            await Plotly.newPlot(container, data, {
                title: {text: "Day after tomorrow · experimental", font: {size: 18}},
                height: window.innerWidth <= 768 ? 360 : 450,
                margin: {l: 50, r: 15, t: 45, b: 75},
                xaxis: {title: "Local time", type: "date", tickformat: "%H:%M", range: [start, end]},
                yaxis: {title: "Wind speed (knots)", rangemode: "tozero"},
                legend: {orientation: "h", y: -0.2}, hovermode: "closest"
            }, {responsive: true, displaylogo: false, scrollZoom: false});
            fallback.hidden = true;
            const toggle = document.createElement("button");
            toggle.className = "button";
            toggle.textContent = "Show full forecast plot";
            toggle.addEventListener("click", function () {
                fallback.hidden = !fallback.hidden;
                container.hidden = !fallback.hidden;
                toggle.textContent = fallback.hidden ? "Show full forecast plot" : "Show interactive forecast";
            });
            container.parentNode.insertBefore(toggle, fallback);
        } catch (_) {
            container.hidden = true;
            fallback.hidden = false;
        }
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
    else initialize();
})();
