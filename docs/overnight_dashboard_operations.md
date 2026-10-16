# Overnight dashboard rollover

Windsurfice collection and the normal dashboard pipeline run every six minutes
from 07:00 through 22:54 in Europe/Amsterdam. The proposed midnight cron entry
runs `scripts/run_wind_dashboard_pipeline.sh --refresh-only` under the same
updater lock. It skips source collection and training, and uses stored forecast
vintages and the existing production models. The operational gate rejects a
cached current-day or next-day forecast with the wrong calendar date, forcing
prediction and plot generation at rollover even when source data are unchanged.
This is a prediction refresh; `--plots-only` would retain yesterday's CSV dates.

Operational next-day inference uses one explicit current issue timestamp for
both speed and direction. Historical calls without a reference timestamp retain
their observation-based date. Calendar-day offsets preserve local midnight
across daylight-saving changes. No observation or fetch timestamp is fabricated:
metadata still reports when the underlying data were actually obtained.

The plot metadata announces the midnight refresh after the final evening plot,
then 07:00 after midnight. Prediction/HARMONIE fetch estimates during the
collector pause point to 07:00. ECMWF runs on its separate overnight timer and
keeps its own six-hour-cycle arrival estimate.

## Deployment

Development changes do not install the cron entry. Before an authorized
production deployment, follow `database_backup_and_restore.md`, verify fresh
online backups and preserve changed source, cron and relevant model/runtime
artifacts. Review the patch and active crontab, retaining its environment
variables and all unrelated jobs. Add only the midnight job from
`ops/cron/wind-fetcher2.crontab`; do not replace the active crontab with the
template. Verify the host timezone is Europe/Amsterdam. After integration,
verify a normal daytime refresh and the midnight publication, checking the
current-day and tomorrow CSV dates, both plot titles, metadata estimates and
GitHub Pages publication. Roll back the source change and remove just the added
midnight cron line if the rollover fails; retain collected data and backups.
