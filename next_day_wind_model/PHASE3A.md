# Phase 3A development workflow

Phase 3A is development-only. Nothing in this change installs or modifies a
cron job, service, production wrapper, live database, `docs/` output, or registry
enablement flag. Do not run these commands from the production checkout.

## Site-isolated paths

`site_model_orchestrator.py` resolves enabled model sites to:

- `next_day_wind_model/artifacts/<site_id>/`
- `docs/` for the default public site
- `docs/spots/<site_id>/` for a future published non-default site
- no web directory for an unpublished site

It is a dry run unless `--execute` is explicitly supplied. Oostvoorne is
disabled and is rejected by the orchestrator.

## Valkenburg migration rehearsal

The migration utility is also a dry run unless `--execute` is supplied. The
source and destination must be disjoint, the destination must not exist, and
the source is never changed. An executing migration copies into a sibling
staging directory, records checksums, loads all three models on CPU, compares
deterministic model predictions and scaler arrays, then atomically renames the
staging directory.

```bash
python next_day_wind_model/migrate_site_artifacts.py \
  --source /path/to/copied-or-read-only-legacy-artifacts \
  --destination /tmp/phase3a-valkenburg-rehearsal \
  --site valkenburgsemeer --execute
```

## Offline Oostvoorne experiment

Restore the verified database backup to temporary storage and make the restored
file read-only. The experiment opens it using SQLite read-only/immutable mode,
uses ordinary HARMONIE only, and verifies the database SHA-256 before and after.
Use reduced CPU and I/O priority, and run only while production collectors are
idle.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 ionice -c3 \
  python next_day_wind_model/oostvoorne_experiment.py \
  --db /tmp/wind_phase3a_20260930/wind_data_restored.db \
  --output-root next_day_wind_model/artifacts
```

All experiment files are written below
`next_day_wind_model/artifacts/oostvoorne/experiments/<run_id>/` and are marked
`experimental` and `production_eligible=false`. The runner has no publishing,
prediction-log, champion-selection, Git, or registry-write code path.

Phase 3B deployment remains separately gated and is not part of these tools.

## Phase 3B Valkenburg cutover

The production wrapper now selects
`next_day_wind_model/artifacts/valkenburgsemeer/` explicitly for both generated
and model artifacts. The legacy directory remains in place during monitoring.
Set `WIND_USE_LEGACY_MODEL_ARTIFACTS=1` on a controlled wrapper invocation to
return immediately to the pre-Phase 3B artifact path.

This cutover does not enable Oostvoorne and does not change its publication
state or eligibility date.
