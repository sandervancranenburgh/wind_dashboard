# Production database backup and restore

The production SQLite database contains observations, forecast vintages,
issued predictions, user accounts, and rider submissions. A database restore
is a last-resort recovery operation because it discards every row written after
the selected snapshot.

## Paths

- Live database: `~/Documents/wind_fetcher_data/data/wind_data_all_sites.db`
- Daily backups: `~/Documents/wind_fetcher_data/backups/wind_data_YYYY-MM-DD.db.gz`
- Backup log: `~/Documents/wind_fetcher_data/backups/backup.log`

The repository's `data` directory is a symlink to the live data directory in
production. Never replace the symlink during verification.

## Daily snapshot

Run `backup_database.sh`. It uses SQLite's online backup API, validates the
snapshot with `PRAGMA quick_check`, compresses it, refuses to overwrite an
existing same-day archive, and applies bounded retention.

Do not copy the live database file directly while WAL mode is active. A plain
file copy can omit committed rows that are still in the WAL.

## Isolated verification drill

First choose a work filesystem with enough free space. The verifier requires a
conservative twelve times the compressed archive size and never extracts into
the production data or backup directory.

```bash
python scripts/verify_database_backup.py \
  ~/Documents/wind_fetcher_data/backups/wind_data_YYYY-MM-DD.db.gz \
  --work-parent /tmp \
  --report /tmp/wind-database-backup-report.json
```

The verifier:

1. rejects the production database as an input archive;
2. extracts into a new temporary directory;
3. calculates the archive SHA-256 checksum;
4. opens the restored database read-only and immutable;
5. runs `PRAGMA integrity_check`;
6. checks required tables, schema, JSON payloads, row counts, site counts, and
   timestamp bounds;
7. deletes only its own temporary restored database after reporting.

Retain the report and checksum with the change record. Copy at least one
verified archive to storage outside the production filesystem before any
database-changing deployment.

## Pre-deployment gate

Before a deployment that can change schema or write a new class of rows:

1. record the production Git commit and active crontab;
2. create a fresh online backup;
3. run the isolated verification drill;
4. copy the verified archive to the secondary backup location;
5. record database table/site counts and latest source timestamps;
6. stop if any check fails.

## Emergency restore

Do not restore merely to roll back application code. Prefer reverting code or
disabling the affected collector. If a complete restore is unavoidable:

1. announce the recovery window and identify the exact data-loss interval;
2. disable the dashboard cron jobs, KNMI listener/fallback, ECMWF timer, and
   rider portal writes;
3. acquire the updater lock and confirm no process has the database open for
   writing;
4. preserve the current live database with SQLite's online backup API under a
   unique incident filename;
5. verify the chosen recovery archive with
   `scripts/verify_database_backup.py`;
6. extract it to a new database filename in the production data directory;
7. run `PRAGMA integrity_check` on that new file;
8. compare schema, user/session counts, per-site source counts, and timestamps;
9. only after explicit approval, atomically repoint the production data path or
   rename the validated file into place;
10. start read-only application checks, then writers one at a time;
11. monitor two collection cycles before closing the recovery window.

Never delete the pre-restore database during the incident. Keep it until the
restored service and all expected writers have been verified.

## Coverage exclusions

The SQLite backup does not include uploaded FIT/GPX/KML/ZIP files, generated
activity-analysis artifacts, model artifacts, secrets, or the separate ECMWF
archive. Those require their own backup and retention policies.
