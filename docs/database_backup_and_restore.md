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

Retain the report and checksum with the change record. Verification is required
for deployment/rollback backups as well as disaster-recovery copies.

## Deployment and rollback backups

A fresh, verified local SQLite backup is sufficient for ordinary deployments
and database-writing changes, including routine schema migrations. It provides
a recovery point for application errors and incorrect database writes while
the production filesystem and the backup remain accessible. An external backup
is not a mandatory prerequisite for every deployment.

Create the snapshot immediately before the relevant change. An earlier daily
archive does not replace a fresh pre-change snapshot. If today's daily archive
already exists, retain it and create the additional online snapshot under a
unique timestamped filename; do not overwrite an existing archive or bypass
verification. Use SQLite's online backup API, validate the snapshot, compress
it, and run the isolated verification drill against that archive.

An external backup is required when the proposed change itself threatens the
local backup's survival or accessibility, such as filesystem replacement, VPS
rebuilding, or operations affecting the backup directory. Record the specific
risk and required backup destination in the change record. Create and verify
a copy outside the affected storage or VPS before proceeding; stop if that
required copy cannot be created or verified. A routine migration or new class
of database rows does not by itself create this requirement.

## Pre-deployment gate

Before an authorized production change that can modify the database, including
schema changes or writing a new class of rows:

1. record the production Git commit and active crontab;
2. create a fresh local online backup;
3. run the isolated verification drill and retain the report and checksum;
4. assess whether the change threatens the local backup; only if it does,
   document the risk and create and verify the required external copy;
5. record database table/site counts and latest source timestamps;
6. stop if any check fails.

## Disaster-recovery backups

Verified off-VPS backups are strongly recommended independently of deployment
timing. Backups on the production filesystem protect against application
errors, but share production's storage failure domain: filesystem failure or
deletion affecting both can remove the database and its recovery copies.
Another filesystem on the same VPS may protect against failure of one
filesystem, but does not protect against losing the VPS.

Keep disaster-recovery copies on independent storage outside the VPS, with
their verification reports and checksums, appropriate retention and access
controls, and periodic restore verification. These archives contain private
user and rider data. This recommendation does not block ordinary deployments
with a fresh, verified local backup; the change-specific elevated-risk rule
above determines when an external copy is required before deployment.

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
