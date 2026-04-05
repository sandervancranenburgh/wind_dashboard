Legacy and archived material lives here.

Purpose:
- keep the active workflow easier to understand
- move superseded or generated clutter out of the main source/artifact paths
- preserve older material for retrieval when it was not safe to delete outright

Conventions:
- `old/code/` holds legacy source files that are no longer part of the active canonical workflow
- `old/reports/` holds older report/dashboard exports kept for reference
- `old/artifacts/`, `old/data/`, and `old/plots/` hold bulky generated archives

Notes:
- files under `old/artifacts/`, `old/data/`, and `old/plots/` are ignored in git so they do not reintroduce working-tree clutter
- active production artifacts, current dashboard outputs, current reports, and canonical pipeline code remain in their original locations
