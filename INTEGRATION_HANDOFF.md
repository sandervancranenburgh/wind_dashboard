# Wingfoil Pipeline Portal Integration Handoff

This note is for the Codex agent that will integrate the wingfoil activity
analysis pipeline into the wind dashboard rider portal.

## Current Pipeline Repo

Source repo:

```text
/Users/sandervancranenburgh/Documents/Repos_and_data/Temp/garmin_export
```

Website repos mentioned by the owner:

```text
/home/sandervancranenburgh/Documents/repos/wind_fetcher2
/home/sandervancranenburgh/Documents/repos/wind_fetcher2_dev
```

Do integration work in `wind_fetcher2_dev` first. Do not edit production
`wind_fetcher2` until the dev integration has been tested.

## Pipeline Files To Copy

Copy these into the website dev repo:

```text
wingfoil_analysis/
requirements.txt
requirements-dev.txt
pyproject.toml
tests/test_pipeline.py
```

Optional compatibility wrapper:

```text
scripts/analyze_wingfoil_gpx.py
```

Do not copy these into the website repo as application code:

```text
outputs/
data/
```

Sample files from `data/` can be copied into a test fixture folder only if
needed for local smoke tests.

## Runtime Dependencies

The pipeline currently needs:

```text
pandas>=2.2
fitparse>=1.2
```

The generated interactive map uses Leaflet and external satellite tiles from
the browser. For a production portal, either allow those external browser
resources in the site's CSP/network policy or port the map rendering into the
portal frontend. The static `map.svg` is generated as a no-JS fallback.

## Public Python API

Use the high-level API from Flask code:

```python
from wingfoil_analysis import analyze_session_file

payload = analyze_session_file(
    input_file=uploaded_file_path,
    output_dir=session_output_dir,
    wind_context=wind_context,
    raise_on_error=False,
)
```

Alternative if the portal stores wind context as JSON:

```python
payload = analyze_session_file(
    input_file=uploaded_file_path,
    output_dir=session_output_dir,
    wind_json=wind_json_path,
    raise_on_error=False,
)
```

The function returns a dict. On success:

```text
status = "ok"
analysis_version
input_filename
input_type
summary_json
map_html
map_svg
runs_csv
artifacts
plots
stats
warnings
artifact_paths
```

On failure with `raise_on_error=False`:

```text
status = "error"
analysis_version
input_filename
input_type
error
warnings
```

## Supported Input Files

The pipeline accepts:

```text
.fit
.gpx
.kml
```

Input type is detected from the file extension. Unsupported extensions return a
structured error through `analyze_session_file(..., raise_on_error=False)`.

## Generated Artifacts

Each analysis output directory contains:

```text
summary.json
runs.csv
map.svg
map.html
run_distance_distribution.svg
run_speed_distribution.svg
run_wind_angle_distribution.svg
run_speed.svg
```

Use `summary.json` as the main machine-readable analysis payload. Store
artifact-relative paths in the database, not absolute server filesystem paths.

Generated public files include the original input filename and type, but should
not expose absolute uploaded-file paths.

## Flask Integration Shape

The existing Flask app is expected around:

```text
next_day_wind_model/web_dashboard/app.py
next_day_wind_model/web_dashboard/templates/
```

Rider submission table:

```text
surf_experiences
```

Recommended integration:

1. Add an authenticated upload endpoint for a rider activity file.
2. Accept only `.fit`, `.gpx`, and `.kml`.
3. Save the upload outside the public static folder using a generated safe
   filename.
4. Create a per-experience output directory for generated analysis artifacts.
5. Build optional wind context from the session/site wind data already stored
   in the portal.
6. Call `analyze_session_file(...)`.
7. Persist the returned payload or selected fields in SQLite.
8. Show `map.html`, `summary.json` stats, `runs.csv`, and SVG charts on the
   rider detail page.
9. Respect the existing portal privacy/share-token model. Never expose files
   unless the current user or share token may view that surf experience.

Example endpoint skeleton:

```python
@app.post("/experiences/<int:experience_id>/upload-track")
def upload_track(experience_id):
    experience = load_experience_for_current_user_or_404(experience_id)
    uploaded = request.files.get("track_file")
    if uploaded is None or uploaded.filename == "":
        flash("Choose a FIT, GPX, or KML activity file.")
        return redirect(url_for("experience_detail", experience_id=experience_id))

    input_path = save_activity_upload_safely(uploaded, experience_id)
    output_dir = ANALYSIS_ROOT / str(experience_id)
    wind_context = build_wind_context_for_experience(experience)

    payload = analyze_session_file(
        input_file=input_path,
        output_dir=output_dir,
        wind_context=wind_context,
        raise_on_error=False,
    )

    store_activity_analysis(experience_id, payload)
    return redirect(url_for("experience_detail", experience_id=experience_id))
```

## Wind Context

Wind is optional. If no wind is supplied, the pipeline still runs and reports
that wind context is unavailable.

Wind context fields:

```text
spot_name
wind_direction_deg
wind_direction_cardinal
wind_speed_kts
wind_source
```

When wind direction is available, `runs.csv` includes:

```text
mean_bearing_deg
angle_to_wind_deg
wind_angle_class
```

Important: the course rose is GPS travel direction. It is not wind direction.
Wind direction is external context and should remain visually distinct.

## Suggested Database Fields

Add either a small analysis table or JSON columns linked to `surf_experiences`.
Recommended table shape:

```sql
CREATE TABLE surf_experience_activity_analysis (
    id INTEGER PRIMARY KEY,
    surf_experience_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    analysis_version TEXT,
    input_filename TEXT,
    input_type TEXT,
    artifact_base_path TEXT,
    summary_json_path TEXT,
    map_html_path TEXT,
    map_svg_path TEXT,
    runs_csv_path TEXT,
    stats_json TEXT,
    warnings_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (surf_experience_id) REFERENCES surf_experiences(id)
);
```

Keep absolute filesystem paths server-side only. Templates should receive URLs
created by Flask routes that enforce authorization.

## Security Requirements

1. Never trust uploaded filenames.
2. Use `secure_filename` plus a generated unique prefix or UUID.
3. Enforce allowed extensions before saving.
4. Enforce a maximum upload size.
5. Store raw uploads outside public static paths.
6. Serve generated artifacts through authenticated Flask routes or signed/share
   token routes.
7. Avoid exposing local absolute paths in HTML, JSON, logs shown to users, or
   database fields that may become public.
8. Treat `map.html` as generated HTML. Only embed or serve it for authorized
   viewers.

## Validation Commands

From the website dev repo after copying the package:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_pipeline.py
python -m wingfoil_analysis analyse --input path/to/session.fit --out-dir /tmp/session_analysis
```

If using the owner's pyenv environment:

```bash
PYENV_VERSION=gen_env python -m pytest tests/test_pipeline.py
PYENV_VERSION=gen_env python -m wingfoil_analysis analyse --input path/to/session.fit --out-dir /tmp/session_analysis
```

Verify that `/tmp/session_analysis` contains all expected artifacts and that
`summary.json` has `analysis_status: "ok"`.

## Prompt For The Website Codex Agent

Use this prompt in the website dev repo:

```text
Integrate the copied `wingfoil_analysis` package into this Flask rider portal.

Do not edit the production repo. Work only in `wind_fetcher2_dev`.

Requirements:
1. Add a secure authenticated upload flow for `.fit`, `.gpx`, and `.kml`
   activity files linked to an existing `surf_experiences` row.
2. Save raw uploads outside public static folders with generated safe names.
3. Create a per-experience analysis output directory.
4. Build optional wind context from the portal's existing site/session wind
   data when available.
5. Call `wingfoil_analysis.analyze_session_file(..., raise_on_error=False)`.
6. Persist the result status, analysis version, input filename/type, artifact
   paths, stats, warnings, and error text in SQLite.
7. Add authorized routes for generated artifacts instead of exposing absolute
   filesystem paths.
8. Update the rider detail template to show the generated map, summary stats,
   run table, and SVG charts.
9. Preserve existing privacy/share-token behavior.
10. Add tests for upload validation, successful analysis with a sample file,
    error handling, and unauthorized artifact access.

Use `summary.json` as the main machine-readable payload. Keep `map.html` as the
MVP interactive map, with `map.svg` as fallback. Do not confuse the GPS course
rose with external wind direction.
```
