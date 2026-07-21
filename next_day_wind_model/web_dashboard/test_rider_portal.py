from __future__ import annotations

import io
import json
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import db_store
from next_day_wind_model.web_dashboard import app as portal
from wingfoil_analysis import pipeline as wingfoil_pipeline


class RiderPortalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        portal.DATA_DIR = Path(self.temp_dir.name)
        portal.app.config.update(TESTING=True, SECRET_KEY="rider-portal-test-secret")

        conn = db_store.connect_db(self.temp_dir.name)
        db_store.init_db(conn)
        self.user_id = db_store.create_user(conn, "test-rider", portal._hash_password("test-password"))
        self.other_user_id = db_store.create_user(conn, "other-rider", portal._hash_password("other-password"))
        conn.close()
        self.client = portal.app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _set_user(self, user_id: int | None) -> None:
        with self.client.session_transaction() as current_session:
            current_session.clear()
            if user_id is not None:
                current_session["user_id"] = user_id

    def _set_profile(self, user_id: int, public_username: str, rider_name: str = "Private rider name") -> None:
        conn = db_store.connect_db(self.temp_dir.name)
        db_store.upsert_user_profile(conn, user_id, public_username, rider_name, 80, "Valkenburgse meer")
        conn.close()

    def _create_submission(
        self,
        user_id: int,
        rider: str,
        day: str,
        visibility: str | None = None,
        measured_summary: dict[str, float] | None = None,
        rider_review: str | None = None,
        perceived_wind_variability: str | None = "moderate",
    ) -> int:
        start_ms, end_ms = portal._local_session_bounds(day, "12:00", "14:00")
        experience = {
            "user_id": user_id,
            "rider": rider,
            "spot": "Valkenburgse meer",
            "date": day,
            "start_time": "12:00",
            "end_time": "14:00",
            "start_ts": start_ms,
            "end_ts": end_ms,
            "session_rating": 4,
            "perceived_wind_variability": perceived_wind_variability,
            "rider_review": f"Review by {rider}" if rider_review is None else rider_review,
            "rider_weight": 80,
            "wing_size": 5,
            "foil_size": 1200,
            "rider_notes": f"Private notes by {rider}",
            "measured_wind": {
                "status": "ok" if measured_summary else "unavailable",
                "records": [],
                "plot_records": [],
                "summary": measured_summary or {},
            },
        }
        if visibility is not None:
            experience["visibility"] = visibility
        conn = db_store.connect_db(self.temp_dir.name)
        experience_id = db_store.create_surf_experience(conn, experience)
        conn.close()
        return experience_id

    def _activity_file(self, filename: str = "session.gpx"):
        gpx = b'<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">\n  <trk><name>Test</name><trkseg>\n    <trkpt lat="52.1" lon="4.4"><time>2026-01-20T11:00:00Z</time></trkpt>\n    <trkpt lat="52.1005" lon="4.4005"><time>2026-01-20T11:00:10Z</time></trkpt>\n  </trkseg></trk>\n</gpx>\n'
        return (io.BytesIO(gpx), filename)

    def _store_activity_summary(
        self,
        experience_id: int,
        user_id: int,
        summary: dict[str, object],
        stats: dict[str, object] | None = None,
    ) -> None:
        conn = db_store.connect_db(self.temp_dir.name)
        db_store.upsert_surf_experience_activity_analysis(
            conn,
            {
                "experience_id": experience_id,
                "user_id": user_id,
                "uploaded_at": "2026-01-20T12:00:00Z",
                "original_filename": "session.gpx",
                "stored_filename": "session.gpx",
                "file_type": "gpx",
                "status": "ok",
                "summary": summary,
                "stats": stats or {},
                "artifacts": {},
                "warnings": [],
                "errors": [],
                "analysis_version": "test-version",
            },
        )
        conn.close()

    def _tcx_bytes(self) -> bytes:
        return b'''<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2" xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2">
  <Activities><Activity Sport="Other"><Lap StartTime="2026-01-20T11:00:00Z"><Track>
    <Trackpoint><Time>2026-01-20T11:00:00Z</Time><Position><LatitudeDegrees>52.1</LatitudeDegrees><LongitudeDegrees>4.4</LongitudeDegrees></Position><AltitudeMeters>1.0</AltitudeMeters><DistanceMeters>0.0</DistanceMeters><HeartRateBpm><Value>101</Value></HeartRateBpm><Extensions><ns3:TPX><ns3:Speed>0.0</ns3:Speed></ns3:TPX></Extensions></Trackpoint>
    <Trackpoint><Time>2026-01-20T11:00:10Z</Time><Position><LatitudeDegrees>52.1005</LatitudeDegrees><LongitudeDegrees>4.4005</LongitudeDegrees></Position><AltitudeMeters>1.2</AltitudeMeters><DistanceMeters>65.0</DistanceMeters><HeartRateBpm><Value>105</Value></HeartRateBpm><Extensions><ns3:TPX><ns3:Speed>6.5</ns3:Speed></ns3:TPX></Extensions></Trackpoint>
  </Track></Lap></Activity></Activities>
</TrainingCenterDatabase>
'''

    def _zip_activity_file(self, entries: dict[str, bytes], filename: str = "session.zip"):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries.items():
                archive.writestr(name, content)
        payload.seek(0)
        return (payload, filename)

    def test_forecast_session_lookup_has_matching_range_index(self) -> None:
        conn = db_store.connect_db(self.temp_dir.name)
        index_columns = [
            (row[2], row[3])
            for row in conn.execute("PRAGMA index_xinfo(idx_fc_site_target_fetched)").fetchall()
            if row[5]
        ]
        plan = conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT target_ts, payload
            FROM forecasts
            WHERE site = ?
              AND target_ts >= ?
              AND target_ts <= ?
            ORDER BY target_ts, fetched_ts DESC
            """,
            ("valkenburgsemeer", 0, 1),
        ).fetchall()
        conn.close()

        self.assertEqual(
            index_columns,
            [("site", 0), ("target_ts", 0), ("fetched_ts", 1)],
        )
        self.assertIn(
            "USING INDEX idx_fc_site_target_fetched",
            " ".join(str(row[3]) for row in plan),
        )

    def _mock_analysis_payload(self, input_file, output_dir, wind_context=None, raise_on_error=False, **_kwargs):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary = {
            "activity": {
                "total_distance_m": 1234.0,
                "water_time_formatted": "20m 0s",
                "avg_run_distance_m": 657.0,
                "avg_speed_on_foil_kmh": 18.5,
                "sample_count": 42,
            },
            "runs_summary": {"count": 13},
            "runs": [{"run_id": index, "distance_m": 650 + index} for index in range(1, 14)],
            "falls_summary": {"count": 1},
            "warnings": ["summary warning"],
        }
        (output_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (output_path / "map.svg").write_text("<svg><title>map</title></svg>", encoding="utf-8")
        (output_path / "map.html").write_text("<html><body>map</body></html>", encoding="utf-8")
        plot_svg = {
            "run_distance_distribution.svg": "<svg><text>0-100 m</text><text>100-200 m</text><text>200-300 m</text><text>&gt;300 m</text></svg>",
            "run_speed_distribution.svg": "<svg><text>&lt;10 km/h</text><text>10-15 km/h</text><text>15-20 km/h</text><text>20-25 km/h</text><text>25-30 km/h</text><text>&gt;30 km/h</text></svg>",
            "run_wind_angle_distribution.svg": "<svg><title>run_wind_angle_distribution.svg</title></svg>",
            "run_speed.svg": "<svg><text>Run distance (m)</text><text>Mean speed (km/h)</text><text>700</text><text>30</text></svg>",
        }
        for name, content in plot_svg.items():
            (output_path / name).write_text(content, encoding="utf-8")
        run_rows = [
            "run_id,distance_m,distance_km,mean_speed_kmh,max_speed_kmh,wind_angle_class",
            *[f"{index},{650 + index},{(650 + index) / 1000:.3f},17.54,24.04,crosswind" for index in range(1, 14)],
        ]
        (output_path / "runs.csv").write_text("\n".join(run_rows) + "\n", encoding="utf-8")
        return {
            "status": "ok",
            "analysis_version": "test-version",
            "input_filename": Path(input_file).name,
            "input_type": Path(input_file).suffix.lower().lstrip("."),
            "summary_json": "summary.json",
            "map_html": "map.html",
            "map_svg": "map.svg",
            "runs_csv": "runs.csv",
            "artifacts": {
                "summary_json": "summary.json",
                "runs_csv": "runs.csv",
                "map_svg": "map.svg",
                "map_html": "map.html",
                "run_distance_distribution_svg": "run_distance_distribution.svg",
                "run_speed_distribution_svg": "run_speed_distribution.svg",
                "run_wind_angle_distribution_svg": "run_wind_angle_distribution.svg",
                "run_speed_profile_svg": "run_speed.svg",
            },
            "plots": {
                "speed_distribution_svg": "run_speed_distribution.svg",
                "distance_distribution_svg": "run_distance_distribution.svg",
                "run_speed_profile_svg": "run_speed.svg",
                "wind_angle_distribution_svg": "run_wind_angle_distribution.svg",
            },
            "stats": {
                "distance_km": 1.234,
                "max_speed_kmh": 24.0,
                "avg_speed_on_foil_kmh": 18.5,
                "run_count": 13,
                "avg_run_distance_m": 657.0,
                "fall_count": 1,
                "track_point_count": 42,
            },
            "warnings": ["timestamps are irregular: median interval is 0.70s and some records deviate by more than 2.00s"],
        }

    def _valid_form(self, visibility: str | None = None) -> dict[str, str]:
        form = {
            "Rider": "Form Rider",
            "Spot": "Valkenburgse meer",
            "Date": "2026-01-20",
            "StartHour": "12",
            "StartMinute": "00",
            "EndHour": "14",
            "EndMinute": "00",
            "SessionRating": "4",
            "PerceivedWindVariability": "moderate",
            "RiderReview": "Form review",
            "RiderWeight": "80",
            "WingSize": "5",
            "FoilSize": "1200",
            "RiderNotes": "Form notes",
        }
        if visibility is not None:
            form["Visibility"] = visibility
        return form

    def test_my_sessions_login_flow_and_account_indicator(self) -> None:
        legacy_submission = self._create_submission(self.user_id, "Legacy Rider", "2026-01-09", "private")
        protected = self.client.get("/experiences")
        self.assertEqual(protected.status_code, 302)
        self.assertEqual(protected.headers["Location"], "/?login=1&next=/experiences")

        login_page = self.client.get(protected.headers["Location"])
        self.assertEqual(login_page.status_code, 200)
        self.assertIn(b'id="open-login">Login</button>', login_page.data)
        self.assertIn(b'name="next" value="/experiences"', login_page.data)
        self.assertGreaterEqual(login_page.data.count(b">Email</label>"), 2)
        self.assertIn(b"Existing users can still use their current login name.", login_page.data)
        self.assertIn(b'placeholder="you@example.com"', login_page.data)

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        login_response = self.client.post(
            "/login",
            data={
                "_csrf_token": csrf_token,
                "username": "test-rider",
                "password": "test-password",
                "next": "/experiences",
            },
        )
        self.assertEqual(login_response.status_code, 302)
        self.assertEqual(login_response.headers["Location"], "/experiences")

        sessions_page = self.client.get("/experiences")
        self.assertEqual(sessions_page.status_code, 200)
        self.assertIn(b'aria-label="Primary navigation"', sessions_page.data)
        self.assertIn(b'aria-label="Account menu for test-rider"', sessions_page.data)
        self.assertIn(b'class="account-dropdown"', sessions_page.data)
        self.assertIn(b">Profile</a>", sessions_page.data)
        self.assertIn(b">Logout</button>", sessions_page.data)
        self.assertNotIn(b"Logged in as", sessions_page.data)
        self.assertEqual(sessions_page.data.count(b'href="/experiences"'), 0)
        self.assertEqual(sessions_page.data.count(b'href="/experience/new"'), 1)
        self.assertIn(b'class="portal-navigation-panel"', sessions_page.data)
        self.assertIn(b'class="portal-action-row"', sessions_page.data)
        self.assertIn(b'class="button portal-action-button portal-action-new"', sessions_page.data)
        self.assertIn(b'class="button portal-action-button portal-action-dashboard"', sessions_page.data)
        self.assertEqual(sessions_page.data.count(b'class="button portal-action-button'), 2)
        self.assertIn(b".portal-action-button { box-sizing: border-box; width: 100%; min-width: 0; min-height: 44px; border-radius: 8px; padding: 9px 8px; color: #fff;", sessions_page.data)
        self.assertIn(b".portal-action-new { border-color: #00C8B3; background: #00C8B3; color: #fff;", sessions_page.data)
        self.assertIn(b".portal-action-dashboard { border-color: #135f86; background: #135f86; color: #fff;", sessions_page.data)
        self.assertLess(sessions_page.data.index(b">New submission</a>"), sessions_page.data.index(b">Forecast dashboard</a>"))
        self.assertEqual(sessions_page.data.count(b">Forecast dashboard</a>"), 1)
        self.assertIn(b">Show submissions</span>", sessions_page.data)
        self.assertIn(b'class="segmented-toggle"', sessions_page.data)
        self.assertEqual(sessions_page.data.count(b'class="segmented-toggle-segment'), 2)
        self.assertEqual(sessions_page.data.count(b'aria-current="page"'), 1)
        self.assertIn(b">My</a>", sessions_page.data)
        self.assertIn(b">All</a>", sessions_page.data)
        self.assertIn(f'href="/experiences/{legacy_submission}"'.encode(), sessions_page.data)

        root_handoff = self.client.get("/?next=/experience/new")
        self.assertEqual(root_handoff.status_code, 302)
        self.assertEqual(root_handoff.headers["Location"], "/experience/new")

        new_submission = self.client.get("/experience/new")
        self.assertEqual(new_submission.status_code, 200)
        self.assertIn(b"<h2>New submission</h2>", new_submission.data)
        self.assertEqual(new_submission.data.count(b'href="/experiences"'), 1)
        self.assertEqual(new_submission.data.count(b'href="/experience/new"'), 0)
        self.assertIn(b'class="primary-nav portal-action-row submission-form-nav"', new_submission.data)
        self.assertIn(b'class="button portal-action-button portal-action-new" href="/experiences">Submissions</a>', new_submission.data)
        self.assertIn(b'class="button portal-action-button portal-action-dashboard"', new_submission.data)
        self.assertNotIn(b">My submissions</a>", new_submission.data)
        self.assertLess(new_submission.data.index(b">Submissions</a>"), new_submission.data.index(b">Forecast dashboard</a>"))

        edit_submission = self.client.get(f"/experiences/{legacy_submission}/edit")
        self.assertEqual(edit_submission.status_code, 200)
        self.assertIn(b"<h2>Modify submission</h2>", edit_submission.data)
        self.assertIn(b'class="primary-nav portal-action-row submission-form-nav"', edit_submission.data)
        self.assertIn(b'class="button portal-action-button portal-action-new" href="/experiences">Submissions</a>', edit_submission.data)
        self.assertIn(b'class="button portal-action-button portal-action-dashboard"', edit_submission.data)

        profile_page = self.client.get("/profile")
        self.assertEqual(profile_page.status_code, 200)
        self.assertIn(b"Public username", profile_page.data)
        self.assertIn(b"Shown with public submissions as your rider identity. Your email/login name is not shown publicly.", profile_page.data)
        self.assertIn(b"Leave it empty to use your public username.", profile_page.data)

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        profile_response = self.client.post(
            "/profile",
            data={
                "_csrf_token": csrf_token,
                "PublicUsername": "Legacy Public Rider",
                "RiderName": "Legacy Private Rider",
                "RiderWeight": "80",
                "DefaultSpot": "Valkenburgse meer",
            },
        )
        self.assertEqual(profile_response.status_code, 302)
        conn = db_store.connect_db(self.temp_dir.name)
        self.assertEqual(db_store.get_user_profile(conn, self.user_id)["public_username"], "Legacy Public Rider")
        conn.close()

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        logout_response = self.client.post("/logout", data={"_csrf_token": csrf_token})
        self.assertEqual(logout_response.status_code, 302)
        self.assertEqual(logout_response.headers["Location"], "/")

    def test_registration_requires_email_but_accepts_valid_email(self) -> None:
        register_page = self.client.get("/?login=1")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        invalid = self.client.post(
            "/register",
            data={"_csrf_token": csrf_token, "username": "new-legacy-name", "password": "test-password"},
        )
        self.assertEqual(invalid.status_code, 302)
        conn = db_store.connect_db(self.temp_dir.name)
        self.assertIsNone(db_store.get_user_by_username(conn, "new-legacy-name"))
        conn.close()

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        valid = self.client.post(
            "/register",
            data={"_csrf_token": csrf_token, "username": "new.rider@example.com", "password": "test-password"},
        )
        self.assertEqual(valid.status_code, 302)
        self.assertEqual(valid.headers["Location"], "/profile")
        conn = db_store.connect_db(self.temp_dir.name)
        created = db_store.get_user_by_username(conn, "new.rider@example.com")
        conn.close()
        self.assertIsNotNone(created)

    def test_duplicate_public_username_is_rejected_for_different_user(self) -> None:
        self._set_profile(self.other_user_id, "Existing Public Rider", "Other Rider")
        self._set_user(self.user_id)

        self.assertEqual(self.client.get("/profile").status_code, 200)
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        response = self.client.post(
            "/profile",
            data={
                "_csrf_token": csrf_token,
                "PublicUsername": "  existing public rider  ",
                "RiderName": "Unique Rider",
                "RiderWeight": "80",
                "DefaultSpot": "Valkenburgse meer",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Public username is already in use.", response.data)
        conn = db_store.connect_db(self.temp_dir.name)
        self.assertIsNone(db_store.get_user_profile(conn, self.user_id))
        conn.close()

    def test_duplicate_rider_name_is_rejected_for_different_user(self) -> None:
        self._set_profile(self.other_user_id, "Other Public Rider", "Existing Rider Name")
        self._set_user(self.user_id)

        self.assertEqual(self.client.get("/profile").status_code, 200)
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        response = self.client.post(
            "/profile",
            data={
                "_csrf_token": csrf_token,
                "PublicUsername": "Unique Public Rider",
                "RiderName": "existing rider name",
                "RiderWeight": "80",
                "DefaultSpot": "Valkenburgse meer",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rider name is already in use.", response.data)
        conn = db_store.connect_db(self.temp_dir.name)
        self.assertIsNone(db_store.get_user_profile(conn, self.user_id))
        conn.close()

    def test_same_user_can_save_existing_profile_identity_case_insensitively(self) -> None:
        self._set_profile(self.user_id, "Current Public Rider", "Current Rider Name")
        self._set_user(self.user_id)

        self.assertEqual(self.client.get("/profile").status_code, 200)
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        response = self.client.post(
            "/profile",
            data={
                "_csrf_token": csrf_token,
                "PublicUsername": " current public rider ",
                "RiderName": " current rider name ",
                "RiderWeight": "80",
                "DefaultSpot": "Valkenburgse meer",
            },
        )

        self.assertEqual(response.status_code, 302)
        conn = db_store.connect_db(self.temp_dir.name)
        profile = db_store.get_user_profile(conn, self.user_id)
        conn.close()
        self.assertEqual(profile["public_username"], "current public rider")
        self.assertEqual(profile["rider_name"], "current rider name")

    def test_blank_public_username_and_rider_name_are_allowed(self) -> None:
        self._set_profile(self.other_user_id, "", "")
        self._set_user(self.user_id)

        self.assertEqual(self.client.get("/profile").status_code, 200)
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        response = self.client.post(
            "/profile",
            data={
                "_csrf_token": csrf_token,
                "PublicUsername": "   ",
                "RiderName": "",
                "RiderWeight": "80",
                "DefaultSpot": "Valkenburgse meer",
            },
        )

        self.assertEqual(response.status_code, 302)
        conn = db_store.connect_db(self.temp_dir.name)
        profile = db_store.get_user_profile(conn, self.user_id)
        conn.close()
        self.assertEqual(profile["public_username"], "")
        self.assertEqual(profile["rider_name"], "")

    def test_rider_name_defaults_to_public_username_without_rewriting_profile(self) -> None:
        self._set_user(self.user_id)
        self.assertEqual(self.client.get("/profile").status_code, 200)

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        profile_response = self.client.post(
            "/profile",
            data={
                "_csrf_token": csrf_token,
                "PublicUsername": "Public Default",
                "RiderName": "",
                "RiderWeight": "80",
                "DefaultSpot": "Valkenburgse meer",
            },
        )
        self.assertEqual(profile_response.status_code, 302)

        conn = db_store.connect_db(self.temp_dir.name)
        self.assertEqual(db_store.get_user_profile(conn, self.user_id)["rider_name"], "")
        conn.close()

        profile_page = self.client.get("/profile")
        self.assertEqual(profile_page.status_code, 200)
        self.assertIn(b'name="RiderName" value="Public Default"', profile_page.data)

        new_page = self.client.get("/experience/new")
        self.assertEqual(new_page.status_code, 200)
        self.assertIn(b'name="Rider" value="Public Default"', new_page.data)
        self.assertNotIn(b"Prefilled from RiderName", new_page.data)

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        custom_form = self._valid_form("public")
        custom_form["_csrf_token"] = csrf_token
        custom_form["Rider"] = "Submission Custom Rider"
        response = self.client.post("/experience/new", data=custom_form)
        self.assertEqual(response.status_code, 302)

        conn = db_store.connect_db(self.temp_dir.name)
        row = db_store.get_surf_experience(conn, self.user_id, 1)
        self.assertEqual(row["rider"], "Submission Custom Rider")
        self.assertEqual(db_store.get_user_profile(conn, self.user_id)["rider_name"], "")
        conn.close()

    def test_explicit_rider_name_overrides_public_username_for_new_submission_default(self) -> None:
        self._set_profile(self.user_id, "Public Identity", "Private Form Rider")
        self._set_user(self.user_id)

        new_page = self.client.get("/experience/new")
        self.assertEqual(new_page.status_code, 200)
        self.assertIn(b'name="Rider" value="Private Form Rider"', new_page.data)
        self.assertNotIn(b'name="Rider" value="Public Identity"', new_page.data)

    def test_public_overview_uses_public_username_not_private_rider_name(self) -> None:
        self._set_profile(self.user_id, "Visible Public Rider", "Private Form Rider")
        public_id = self._create_submission(self.user_id, "Submission Private Rider", "2026-02-06", "public")
        self._set_user(self.other_user_id)

        overview = self.client.get("/experiences?scope=all")
        self.assertEqual(overview.status_code, 200)
        self.assertIn(f'href="/experiences/{public_id}"'.encode(), overview.data)
        self.assertIn(b"Visible Public Rider", overview.data)
        self.assertNotIn(b"Private Form Rider", overview.data)
        self.assertNotIn(b"Submission Private Rider", overview.data)

    def test_measured_report_min_max_and_variability_without_session_trend(self) -> None:
        start_ms, end_ms = portal._local_session_bounds("2026-01-15", "12:00", "13:00")
        self.assertIsNotNone(start_ms)
        self.assertIsNotNone(end_ms)

        speeds = [16.0, 8.0, 16.0, 8.0, 12.0]
        rows = [
            {
                "timestamp": start_ms + index * 3 * 60 * 1000,
                "AverageWind": speed,
                "MinWind": speed - 2.0,
                "MaxWind": speed + 3.0,
                "WindDirection": 225.0,
            }
            for index, speed in enumerate(speeds)
        ]

        conn = db_store.connect_db(self.temp_dir.name)
        db_store.upsert_observations(conn, "valkenburgsemeer", rows)
        measured = db_store.get_measured_wind_for_session(conn, "Valkenburgse meer", start_ms, end_ms)
        conn.close()

        self.assertEqual(measured["status"], "ok")
        self.assertAlmostEqual(measured["summary"]["wind_variability"], 0.962673611111111)
        self.assertEqual(
            measured["summary"]["wind_variability_kind"],
            db_store.POWER_WIND_VARIABILITY_KIND,
        )
        sparse_conn = db_store.connect_db(self.temp_dir.name)
        sparse = db_store.get_measured_wind_for_session(
            sparse_conn,
            "Valkenburgse meer",
            start_ms,
            start_ms + 3 * 60 * 1000,
        )
        sparse_conn.close()
        self.assertAlmostEqual(sparse["summary"]["wind_variability"], 0.986328125)
        self.assertTrue(all("measured_wind_min" in record for record in measured["plot_records"]))
        self.assertTrue(all("measured_wind_max" in record for record in measured["plot_records"]))
        plot_timestamps = [record["timestamp"] for record in measured["plot_records"]]
        self.assertEqual(plot_timestamps, [row["timestamp"] for row in rows])
        self.assertEqual(
            [plot_timestamps[index + 1] - plot_timestamps[index] for index in range(len(plot_timestamps) - 1)],
            [3 * 60 * 1000] * (len(plot_timestamps) - 1),
        )

        plot = portal._measured_wind_plot(
            {
                "date": "2026-01-15",
                "start_time": "12:00",
                "end_time": "13:00",
                "measured_wind": measured,
            }
        )
        self.assertTrue(plot["available"])
        self.assertTrue(plot["min_points"])
        self.assertTrue(plot["max_points"])
        self.assertNotIn("trend_points", plot)
        self.assertIn("threshold_y", plot)

    def test_objective_variability_mean_power_metric_used_in_detail_and_overview(self) -> None:
        start_ms, end_ms = portal._local_session_bounds("2026-01-16", "12:00", "13:00")
        records = [
            {"timestamp": start_ms, "measured_wind_speed": 10.0, "measured_wind_min": 8.0, "measured_wind_max": 13.0},
            {"timestamp": start_ms + 3 * 60 * 1000, "measured_wind_speed": 12.0, "measured_wind_min": 10.0, "measured_wind_max": 15.0},
            {"timestamp": start_ms + 6 * 60 * 1000, "measured_wind_speed": 14.0, "measured_wind_min": 11.0, "measured_wind_max": 16.0},
        ]
        measured = {
            "status": "ok",
            "records": records,
            "plot_records": records,
            "summary": {
                "point_count": len(records),
                "avg_wind_speed": 12.0,
                "max_wind_speed": 14.0,
                "min_wind_speed": 10.0,
                "max_wind_gust": 16.0,
            },
        }
        conn = db_store.connect_db(self.temp_dir.name)
        experience_id = db_store.create_surf_experience(
            conn,
            {
                "user_id": self.user_id,
                "rider": "Objective Rider",
                "spot": "Valkenburgse meer",
                "date": "2026-01-16",
                "start_time": "12:00",
                "end_time": "13:00",
                "start_ts": start_ms,
                "end_ts": end_ms,
                "session_rating": 4,
                "perceived_wind_variability": "very_gusty",
                "rider_review": "Objective review",
                "rider_weight": 80,
                "wing_size": 5,
                "foil_size": 1200,
                "rider_notes": "Notes",
                "visibility": "public",
                "measured_wind": measured,
            },
        )
        detail_row = db_store.get_surf_experience(conn, self.user_id, experience_id)
        overview_row = next(row for row in db_store.list_surf_experiences(conn, self.user_id) if row["id"] == experience_id)
        conn.close()

        expected = sum([1.05, 125.0 / 144.0, 135.0 / 196.0]) / 3.0
        self.assertAlmostEqual(detail_row["measured_wind"]["summary"]["wind_variability"], expected)
        self.assertEqual(detail_row["measured_wind"]["summary"]["wind_variability_kind"], db_store.POWER_WIND_VARIABILITY_KIND)
        self.assertAlmostEqual(overview_row["wind_variability"], expected)
        self.assertEqual(overview_row["perceived_wind_variability"], "very_gusty")

        with portal.app.test_request_context(f"/experiences/{experience_id}"):
            portal.session["user_id"] = self.user_id
            detail = portal.render_template(
                "submission_detail.html",
                row=detail_row,
                wind_plot={"available": False},
                wind_variability_plot={"available": False},
                current_day_archive_plot=None,
            ).encode()
        self.assertIn(f"{expected:.2f}".encode(), detail)
        self.assertNotIn(f"{expected:.2f} kts".encode(), detail)
        self.assertIn(b"Very gusty", detail)

    def test_session_plots_use_europe_amsterdam_summer_time_labels(self) -> None:
        start_ms, end_ms = portal._local_session_bounds("2026-06-09", "12:30", "15:00")
        self.assertEqual(datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%H:%M"), "10:30")
        self.assertEqual(datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime("%H:%M"), "13:00")

        records = [
            {
                "timestamp": start_ms + index * 30 * 60 * 1000,
                "measured_wind_speed": 12.0 + index,
                "measured_wind_min": 9.0 + index,
                "measured_wind_max": 15.0 + index,
                "measured_wind_direction": 225.0,
            }
            for index in range(6)
        ]
        row = {
            "date": "2026-06-09",
            "start_time": "12:30",
            "end_time": "15:00",
            "measured_wind": {"records": records, "plot_records": records},
        }

        measured_plot = portal._measured_wind_plot(row)
        variability_plot = portal._measured_wind_variability_plot(row)

        self.assertEqual(measured_plot["hour_ticks"], variability_plot["hour_ticks"])
        self.assertEqual([tick["label"] for tick in measured_plot["hour_ticks"]], ["12:30", "13:00", "13:30", "14:00", "14:30", "15:00"])
        self.assertEqual(float(measured_plot["hour_ticks"][0]["x"]), measured_plot["pad_left"])
        self.assertEqual(float(variability_plot["hour_ticks"][0]["x"]), variability_plot["pad_left"])

    def test_session_plots_use_precise_ten_minute_time_limits(self) -> None:
        start_ms, end_ms = portal._local_session_bounds("2026-06-09", "13:20", "15:50")
        self.assertEqual(end_ms - start_ms, 150 * 60 * 1000)
        records = [
            {
                "timestamp": start_ms + index * 30 * 60 * 1000,
                "measured_wind_speed": 12.0 + index,
                "measured_wind_min": 9.0 + index,
                "measured_wind_max": 15.0 + index,
                "measured_wind_direction": 225.0,
            }
            for index in range(6)
        ]
        row = {
            "date": "2026-06-09",
            "start_time": "13:20",
            "end_time": "15:50",
            "measured_wind": {"records": records, "plot_records": records},
        }

        measured_plot = portal._measured_wind_plot(row)
        variability_plot = portal._measured_wind_variability_plot(row)

        self.assertTrue(measured_plot["available"])
        self.assertEqual(measured_plot["hour_ticks"][0]["label"], "13:20")
        self.assertEqual(measured_plot["hour_ticks"][-1]["label"], "15:50")
        self.assertEqual(variability_plot["hour_ticks"][0]["label"], "13:20")
        self.assertEqual(variability_plot["hour_ticks"][-1]["label"], "15:50")
        self.assertEqual(float(measured_plot["hour_ticks"][0]["x"]), measured_plot["pad_left"])
        self.assertEqual(float(measured_plot["hour_ticks"][-1]["x"]), measured_plot["plot_right"])

    def test_measured_wind_variability_plot_uses_power_based_rolling_window(self) -> None:
        start_ms, _end_ms = portal._local_session_bounds("2026-01-15", "12:00", "13:00")
        records = [
            {
                "timestamp": start_ms + index * 3 * 60 * 1000,
                "measured_wind_speed": speed,
                "measured_wind_min": speed - 2.0,
                "measured_wind_max": speed + 3.0,
            }
            for index, speed in enumerate([8.0, 16.0, 8.0, 16.0, 12.0])
        ]
        measured = {
            "records": records,
            "plot_records": records[::2],
            "summary": {},
        }

        plot = portal._measured_wind_variability_plot(
            {
                "date": "2026-01-15",
                "start_time": "12:00",
                "end_time": "13:00",
                "measured_wind": measured,
            }
        )

        self.assertTrue(plot["available"])
        self.assertEqual(plot["min_value"], 0.5)
        self.assertEqual(plot["max_value"], 2.0)
        self.assertEqual(plot["window_minutes"], 30)
        self.assertEqual(plot["min_periods"], 3)
        self.assertEqual(len(plot["raw_points"].split()), 5)
        self.assertEqual(len(plot["trend_points"].split()), 3)
        self.assertEqual(plot["latest_label"], "Variability: 0.96")

    def test_variability_plot_textbox_uses_session_mean_not_latest_trend(self) -> None:
        start_ms, _end_ms = portal._local_session_bounds("2026-01-15", "12:00", "13:00")
        records = [
            {
                "timestamp": start_ms,
                "measured_wind_speed": 10.0,
                "measured_wind_min": 1.0,
                "measured_wind_max": 20.0,
            }
        ]
        records.extend(
            {
                "timestamp": start_ms + index * 3 * 60 * 1000,
                "measured_wind_speed": 10.0,
                "measured_wind_min": 9.0,
                "measured_wind_max": 11.0,
            }
            for index in range(1, 12)
        )
        measured = {"records": records, "plot_records": records, "summary": {}}
        row = {
            "date": "2026-01-15",
            "start_time": "12:00",
            "end_time": "13:00",
            "measured_wind": measured,
        }

        session_mean = db_store.measured_wind_power_variability_mean(measured)
        latest_trend = 0.4
        plot = portal._measured_wind_variability_plot(row)

        self.assertAlmostEqual(session_mean, 0.6991666666666667)
        self.assertNotEqual(f"Variability: {session_mean:.2f}", f"Variability: {latest_trend:.2f}")
        self.assertEqual(plot["latest_label"], f"Variability: {session_mean:.2f}")
        self.assertNotEqual(plot["latest_label"], f"Variability: {latest_trend:.2f}")

        measured["summary"] = {"max_wind_gust": 20.0, "wind_variability": session_mean}
        detail_row = {
            "date": "2026-01-15",
            "spot": "Valkenburgse meer",
            "start_time": "12:00",
            "end_time": "13:00",
            "avg_forecast_temperature": 10.0,
            "session_rating": 4,
            "perceived_wind_variability": "moderate",
            "rider": "Test Rider",
            "rider_weight": 80,
            "wing_size": 5,
            "foil_size": 1200,
            "rider_review": "Good",
            "rider_notes": "",
            "measured_wind_status": "ok",
            "measured_wind": measured,
            "avg_measured_wind_speed": 10.0,
            "max_measured_wind_speed": 10.0,
            "min_measured_wind_speed": 10.0,
            "mean_measured_direction_display": "SW (225 deg)",
            "visibility": "private",
            "is_owner": True,
            "submitted_by": "Test Public Rider",
            "rider_display": "Test Rider",
        }
        with portal.app.test_request_context("/experiences/1"):
            portal.session["user_id"] = 1
            detail = portal.render_template(
                "submission_detail.html",
                row=detail_row,
                wind_plot={"available": False},
                wind_variability_plot=plot,
                current_day_archive_plot=None,
            ).encode()
        formatted_mean = f"{session_mean:.2f}".encode()
        self.assertGreaterEqual(detail.count(formatted_mean), 2)
        self.assertIn(f"Variability: {session_mean:.2f}".encode(), detail)
        self.assertNotIn(f"Variability: {latest_trend:.2f}".encode(), detail)

    def test_measured_report_keeps_six_minute_source_cadence_when_that_is_all_available(self) -> None:
        start_ms, end_ms = portal._local_session_bounds("2026-01-15", "12:00", "13:00")
        rows = [
            {
                "timestamp": start_ms + index * 6 * 60 * 1000,
                "AverageWind": 10.0 + index,
                "MinWind": 9.0 + index,
                "MaxWind": 12.0 + index,
                "WindDirection": 225.0,
            }
            for index in range(4)
        ]

        conn = db_store.connect_db(self.temp_dir.name)
        db_store.upsert_observations(conn, "valkenburgsemeer", rows)
        measured = db_store.get_measured_wind_for_session(conn, "Valkenburgse meer", start_ms, end_ms)
        conn.close()

        plot_timestamps = [record["timestamp"] for record in measured["plot_records"]]
        self.assertEqual(plot_timestamps, [row["timestamp"] for row in rows])
        self.assertEqual(
            [plot_timestamps[index + 1] - plot_timestamps[index] for index in range(len(plot_timestamps) - 1)],
            [6 * 60 * 1000] * (len(plot_timestamps) - 1),
        )
        plot = portal._measured_wind_plot(
            {
                "date": "2026-01-15",
                "start_time": "12:00",
                "end_time": "13:00",
                "measured_wind": measured,
            }
        )
        self.assertTrue(plot["available"])
        self.assertEqual(len(plot["speed_points"].split()), len(rows))

    def test_measured_wind_plot_speed_uses_measured_speed_records_only(self) -> None:
        start_ms, _end_ms = portal._local_session_bounds("2026-01-15", "12:00", "13:00")
        measured = {
            "plot_records": [
                {"timestamp": start_ms, "measured_wind_speed": 10.0, "measured_wind_max": 12.0},
                {"timestamp": start_ms + 3 * 60 * 1000, "measured_wind_speed": 12.0, "measured_wind_max": 14.0},
                {"timestamp": start_ms + 6 * 60 * 1000, "measured_wind_max": 16.0},
            ],
            "summary": {},
        }

        plot = portal._measured_wind_plot(
            {
                "date": "2026-01-15",
                "start_time": "12:00",
                "end_time": "13:00",
                "measured_wind": measured,
            }
        )

        speed_coords = plot["speed_points"].split()
        self.assertEqual(len(speed_coords), 2)
        self.assertNotIn("trend_points", plot)

    def test_portal_head_serves_icons_without_dashboard_refresh_logic(self) -> None:
        portal_home = self.client.get("/")
        self.assertEqual(portal_home.status_code, 200)
        self.assertIn(b'rel="apple-touch-icon"', portal_home.data)
        self.assertIn(b'href="/site.webmanifest?v=1"', portal_home.data)
        self.assertIn(b'href="/site-assets/favicon.ico?v=1"', portal_home.data)
        self.assertIn(b'href="/site-assets/favicon-32x32.png?v=1"', portal_home.data)
        self.assertIn(b'href="/site-assets/favicon-16x16.png?v=1"', portal_home.data)
        self.assertIn(b'href="/site-assets/apple-touch-icon.png?v=1"', portal_home.data)
        self.assertNotIn(b'id="dashboard-refresh"', portal_home.data)

        self._set_user(self.user_id)
        submission_form = self.client.get("/experience/new")
        self.assertEqual(submission_form.status_code, 200)
        self.assertNotIn(b'id="dashboard-refresh"', submission_form.data)
        self.assertNotIn(b"visibilitychange", submission_form.data)

        manifest = self.client.get("/site.webmanifest")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(manifest.mimetype, "application/manifest+json")
        self.assertIn(b"site-assets/icon-192x192.png?v=1", manifest.data)
        self.assertIn(b"site-assets/icon-512x512.png?v=1", manifest.data)
        manifest.close()
        icon_response = self.client.get("/site-assets/apple-touch-icon.png")
        self.assertEqual(icon_response.status_code, 200)
        icon_response.close()
        self.assertEqual(self.client.get("/site-assets/not-an-icon.png").status_code, 404)

    def test_published_dashboard_has_throttled_foreground_refresh_and_versioned_assets(self) -> None:
        model_dir = str(Path(__file__).resolve().parents[1])
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from next_day_wind_model import update_model_and_predict as updater

        output_dir = Path(self.temp_dir.name) / "published-dashboard"
        source_dir = Path(self.temp_dir.name) / "dashboard-source"
        source_dir.mkdir()
        next_day_png = source_dir / "next-day.png"
        current_day_png = source_dir / "current-day.png"
        next_day_png.write_bytes(b"next-day-image")
        current_day_png.write_bytes(b"current-day-image")
        next_day_csv = source_dir / "next-day.csv"
        updater.pd.DataFrame(
            {
                "target_time_local": ["2026-01-15T12:00:00+01:00"],
                "forecast_wind_speed": [12.0],
            }
        ).to_csv(next_day_csv, index=False)
        missing_csv = source_dir / "missing-current-day.csv"

        copied = updater.publish_web_dashboard(
            web_out_dir=output_dir,
            local_tz="Europe/Amsterdam",
            web_refresh_seconds=360,
            next_day_png=next_day_png,
            next_day_png_mobile=None,
            next_day_csv=next_day_csv,
            current_day_png=current_day_png,
            current_day_png_mobile=None,
            current_day_csv=missing_csv,
            daily_mae_png=None,
            daily_mae_png_mobile=None,
            daily_mae_csv=None,
            gate_eval_png=None,
            gate_eval_csv=None,
            direction_spider_png=None,
            direction_spider_csv=None,
            current_day_direction_spider_png=None,
            current_day_direction_spider_csv=None,
            spot_name="Valkenburgse meer",
            companion_app_base_url="https://portal.example",
        )

        html = (output_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="dashboard-refresh"', html)
        self.assertIn("↻ Refresh", html)
        self.assertEqual(updater._site_display_name("valkenburgsemeer"), "Valkenburgse meer")
        self.assertIn("<title>Super local wind prediction - Valkenburgse meer</title>", html)
        self.assertIn("<h1>Super local wind prediction</h1>", html)
        self.assertNotIn("<h1>Super local wind prediction Valkenburgse meer", html)
        self.assertIn('<p class="development-status"># Under development #</p>', html)
        self.assertIn('<strong data-dashboard-spot>Valkenburgse meer</strong>', html)
        self.assertLess(html.index("<h1>Super local wind prediction</h1>"), html.index("# Under development #"))
        self.assertLess(html.index("# Under development #"), html.index("data-dashboard-spot"))
        self.assertLess(html.index("data-dashboard-spot"), html.index("Last updated:"))
        self.assertNotIn(">New submission</a>", html)
        self.assertNotIn(">My sessions</a>", html)
        self.assertIn('class="button primary dashboard-action"', html)
        self.assertIn('class="button dashboard-action dashboard-refresh"', html)
        self.assertLess(html.index(">Rider portal</a>"), html.index("↻ Refresh</button>"))
        self.assertIn("grid-template-columns: repeat(auto-fit, minmax(140px, 1fr))", html)
        self.assertNotIn('http-equiv="refresh"', html)
        self.assertIn('rel="apple-touch-icon"', html)
        self.assertIn('href="site-assets/favicon.ico?v=1"', html)
        self.assertIn('href="site-assets/favicon-32x32.png?v=1"', html)
        self.assertIn('href="site-assets/favicon-16x16.png?v=1"', html)
        self.assertIn('href="site-assets/apple-touch-icon.png?v=1"', html)
        self.assertIn('href="site.webmanifest?v=1"', html)
        self.assertIn('src="dashboard_refresh.js?v=', html)
        self.assertIn('current_day_predictions.png?v=', html)
        self.assertIn('next_day_predictions.png?v=', html)
        self.assertIn('data-json-url="next_day_interactive_data.json?v=', html)
        self.assertIn("minimumIntervalMs: 300000", html)
        self.assertIn('metadataUrl: "metadata_update.json"', html)
        self.assertIn("dashboard_refresh.js", copied)
        self.assertIn("site.webmanifest", copied)
        self.assertTrue((output_dir / "site-assets" / "apple-touch-icon.png").exists())

        metadata = json.loads((output_dir / "metadata_update.json").read_text(encoding="utf-8"))
        self.assertIn("current_day_predictions.png", metadata["static_images"])
        self.assertIn("next_day_predictions.png", metadata["static_images"])
        self.assertEqual(metadata["interactive_json"], ["next_day_interactive_data.json"])

        refresh_source = (output_dir / "dashboard_refresh.js").read_text(encoding="utf-8")
        self.assertIn('cache: "no-store"', refresh_source)
        self.assertIn('document.addEventListener("visibilitychange"', refresh_source)
        self.assertIn('window.addEventListener("pageshow"', refresh_source)
        self.assertIn("shouldCheck(lastCheckAt", refresh_source)
        self.assertIn("if (inFlight)", refresh_source)
        self.assertIn('reason: "manual-after-check"', refresh_source)
        self.assertIn('checkForUpdate({ force: true, manual: true', refresh_source)

    def test_current_day_plot_frame_keeps_three_minute_measured_rows(self) -> None:
        model_dir = str(Path(__file__).resolve().parents[1])
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from next_day_wind_model import update_model_and_predict as updater

        dense_times = updater.pd.date_range(
            "2026-01-15 08:00", periods=3, freq="6min", tz="Europe/Amsterdam"
        )
        actual_times = updater.pd.date_range(
            "2026-01-15 08:00", periods=5, freq="3min", tz="Europe/Amsterdam"
        )
        actual_raw = updater.pd.DataFrame(
            {
                "actual_avg": [10.0, 11.0, 12.0, 13.0, 14.0],
                "actual_min": [9.0, 10.0, 11.0, 12.0, 13.0],
                "actual_max": [12.0, 13.0, 14.0, 15.0, 16.0],
                "actual_dir": [220.0, 221.0, 222.0, 223.0, 224.0],
            },
            index=actual_times,
        )
        forecast_columns = {
            "forecast_wind_speed": updater.np.array([10.0, 11.0, 12.0], dtype=updater.np.float32),
            "forecast_wind_min": updater.np.array([9.0, 10.0, 11.0], dtype=updater.np.float32),
            "forecast_wind_max": updater.np.array([12.0, 13.0, 14.0], dtype=updater.np.float32),
            "forecast_wind_dir_deg": updater.np.array([220.0, 221.0, 222.0], dtype=updater.np.float32),
            "lstm_pred_wind_speed_full": updater.np.array([10.5, 11.5, 12.5], dtype=updater.np.float32),
            "lstm_pred_wind_dir_deg_full": updater.np.array([220.0, 221.0, 222.0], dtype=updater.np.float32),
            "lstm_pred_wind_speed": updater.np.array([updater.np.nan, 11.5, 12.5], dtype=updater.np.float32),
            "lstm_pred_wind_dir_deg": updater.np.array([updater.np.nan, 221.0, 222.0], dtype=updater.np.float32),
        }

        table = updater._build_current_day_plot_frame(
            dense_times,
            forecast_columns,
            actual_raw,
            now_local=actual_times[-1],
            future_start=dense_times[-1] + updater.pd.Timedelta(hours=1),
        )

        measured_times = table.loc[table["actual_wind_speed"].notna(), "time_local"]
        measured_deltas = measured_times.diff().dropna().dt.total_seconds().to_list()
        self.assertEqual(measured_deltas, [180.0, 180.0, 180.0, 180.0])
        forecast_times = table.loc[table["is_forecast_grid"], "time_local"]
        forecast_deltas = forecast_times.diff().dropna().dt.total_seconds().to_list()
        self.assertEqual(forecast_deltas, [360.0, 360.0])
        self.assertEqual(int(table["is_actual_observation"].sum()), 5)

    def test_current_day_plot_frame_keeps_six_minute_measured_rows_when_source_is_six_minute(self) -> None:
        model_dir = str(Path(__file__).resolve().parents[1])
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from next_day_wind_model import update_model_and_predict as updater

        dense_times = updater.pd.date_range(
            "2026-01-15 08:00", periods=4, freq="6min", tz="Europe/Amsterdam"
        )
        actual_raw = updater.pd.DataFrame(
            {
                "actual_avg": [10.0, 11.0, 12.0, 13.0],
                "actual_min": [9.0, 10.0, 11.0, 12.0],
                "actual_max": [12.0, 13.0, 14.0, 15.0],
                "actual_dir": [220.0, 221.0, 222.0, 223.0],
            },
            index=dense_times,
        )
        forecast_columns = {
            col: updater.np.arange(len(dense_times), dtype=updater.np.float32)
            for col in [
                "forecast_wind_speed",
                "forecast_wind_min",
                "forecast_wind_max",
                "forecast_wind_dir_deg",
                "lstm_pred_wind_speed_full",
                "lstm_pred_wind_dir_deg_full",
                "lstm_pred_wind_speed",
                "lstm_pred_wind_dir_deg",
            ]
        }

        table = updater._build_current_day_plot_frame(
            dense_times,
            forecast_columns,
            actual_raw,
            now_local=dense_times[-1],
            future_start=dense_times[-1] + updater.pd.Timedelta(hours=1),
        )

        measured_times = table.loc[table["actual_wind_speed"].notna(), "time_local"]
        measured_deltas = measured_times.diff().dropna().dt.total_seconds().to_list()
        self.assertEqual(measured_deltas, [360.0, 360.0, 360.0])
        self.assertEqual(len(table), len(dense_times))

    def test_current_day_table_mae_scores_three_minute_actual_rows(self) -> None:
        model_dir = str(Path(__file__).resolve().parents[1])
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from next_day_wind_model import update_model_and_predict as updater

        dense_times = updater.pd.date_range(
            "2026-01-15 08:00", periods=3, freq="6min", tz="Europe/Amsterdam"
        )
        actual_times = updater.pd.date_range(
            "2026-01-15 08:00", periods=5, freq="3min", tz="Europe/Amsterdam"
        )
        actual_raw = updater.pd.DataFrame(
            {"actual_avg": [10.0, 10.5, 11.0, 11.5, 12.0]},
            index=actual_times,
        )
        table = updater._build_current_day_plot_frame(
            dense_times,
            {
                "forecast_wind_speed": updater.np.array([10.0, 11.0, 12.0], dtype=updater.np.float32),
                "forecast_wind_min": updater.np.array([9.0, 10.0, 11.0], dtype=updater.np.float32),
                "forecast_wind_max": updater.np.array([12.0, 13.0, 14.0], dtype=updater.np.float32),
                "forecast_wind_dir_deg": updater.np.array([220.0, 221.0, 222.0], dtype=updater.np.float32),
                "lstm_pred_wind_speed_full": updater.np.array([10.0, 11.0, 12.0], dtype=updater.np.float32),
                "lstm_pred_wind_dir_deg_full": updater.np.array([220.0, 221.0, 222.0], dtype=updater.np.float32),
                "lstm_pred_wind_speed": updater.np.array([updater.np.nan, 11.0, 12.0], dtype=updater.np.float32),
                "lstm_pred_wind_dir_deg": updater.np.array([updater.np.nan, 221.0, 222.0], dtype=updater.np.float32),
            },
            actual_raw,
            now_local=actual_times[-1],
            future_start=dense_times[-1] + updater.pd.Timedelta(hours=1),
        )

        metric = updater.compute_current_day_table_mae(table)
        self.assertTrue(metric["available"])
        self.assertEqual(metric["measurement_point_count"], 5)

    def test_current_day_actual_trend_uses_measured_rows_only(self) -> None:
        model_dir = str(Path(__file__).resolve().parents[1])
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from next_day_wind_model import update_model_and_predict as updater

        time_local = updater.pd.date_range(
            "2026-01-15 08:00", periods=5, freq="3min", tz="Europe/Amsterdam"
        )
        trend = updater._measured_actual_trend_values(
            updater.pd.Series(time_local),
            updater.np.array([10.0, 12.0, updater.np.nan, updater.np.nan, updater.np.nan]),
        )

        self.assertEqual(trend[0], 10.0)
        self.assertEqual(trend[1], 11.0)
        self.assertTrue(updater.np.isnan(trend[2]))
        self.assertTrue(updater.np.isnan(trend[-1]))

    def test_submission_detail_labels_units_and_primary_navigation_only(self) -> None:
        row = {
            "date": "2026-01-15",
            "spot": "Valkenburgse meer",
            "start_time": "12:00",
            "end_time": "13:00",
            "avg_forecast_temperature": 10.0,
            "session_rating": 4,
            "perceived_wind_variability": "gusty",
            "rider": "Test Rider",
            "rider_weight": 80,
            "wing_size": 5,
            "foil_size": 1200,
            "rider_review": "Good",
            "rider_notes": "",
            "measured_wind_status": "ok",
            "measured_wind": {"summary": {"max_wind_gust": 30.0, "wind_variability": 1.8}},
            "avg_measured_wind_speed": 14.4,
            "max_measured_wind_speed": 20.0,
            "min_measured_wind_speed": 9.0,
            "mean_measured_direction_display": "SW (208 deg)",
            "visibility": "private",
            "is_owner": True,
            "submitted_by": "Test Public Rider",
            "rider_display": "Test Rider",
        }
        with portal.app.test_request_context("/experiences/1"):
            portal.session["user_id"] = 1
            detail = portal.render_template(
                "submission_detail.html",
                row=row,
                wind_plot={"available": False},
                wind_variability_plot={"available": False},
                current_day_archive_plot=None,
            ).encode()

        for label in (b"avg speed", b"max avg speed", b"min avg speed", b"max gust", b"Wind variability", b"avg direction"):
            self.assertIn(label, detail)
        for value in (b"14.4 kts", b"20.0 kts", b"9.0 kts", b"30.0 kts", b"1.80", b"SW (208 deg)", b"Gusty"):
            self.assertIn(value, detail)
        self.assertIn(b'aria-label="4 out of 5"', detail)
        self.assertNotIn(b"<dt>SessionRating</dt><dd>4/5</dd>", detail)
        self.assertIn(b"Make this submission public to share it.", detail)
        self.assertNotIn(b'class="share-icon-button"', detail)
        self.assertNotIn(b'class="primary share-button"', detail)
        self.assertIn(b"<dt>Date</dt><dd>Thursday 15 January 2026</dd>", detail)
        self.assertNotIn(b'class="form-actions"', detail)
        self.assertEqual(detail.count(b'href="/experiences"'), 1)
        self.assertEqual(detail.count(b'href="/experience/new"'), 1)

    def test_current_day_archive_lookup_supports_daily_and_timestamped_names(self) -> None:
        archive_dir = Path(self.temp_dir.name) / "current_day_plot_archive"
        archive_dir.mkdir()
        old_early = archive_dir / "20260115-220000_current_day_predictions.png"
        old_late = archive_dir / "20260115-225959_current_day_predictions.png"
        daily = archive_dir / "current_day_predictions_2026-01-15.png"
        daily_mobile = archive_dir / "current_day_predictions_mobile_2026-01-15.png"
        old_early.write_bytes(b"old early")
        old_late.write_bytes(b"old late")
        original_archive_dir = portal.CURRENT_DAY_PLOT_ARCHIVE_DIR
        portal.CURRENT_DAY_PLOT_ARCHIVE_DIR = archive_dir
        try:
            with portal.app.test_request_context():
                old_match = portal._current_day_archive_plot_for_submission("2026-01-15")
            self.assertIsNotNone(old_match)
            self.assertEqual(old_match["filename"], old_late.name)

            daily.write_bytes(b"daily")
            daily_mobile.write_bytes(b"daily mobile")
            with portal.app.test_request_context():
                daily_match = portal._current_day_archive_plot_for_submission("2026-01-15")
            self.assertIsNotNone(daily_match)
            self.assertEqual(daily_match["filename"], daily.name)

            for archive_name in [old_late.name, daily.name, daily_mobile.name]:
                response = self.client.get(f"/current-day-plot-archive/{archive_name}")
                try:
                    self.assertEqual(response.status_code, 200)
                finally:
                    response.close()
            invalid_response = self.client.get("/current-day-plot-archive/not_an_archive.png")
            try:
                self.assertEqual(invalid_response.status_code, 404)
            finally:
                invalid_response.close()
        finally:
            portal.CURRENT_DAY_PLOT_ARCHIVE_DIR = original_archive_dir

    def test_submission_detail_places_wind_variability_before_archive_plot(self) -> None:
        row = {
            "date": "2026-01-15",
            "spot": "Valkenburgse meer",
            "start_time": "12:00",
            "end_time": "13:00",
            "avg_forecast_temperature": 10.0,
            "session_rating": 4,
            "perceived_wind_variability": "gusty",
            "rider": "Test Rider",
            "rider_weight": 80,
            "wing_size": 5,
            "foil_size": 1200,
            "rider_review": "Good",
            "rider_notes": "",
            "measured_wind_status": "ok",
            "measured_wind": {"summary": {"max_wind_gust": 30.0, "wind_variability": 1.8}},
            "avg_measured_wind_speed": 14.4,
            "max_measured_wind_speed": 20.0,
            "min_measured_wind_speed": 9.0,
            "mean_measured_direction_display": "SW (208 deg)",
            "visibility": "private",
            "is_owner": True,
            "submitted_by": "Test Public Rider",
            "rider_display": "Test Rider",
        }
        with portal.app.test_request_context("/experiences/1"):
            portal.session["user_id"] = 1
            detail = portal.render_template(
                "submission_detail.html",
                row=row,
                wind_plot={"available": False},
                wind_variability_plot={
                    "available": True,
                    "width": 820,
                    "height": 178,
                    "pad_left": 48,
                    "pad_top": 18,
                    "plot_right": 798,
                    "axis_y": 132,
                    "plot_width": 750,
                    "plot_height": 114,
                    "raw_points": "48.0,80.0 85.5,70.0",
                    "trend_points": "85.5,70.0",
                    "hour_ticks": [{"x": "48.0", "label": "12:00"}],
                    "y_ticks": [{"y": "132.0", "label_y": "136.0", "label": "0.5"}],
                    "threshold_y": "78.8",
                    "threshold_label_y": "73.8",
                    "latest_label": "Variability: 1.45",
                },
                current_day_archive_plot={"url": "/current-day-plot-archive/example_current_day_predictions.png"},
            ).encode()

        self.assertLess(detail.index(b"<h2>Measured wind</h2>"), detail.index(b"<h2>Wind variability</h2>"))
        self.assertLess(detail.index(b"<h2>Wind variability</h2>"), detail.index(b"<h2>Measured wind full day from archive</h2>"))
        self.assertIn(b"Variability: 1.45", detail)
        self.assertIn(b"30-min avg", detail)

    def test_visibility_defaults_and_form_validation(self) -> None:
        default_private_id = self._create_submission(self.user_id, "Default Private", "2026-01-10")
        invalid_private_id = self._create_submission(self.user_id, "Invalid Private", "2026-01-11", "unexpected")
        public_id = self._create_submission(self.user_id, "Public Rider", "2026-01-12", "public")

        conn = db_store.connect_db(self.temp_dir.name)
        self.assertEqual(db_store.get_surf_experience(conn, self.user_id, default_private_id)["visibility"], "private")
        self.assertEqual(db_store.get_surf_experience(conn, self.user_id, invalid_private_id)["visibility"], "private")
        self.assertEqual(db_store.get_surf_experience(conn, self.user_id, public_id)["visibility"], "public")
        visibility_column = next(row for row in conn.execute("PRAGMA table_info(surf_experiences)") if row[1] == "visibility")
        perceived_column = next(row for row in conn.execute("PRAGMA table_info(surf_experiences)") if row[1] == "perceived_wind_variability")
        self.assertEqual(visibility_column[4], "'private'")
        self.assertEqual(perceived_column[2], "TEXT")
        conn.close()

        missing_visibility, missing_errors = portal._validate_experience_form(self._valid_form())
        self.assertEqual(missing_visibility["visibility"], "public")
        self.assertEqual(missing_errors, [])
        public_form, public_errors = portal._validate_experience_form(self._valid_form("public"))
        self.assertEqual(public_form["visibility"], "public")
        self.assertEqual(public_form["perceived_wind_variability"], "moderate")
        self.assertEqual(public_errors, [])
        gusty_form = self._valid_form()
        gusty_form["PerceivedWindVariability"] = "gusty"
        gusty_experience, gusty_errors = portal._validate_experience_form(gusty_form)
        self.assertEqual(gusty_errors, [])
        self.assertEqual(gusty_experience["perceived_wind_variability"], "gusty")
        invalid_perceived_form = self._valid_form()
        invalid_perceived_form["PerceivedWindVariability"] = "wild"
        _, invalid_perceived_errors = portal._validate_experience_form(invalid_perceived_form)
        self.assertIn("PerceivedWindVariability must be one of the allowed options.", invalid_perceived_errors)
        ten_minute_form = self._valid_form()
        ten_minute_form.update({"StartHour": "15", "StartMinute": "20", "EndHour": "17", "EndMinute": "50"})
        ten_minute_experience, ten_minute_errors = portal._validate_experience_form(ten_minute_form)
        self.assertEqual(ten_minute_errors, [])
        self.assertEqual(ten_minute_experience["start_time"], "15:20")
        self.assertEqual(ten_minute_experience["end_time"], "17:50")
        self.assertEqual(ten_minute_experience["end_ts"] - ten_minute_experience["start_ts"], 150 * 60 * 1000)
        legacy_time_form = self._valid_form()
        for key in ["StartHour", "StartMinute", "EndHour", "EndMinute"]:
            legacy_time_form.pop(key)
        legacy_time_form["StartTime"] = "15:30"
        legacy_time_form["EndTime"] = "17:30"
        legacy_experience, legacy_errors = portal._validate_experience_form(legacy_time_form)
        self.assertEqual(legacy_errors, [])
        self.assertEqual(legacy_experience["start_time"], "15:30")
        self.assertEqual(legacy_experience["end_time"], "17:30")
        for invalid_minute in ["07", "61", ""]:
            invalid_time_form = self._valid_form()
            invalid_time_form["StartMinute"] = invalid_minute
            _, invalid_time_errors = portal._validate_experience_form(invalid_time_form)
            self.assertIn("Start minute must use a 10-minute interval.", invalid_time_errors)
        existing_exact_form = self._valid_form()
        existing_exact_form.update({"StartHour": "13", "StartMinute": "07", "EndHour": "15", "EndMinute": "44"})
        existing_exact, existing_exact_errors = portal._validate_experience_form(
            existing_exact_form,
            existing_times={"start_time": "13:07", "end_time": "15:44"},
        )
        self.assertEqual(existing_exact_errors, [])
        self.assertEqual(existing_exact["start_time"], "13:07")
        self.assertEqual(existing_exact["end_time"], "15:44")
        equal_time_form = self._valid_form()
        equal_time_form.update({"StartHour": "15", "StartMinute": "20", "EndHour": "15", "EndMinute": "20"})
        _, equal_time_errors = portal._validate_experience_form(equal_time_form)
        self.assertIn("EndTime must be after StartTime.", equal_time_errors)
        earlier_time_form = self._valid_form()
        earlier_time_form.update({"StartHour": "15", "StartMinute": "20", "EndHour": "15", "EndMinute": "10"})
        _, earlier_time_errors = portal._validate_experience_form(earlier_time_form)
        self.assertIn("EndTime must be after StartTime.", earlier_time_errors)
        _, invalid_errors = portal._validate_experience_form(self._valid_form("friends"))
        self.assertIn("Visibility must be private or public.", invalid_errors)
        custom_foil_form = self._valid_form()
        custom_foil_form["FoilSize"] = "1501"
        custom_foil_experience, custom_foil_errors = portal._validate_experience_form(custom_foil_form)
        self.assertEqual(custom_foil_errors, [])
        self.assertEqual(custom_foil_experience["foil_size"], 1501)
        for invalid_foil_size, expected_error in [
            ("0", "FoilSize must be a positive whole number."),
            ("-100", "FoilSize must be a positive whole number."),
            ("1500.5", "FoilSize must be a whole number."),
            ("not-a-number", "FoilSize must be a whole number."),
            ("", "FoilSize is required."),
        ]:
            invalid_foil_form = self._valid_form()
            invalid_foil_form["FoilSize"] = invalid_foil_size
            _, invalid_foil_errors = portal._validate_experience_form(invalid_foil_form)
            self.assertIn(expected_error, invalid_foil_errors)

    def test_legacy_account_schema_migration_preserves_users_profiles_and_ownership(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                username_norm TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_ts INTEGER NOT NULL,
                created_iso TEXT NOT NULL,
                last_login_ts INTEGER,
                last_login_iso TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE user_profiles (
                user_id INTEGER PRIMARY KEY,
                rider_name TEXT,
                rider_weight INTEGER,
                default_spot TEXT,
                updated_ts INTEGER NOT NULL,
                updated_iso TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE surf_experiences (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                start_time TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO users(id, username, username_norm, password_hash, created_ts, created_iso)
            VALUES (7, 'legacy-login', 'legacy-login', 'hash', 1, 'created')
            """
        )
        conn.execute(
            """
            INSERT INTO user_profiles(user_id, rider_name, rider_weight, default_spot, updated_ts, updated_iso)
            VALUES (7, 'Existing private name', 81, 'Valkenburgse meer', 2, 'updated')
            """
        )
        conn.execute(
            """
            INSERT INTO surf_experiences(id, user_id, date, start_time)
            VALUES (1, 7, '2026-01-01', '12:00')
            """
        )

        db_store.init_account_db(conn)
        db_store.init_account_db(conn)

        visibility_column = next(row for row in conn.execute("PRAGMA table_info(surf_experiences)") if row[1] == "visibility")
        share_token_column = next(row for row in conn.execute("PRAGMA table_info(surf_experiences)") if row[1] == "share_token")
        shared_at_column = next(row for row in conn.execute("PRAGMA table_info(surf_experiences)") if row[1] == "shared_at")
        perceived_column = next(row for row in conn.execute("PRAGMA table_info(surf_experiences)") if row[1] == "perceived_wind_variability")
        public_username_column = next(row for row in conn.execute("PRAGMA table_info(user_profiles)") if row[1] == "public_username")
        self.assertEqual(visibility_column[3], 1)
        self.assertEqual(visibility_column[4], "'private'")
        self.assertEqual(conn.execute("SELECT visibility FROM surf_experiences WHERE id = 1").fetchone()[0], "private")
        self.assertEqual(public_username_column[2], "TEXT")
        self.assertEqual(share_token_column[2], "TEXT")
        self.assertEqual(shared_at_column[2], "TEXT")
        self.assertEqual(perceived_column[2], "TEXT")
        self.assertIsNone(conn.execute("SELECT perceived_wind_variability FROM surf_experiences WHERE id = 1").fetchone()[0])
        self.assertIsNone(conn.execute("SELECT share_token FROM surf_experiences WHERE id = 1").fetchone()[0])
        self.assertEqual(conn.execute("SELECT username FROM users WHERE id = 7").fetchone()[0], "legacy-login")
        self.assertEqual(conn.execute("SELECT rider_name, public_username FROM user_profiles WHERE user_id = 7").fetchone(), ("Existing private name", None))
        self.assertEqual(conn.execute("SELECT user_id FROM surf_experiences WHERE id = 1").fetchone()[0], 7)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM surf_experiences").fetchone()[0], 1)
        conn.close()

    def test_split_time_submission_handles_missing_observations_table(self) -> None:
        self._set_user(self.user_id)
        conn = db_store.connect_db(self.temp_dir.name)
        conn.execute("DROP TABLE observations")
        conn.commit()
        conn.close()

        new_page = self.client.get("/experience/new")
        self.assertEqual(new_page.status_code, 200)
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        form = self._valid_form()
        form.update({"StartHour": "13", "StartMinute": "20", "EndHour": "15", "EndMinute": "50"})
        form["_csrf_token"] = csrf_token
        response = self.client.post("/experience/new", data=form, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Experience submitted. Measured wind data was unavailable for that session.", response.data)
        self.assertIn(b"13:20 to 15:50", response.data)
        conn = db_store.connect_db(self.temp_dir.name)
        row = conn.execute(
            "SELECT start_time, end_time, measured_wind_status FROM surf_experiences WHERE date = ?",
            ("2026-01-20",),
        ).fetchone()
        conn.close()
        self.assertEqual(row, ("13:20", "15:50", "unavailable"))

    def test_new_submission_route_stores_private_and_public_visibility(self) -> None:
        self._set_user(self.user_id)
        new_page = self.client.get("/experience/new")
        self.assertEqual(new_page.status_code, 200)
        private_visibility = new_page.data.split(b'value="private"', 1)[1].split(b'>', 1)[0]
        public_visibility = new_page.data.split(b'value="public"', 1)[1].split(b'>', 1)[0]
        self.assertNotIn(b'checked', private_visibility)
        self.assertIn(b'checked', public_visibility)
        self.assertIn(b'<button class="primary" type="submit">Save submission</button>', new_page.data)
        self.assertIn(b'<select id="StartHour" name="StartHour" aria-label="Start hour" required>', new_page.data)
        self.assertIn(b'<select id="StartMinute" name="StartMinute" aria-label="Start minute" required>', new_page.data)
        self.assertIn(b'<select id="EndHour" name="EndHour" aria-label="End hour" required>', new_page.data)
        self.assertIn(b'<select id="EndMinute" name="EndMinute" aria-label="End minute" required>', new_page.data)
        self.assertNotIn(b'<label class="sub-label"', new_page.data)
        self.assertIn(b'<option value="13"', new_page.data)
        start_minute_select = new_page.data.split(b'id="StartMinute"', 1)[1].split(b'</select>', 1)[0]
        for minute in [b"00", b"10", b"20", b"30", b"40", b"50"]:
            self.assertIn(b'<option value="' + minute + b'"', start_minute_select)
        self.assertNotIn(b'<option value="07"', start_minute_select)
        self.assertIn(b"function selectedTimeToMinutes(hourSelect, minuteSelect)", new_page.data)
        self.assertIn(b"setSelectedTime(endHourSelect, endMinuteSelect, start + 120)", new_page.data)
        self.assertIn(b"startMinuteSelect.addEventListener", new_page.data)
        self.assertIn(
            b'<div class="form-row-full form-review-row">\n          <label for="RiderReview">RiderReview</label>\n          <textarea id="RiderReview" name="RiderReview">',
            new_page.data,
        )
        self.assertIn(
            b'<div class="form-row-full form-notes-row">\n          <label for="RiderNotes">Private RiderNotes</label>\n          <textarea id="RiderNotes" name="RiderNotes">',
            new_page.data,
        )
        self.assertIn(b'<div class="form-row-full activity-file-control form-upload-row">', new_page.data)
        self.assertIn(b'<fieldset class="form-row-full visibility-row">', new_page.data)
        self.assertIn(b"Private RiderNotes", new_page.data)
        self.assertIn(b"Only visible to you. Not shown on public submissions.", new_page.data)
        self.assertIn(b'<label class="activity-file-trigger" for="ActivityFile">Upload file</label>', new_page.data)
        self.assertIn(b'.tcx,.TCX', new_page.data)
        self.assertIn(b'.zip,.ZIP', new_page.data)
        self.assertIn(b"Upload a FIT, TCX, GPX, KML, or ZIP activity file.", new_page.data)
        self.assertIn(b"No activity file uploaded", new_page.data)
        self.assertNotIn(b'<label for="ActivityFile">Activity file</label>', new_page.data)
        self.assertNotIn(b"Optional FIT, GPX or KML file from your session.", new_page.data)
        self.assertIn(b'<div class="form-date-row">', new_page.data)
        self.assertIn(b'<div class="form-time-start time-pair-field form-row-mobile-full">', new_page.data)
        self.assertIn(b'<div class="form-time-end time-pair-field form-row-mobile-full">', new_page.data)
        self.assertIn(b'<div class="form-rating-row">', new_page.data)
        self.assertIn(b'<div class="form-perceived-row">', new_page.data)
        self.assertIn(b'<select id="PerceivedWindVariability" name="PerceivedWindVariability" required>', new_page.data)
        self.assertIn(b'<label for="PerceivedWindVariability">Perceived wind variability *</label>', new_page.data)
        self.assertNotIn(b'<label for="PerceivedWindVariability">Perceived variability *</label>', new_page.data)
        self.assertIn(b'<option value="gusty"', new_page.data)
        self.assertIn(b'<input id="FoilSize" name="FoilSize" type="number" min="1" step="1" value="1500" required>', new_page.data)
        self.assertNotIn(b'<select id="FoilSize"', new_page.data)
        self.assertLess(new_page.data.index(b'id="Date"'), new_page.data.index(b'id="StartHour"'))
        self.assertLess(new_page.data.index(b'id="StartHour"'), new_page.data.index(b'id="EndHour"'))
        self.assertLess(new_page.data.index(b'name="SessionRating"'), new_page.data.index(b'id="PerceivedWindVariability"'))

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        private_form = self._valid_form("private")
        private_form.update({"StartHour": "13", "StartMinute": "20", "EndHour": "15", "EndMinute": "50"})
        private_form["FoilSize"] = "1140"
        private_form["PerceivedWindVariability"] = "gusty"
        private_form["_csrf_token"] = csrf_token
        private_response = self.client.post("/experience/new", data=private_form)
        self.assertEqual(private_response.status_code, 302)

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        public_form = self._valid_form("public")
        public_form["Date"] = "2026-01-21"
        public_form["FoilSize"] = "1501"
        public_form["_csrf_token"] = csrf_token
        public_response = self.client.post("/experience/new", data=public_form)
        self.assertEqual(public_response.status_code, 302)

        conn = db_store.connect_db(self.temp_dir.name)
        rows = db_store.list_surf_experiences(conn, self.user_id)
        visibility_by_date = {row["date"]: row["visibility"] for row in rows}
        private_time_row = conn.execute(
            "SELECT start_time, end_time, end_ts - start_ts FROM surf_experiences WHERE date = ?",
            ("2026-01-20",),
        ).fetchone()
        private_notes_row = conn.execute(
            "SELECT rider_notes FROM surf_experiences WHERE date = ?",
            ("2026-01-20",),
        ).fetchone()
        private_perceived_row = conn.execute(
            "SELECT perceived_wind_variability FROM surf_experiences WHERE date = ?",
            ("2026-01-20",),
        ).fetchone()
        foil_rows = dict(
            conn.execute("SELECT date, foil_size FROM surf_experiences WHERE date IN (?, ?)", ("2026-01-20", "2026-01-21"))
        )
        conn.close()
        self.assertEqual(visibility_by_date["2026-01-20"], "private")
        self.assertEqual(visibility_by_date["2026-01-21"], "public")
        self.assertEqual(private_time_row, ("13:20", "15:50", 150 * 60 * 1000))
        detail = self.client.get(private_response.headers["Location"])
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"13:20 to 15:50", detail.data)
        self.assertIn(b"1140 cm2", detail.data)
        overview = self.client.get("/experiences")
        self.assertEqual(overview.status_code, 200)
        self.assertIn(b">13:20</td>", overview.data)
        self.assertIn(b">15:50</td>", overview.data)
        self.assertEqual(private_notes_row, ("Form notes",))
        self.assertEqual(private_perceived_row, ("gusty",))
        self.assertEqual(foil_rows["2026-01-20"], 1140)
        self.assertEqual(foil_rows["2026-01-21"], 1501)

        for invalid_foil_size in ["0", "-100", "1500.5", "not-a-number"]:
            with self.client.session_transaction() as current_session:
                csrf_token = current_session["_csrf_token"]
            invalid_form = self._valid_form("public")
            invalid_form["Date"] = "2026-01-22"
            invalid_form["FoilSize"] = invalid_foil_size
            invalid_form["_csrf_token"] = csrf_token
            invalid_response = self.client.post("/experience/new", data=invalid_form)
            self.assertEqual(invalid_response.status_code, 200)
            self.assertIn(b"FoilSize", invalid_response.data)

    def test_perceived_variability_sort_uses_ordinal_order(self) -> None:
        self._set_profile(self.user_id, "Ordinal Rider", "Ordinal Private")
        submissions = [
            ("Very Gusty Rider", "2026-03-01", "very_gusty"),
            ("Moderate Rider", "2026-03-02", "moderate"),
            ("Very Steady Rider", "2026-03-03", "very_steady"),
            ("Gusty Rider", "2026-03-04", "gusty"),
            ("Steady Rider", "2026-03-05", "steady"),
        ]
        ids = {
            value: self._create_submission(self.user_id, rider, day, "public", perceived_wind_variability=value)
            for rider, day, value in submissions
        }

        self._set_user(self.user_id)
        response = self.client.get("/experiences?scope=all&sort=perceived_wind_variability&dir=asc")
        self.assertEqual(response.status_code, 200)
        ordered_ids = [ids[value] for value in ["very_steady", "steady", "moderate", "gusty", "very_gusty"]]
        positions = [response.data.index(f'href="/experiences/{experience_id}"'.encode()) for experience_id in ordered_ids]
        self.assertEqual(positions, sorted(positions))

    def test_generated_activity_dashboard_formats_session_stats(self) -> None:
        base_time = datetime(2026, 1, 20, 11, 0, tzinfo=timezone.utc)
        samples = [
            wingfoil_pipeline.Sample(
                index=0,
                lat=52.0,
                lon=4.0,
                ele_m=None,
                time=base_time,
                dt_s=0.0,
                segment_distance_m=0.0,
                speed_mps=0.0,
                smooth_speed_mps=0.0,
                in_run=True,
            )
        ]
        runs = []
        for index in range(1, 30):
            samples.append(
                wingfoil_pipeline.Sample(
                    index=index,
                    lat=52.0,
                    lon=4.0 + index * 0.0001,
                    ele_m=None,
                    time=base_time + timedelta(seconds=index * 10),
                    dt_s=10.0,
                    segment_distance_m=190.0,
                    speed_mps=19.0,
                    smooth_speed_mps=19.0,
                    in_run=True,
                )
            )
            runs.append(
                wingfoil_pipeline.Run(
                    run_id=index,
                    start_index=index,
                    end_index=index,
                    start_time=base_time + timedelta(seconds=index * 10),
                    end_time=base_time + timedelta(seconds=index * 10),
                    duration_s=10.0,
                    distance_m=190.0,
                    mean_speed_mps=19.0,
                    median_speed_mps=19.0,
                    max_speed_mps=19.0,
                    start_lat=52.0,
                    start_lon=4.0 + index * 0.0001,
                    end_lat=52.0,
                    end_lon=4.0 + index * 0.0001,
                    wind_angle_class="crosswind",
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            map_path = Path(temp_dir) / "map.html"
            wingfoil_pipeline.write_map_html(
                map_path,
                Path("private-session.tcx"),
                samples,
                runs,
                [],
                wingfoil_pipeline.WindContext(),
                water_time_s=3457.0,
            )
            map_text = map_path.read_text(encoding="utf-8")

        self.assertIn('"foil_distance_m": 5510.0', map_text)
        self.assertIn('"avg_run_distance_m": 190.0', map_text)
        self.assertIn('statCard(distanceOnFoil, "distance on foil")', map_text)
        self.assertIn('formatDistanceMeters(data.activity.avg_run_distance_m)', map_text)
        self.assertIn('return `${Math.round(distanceM)} m`', map_text)
        self.assertIn('return `${distanceKm.toFixed(1).replace(/\\.0$/, "")} km`', map_text)
        self.assertNotIn('statCard(data.activity.water_time_formatted, "time in water")', map_text)
        self.assertNotIn('>TIME IN WATER<', map_text.upper())

    def test_activity_metrics_render_and_sort_on_submissions_overview(self) -> None:
        small_id = self._create_submission(self.user_id, "Small Activity", "2026-03-11", "private")
        large_id = self._create_submission(self.user_id, "Large Activity", "2026-03-12", "private")
        missing_id = self._create_submission(self.user_id, "Missing Activity", "2026-03-13", "private")
        self._store_activity_summary(
            small_id,
            self.user_id,
            {
                "activity": {"avg_run_distance_m": 236.0},
                "runs_summary": {"count": 3},
                "runs": [{"run_id": 1, "distance_m": 200.0}, {"run_id": 2, "distance_m": 242.0}, {"run_id": 3, "distance_m": 300.0}],
            },
        )
        self._store_activity_summary(
            large_id,
            self.user_id,
            {
                "activity": {"avg_run_distance_m": 1200.0},
                "runs_summary": {"count": 2},
                "runs": [{"run_id": 1, "distance_m": 1100.0}, {"run_id": 2, "distance_m": 1100.0}],
            },
        )
        self._store_activity_summary(
            missing_id,
            self.user_id,
            {"activity": {"total_distance_m": 5000.0}, "runs_summary": {"count": 2}},
        )

        self._set_user(self.user_id)
        overview = self.client.get("/experiences?scope=mine")
        self.assertEqual(overview.status_code, 200)
        self.assertIn(b"Distance on foil", overview.data)
        self.assertIn(b"Avg run distance", overview.data)
        self.assertIn(b"sort=activity_foil_distance_m", overview.data)
        self.assertIn(b"sort=activity_avg_run_distance_m", overview.data)
        self.assertIn(b"742 m", overview.data)
        self.assertIn(b"236 m", overview.data)
        self.assertIn(b"2.2 km", overview.data)
        self.assertIn(b"1.2 km", overview.data)

        missing_link = f'href="/experiences/{missing_id}"'.encode()
        missing_position = overview.data.index(missing_link)
        row_start = overview.data.rfind(b"<tr", 0, missing_position)
        row_end = overview.data.find(b"</tr>", missing_position)
        self.assertIn(b">n/a</td>", overview.data[row_start:row_end])

        by_foil_desc = self.client.get("/experiences?scope=mine&sort=activity_foil_distance_m&dir=desc")
        self.assertEqual(by_foil_desc.status_code, 200)
        self.assertLess(
            by_foil_desc.data.index(f'href="/experiences/{large_id}"'.encode()),
            by_foil_desc.data.index(f'href="/experiences/{small_id}"'.encode()),
        )
        self.assertLess(
            by_foil_desc.data.index(f'href="/experiences/{small_id}"'.encode()),
            by_foil_desc.data.index(f'href="/experiences/{missing_id}"'.encode()),
        )

        by_avg_asc = self.client.get("/experiences?scope=mine&sort=activity_avg_run_distance_m&dir=asc")
        self.assertEqual(by_avg_asc.status_code, 200)
        self.assertLess(
            by_avg_asc.data.index(f'href="/experiences/{small_id}"'.encode()),
            by_avg_asc.data.index(f'href="/experiences/{large_id}"'.encode()),
        )
        self.assertLess(
            by_avg_asc.data.index(f'href="/experiences/{large_id}"'.encode()),
            by_avg_asc.data.index(f'href="/experiences/{missing_id}"'.encode()),
        )

    def test_submission_scopes_and_detail_access_control(self) -> None:
        self._set_profile(self.user_id, "Zulu Rider", "Owner Private Name")
        self._set_profile(self.other_user_id, "Alpha Rider", "Other Private Name")
        own_private = self._create_submission(self.user_id, "Owner Private", "2026-02-01", "private")
        own_public = self._create_submission(
            self.user_id,
            "Owner Public",
            "2026-02-02",
            "public",
            {"wind_variability": 1.8, "point_count": 3},
            perceived_wind_variability="gusty",
        )
        other_private = self._create_submission(self.other_user_id, "Other Private", "2026-02-03", "private")
        other_public = self._create_submission(
            self.other_user_id,
            "Other Public",
            "2026-02-04",
            "public",
            {"wind_variability": 2.4, "point_count": 3},
            perceived_wind_variability="steady",
        )

        self._set_user(self.user_id)
        mine = self.client.get("/experiences?scope=mine")
        self.assertEqual(mine.status_code, 200)
        self.assertIn(f'href="/experiences/{own_private}"'.encode(), mine.data)
        self.assertIn(f'href="/experiences/{own_public}"'.encode(), mine.data)
        self.assertNotIn(f'href="/experiences/{other_private}"'.encode(), mine.data)
        self.assertNotIn(f'href="/experiences/{other_public}"'.encode(), mine.data)
        self.assertIn(b"Zulu Rider", mine.data)
        self.assertIn(b">01-02-2026</a>", mine.data)
        self.assertIn(b"data-sort=\"2026-02-01\"", mine.data)

        self.assertIn(b'class="segmented-toggle-segment active" aria-current="page" href="/experiences?scope=mine">My</a>', mine.data)
        self.assertIn(b'class="segmented-toggle-segment" href="/experiences?scope=all">All</a>', mine.data)
        self.assertEqual(mine.data.count(b'aria-current="page"'), 1)

        mine_by_rider = self.client.get("/experiences?scope=mine&sort=rider&dir=asc")
        self.assertEqual(mine_by_rider.status_code, 200)
        self.assertIn(b"sort=rider", mine_by_rider.data)

        all_submissions = self.client.get("/experiences?scope=all")
        self.assertEqual(all_submissions.status_code, 200)
        self.assertIn(f'href="/experiences/{own_private}"'.encode(), all_submissions.data)
        self.assertIn(f'href="/experiences/{own_public}"'.encode(), all_submissions.data)
        self.assertNotIn(f'href="/experiences/{other_private}"'.encode(), all_submissions.data)
        self.assertIn(f'href="/experiences/{other_public}"'.encode(), all_submissions.data)
        self.assertIn(b"Alpha Rider", all_submissions.data)
        self.assertIn(b">04-02-2026</a>", all_submissions.data)
        self.assertIn(b"data-sort=\"2026-02-04\"", all_submissions.data)
        self.assertNotIn(b"other-rider", all_submissions.data)
        self.assertNotIn(b"Other Public", all_submissions.data)
        self.assertNotIn(b"<h2>All submissions</h2>", all_submissions.data)

        self.assertIn(b'class="segmented-toggle-segment" href="/experiences?scope=mine">My</a>', all_submissions.data)
        self.assertIn(b'class="segmented-toggle-segment active" aria-current="page" href="/experiences?scope=all">All</a>', all_submissions.data)
        self.assertEqual(all_submissions.data.count(b'aria-current="page"'), 1)

        all_by_visibility = self.client.get("/experiences?scope=all&sort=visibility&dir=asc")
        self.assertEqual(all_by_visibility.status_code, 200)
        self.assertLess(
            all_by_visibility.data.index(f'href="/experiences/{own_private}"'.encode()),
            all_by_visibility.data.index(f'href="/experiences/{own_public}"'.encode()),
        )
        all_by_rider = self.client.get("/experiences?scope=all&sort=rider&dir=asc")
        self.assertEqual(all_by_rider.status_code, 200)
        self.assertLess(
            all_by_rider.data.index(f'href="/experiences/{other_public}"'.encode()),
            all_by_rider.data.index(f'href="/experiences/{own_private}"'.encode()),
        )
        all_by_variability = self.client.get("/experiences?scope=all&sort=wind_variability&dir=asc")
        self.assertEqual(all_by_variability.status_code, 200)
        self.assertIn(b"sort=wind_variability", all_by_variability.data)
        self.assertIn(b"Measured variability", all_submissions.data)
        self.assertNotIn(b">Variability</a>", all_submissions.data)
        self.assertLess(all_submissions.data.index(b"Perceived variability"), all_submissions.data.index(b"Measured variability"))
        self.assertIn(b"Steady", all_submissions.data)
        self.assertIn(b"Gusty", all_submissions.data)
        all_by_perceived = self.client.get("/experiences?scope=all&sort=perceived_wind_variability&dir=asc")
        self.assertEqual(all_by_perceived.status_code, 200)
        self.assertIn(b"sort=perceived_wind_variability", all_by_perceived.data)
        self.assertLess(
            all_by_perceived.data.index(f'href="/experiences/{other_public}"'.encode()),
            all_by_perceived.data.index(f'href="/experiences/{own_public}"'.encode()),
        )

        conn = db_store.connect_db(self.temp_dir.name)
        other_public_row = next(
            row for row in db_store.list_surf_experiences(conn, self.user_id, scope="all") if row["id"] == other_public
        )
        other_public_detail_row = db_store.get_visible_surf_experience(conn, self.user_id, other_public)
        conn.close()
        self.assertIsNone(other_public_row["rider"])
        self.assertEqual(other_public_row["submitted_by"], "Alpha Rider")
        self.assertEqual(other_public_row["rider_notes"], "")
        self.assertEqual(other_public_row["perceived_wind_variability"], "steady")
        self.assertIsNotNone(other_public_detail_row)
        self.assertIsNone(other_public_detail_row["rider"])
        self.assertEqual(other_public_detail_row["submitted_by"], "Alpha Rider")
        self.assertIsNone(other_public_detail_row["rider_weight"])
        self.assertEqual(other_public_detail_row["rider_notes"], "")

        owner_private_detail = self.client.get(f"/experiences/{own_private}")
        self.assertEqual(owner_private_detail.status_code, 200)
        self.assertIn(b"Owner Private", owner_private_detail.data)
        self.assertIn(b"<dt>Private RiderNotes</dt><dd>Private notes by Owner Private</dd>", owner_private_detail.data)
        self.assertIn(b"Make this submission public to share it.", owner_private_detail.data)
        self.assertNotIn(b'class="share-icon-button"', owner_private_detail.data)
        self.assertNotIn(b'class="primary share-button"', owner_private_detail.data)
        owner_public_detail = self.client.get(f"/experiences/{own_public}")
        self.assertEqual(owner_public_detail.status_code, 200)
        self.assertIn(b"<dt>Private RiderNotes</dt><dd>Private notes by Owner Public</dd>", owner_public_detail.data)
        self.assertIn(b'class="share-icon-button"', owner_public_detail.data)
        self.assertIn(b'aria-label="Share public submission"', owner_public_detail.data)
        self.assertIn(f'action="/experiences/{own_public}/share"'.encode(), owner_public_detail.data)
        self.assertIn(b'data-share-url=""', owner_public_detail.data)
        self.assertNotIn(f"/share/experience/{own_public}".encode(), owner_public_detail.data)
        self.assertNotIn(f'value="http://localhost/experiences/{own_public}"'.encode(), owner_public_detail.data)

        other_private_detail = self.client.get(f"/experiences/{other_private}")
        self.assertEqual(other_private_detail.status_code, 404)
        other_public_detail = self.client.get(f"/experiences/{other_public}")
        self.assertEqual(other_public_detail.status_code, 200)
        self.assertIn(b"<dt>Submitted by</dt><dd>Alpha Rider</dd>", other_public_detail.data)
        self.assertIn(b"<dt>Perceived variability</dt><dd>Steady</dd>", other_public_detail.data)
        self.assertNotIn(b'class="share-icon-button"', other_public_detail.data)
        self.assertNotIn(f'action="/experiences/{other_public}/share"'.encode(), other_public_detail.data)
        self.assertNotIn(b"other-rider", other_public_detail.data)
        self.assertNotIn(b"<dt>Rider</dt><dd>Other Public</dd>", other_public_detail.data)
        self.assertNotIn(b"<dt>RiderWeight</dt>", other_public_detail.data)
        self.assertNotIn(b"Private RiderNotes", other_public_detail.data)
        self.assertNotIn(b"Private notes by Other Public", other_public_detail.data)
        self.assertNotIn(b"Modify", other_public_detail.data)

        other_public_edit = self.client.get(f"/experiences/{other_public}/edit")
        self.assertEqual(other_public_edit.status_code, 404)

        self.assertEqual(self.client.get(f"/share/experience/{other_public}").status_code, 404)
        conn = db_store.connect_db(self.temp_dir.name)
        other_public_token = db_store.create_or_get_surf_experience_share_token(conn, other_public, self.other_user_id)
        conn.close()
        self.assertIsNotNone(other_public_token)
        self.assertNotEqual(other_public_token, str(other_public))
        self.assertGreaterEqual(len(other_public_token), 22)

        self._set_user(None)
        public_share = self.client.get(f"/share/experience/{other_public_token}")
        self.assertEqual(public_share.status_code, 200)
        self.assertIn(b"Public session", public_share.data)
        self.assertIn(b"<dt>Submitted by</dt><dd>Alpha Rider</dd>", public_share.data)
        self.assertIn(b"<dt>Spot</dt><dd>Valkenburgse meer</dd>", public_share.data)
        self.assertIn(b"<dt>RiderReview</dt><dd>Review by Other Public</dd>", public_share.data)
        self.assertIn(b"<dt>Perceived variability</dt><dd>Steady</dd>", public_share.data)
        self.assertIn(b'aria-label="4 out of 5"', public_share.data)
        self.assertNotIn(b"<dt>SessionRating</dt><dd>4/5</dd>", public_share.data)
        self.assertIn(b"Measured wind", public_share.data)
        self.assertNotIn(b"other-rider", public_share.data)
        self.assertNotIn(b"<dt>Rider</dt>", public_share.data)
        self.assertNotIn(b"RiderWeight", public_share.data)
        self.assertNotIn(b"Private RiderNotes", public_share.data)
        self.assertNotIn(b"Private notes by Other Public", public_share.data)
        self.assertNotIn(b"Modify", public_share.data)
        self.assertNotIn(b"Delete", public_share.data)
        self.assertNotIn(b"_csrf_token", public_share.data)
        self.assertNotIn(b"Profile", public_share.data)
        self.assertNotIn(b"Login", public_share.data)

        self.assertEqual(self.client.get(f"/share/experience/{own_private}").status_code, 404)
        self.assertEqual(self.client.get(f"/share/experience/{other_private}").status_code, 404)

        logged_out_detail = self.client.get(f"/experiences/{other_public}")
        self.assertEqual(logged_out_detail.status_code, 302)
        self.assertIn("login=1", logged_out_detail.headers["Location"])

    def test_owner_generates_unguessable_share_token_for_public_submission(self) -> None:
        self._set_profile(self.user_id, "Share Owner", "Private Owner Name")
        public_id = self._create_submission(self.user_id, "Shared Public", "2026-02-06", "public")
        private_id = self._create_submission(self.user_id, "Shared Private", "2026-02-07", "private")
        other_public_id = self._create_submission(self.other_user_id, "Other Shared Public", "2026-02-08", "public")

        self.assertEqual(self.client.get(f"/share/experience/{public_id}").status_code, 404)

        self._set_user(self.user_id)
        self.assertEqual(self.client.get(f"/experiences/{public_id}").status_code, 200)
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        response = self.client.post(
            f"/experiences/{public_id}/share",
            data={"_csrf_token": csrf_token},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        token = payload["share_token"]
        self.assertTrue(token)
        self.assertNotEqual(token, str(public_id))
        self.assertGreaterEqual(len(token), 22)
        self.assertIn(f"/share/experience/{token}", payload["share_url"])

        detail = self.client.get(f"/experiences/{public_id}?shared=1")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(f'value="http://localhost/share/experience/{token}"'.encode(), detail.data)
        self.assertNotIn(f"/share/experience/{public_id}".encode(), detail.data)

        anonymous_share = self.client.get(f"/share/experience/{token}")
        self.assertEqual(anonymous_share.status_code, 200)
        self.assertIn(b"Public session", anonymous_share.data)
        self.assertNotIn(b"Private notes by Shared Public", anonymous_share.data)
        self.assertNotIn(b"RiderWeight", anonymous_share.data)
        self.assertNotIn(b"test-rider", anonymous_share.data)
        self.assertNotIn(b"_csrf_token", anonymous_share.data)
        self.assertNotIn(b"Profile", anonymous_share.data)
        self.assertEqual(self.client.get(f"/share/experience/{public_id}").status_code, 404)

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        private_response = self.client.post(f"/experiences/{private_id}/share", data={"_csrf_token": csrf_token})
        self.assertEqual(private_response.status_code, 404)

        self._set_user(self.other_user_id)
        self.assertEqual(self.client.get(f"/experiences/{public_id}").status_code, 200)
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        other_response = self.client.post(f"/experiences/{public_id}/share", data={"_csrf_token": csrf_token})
        self.assertEqual(other_response.status_code, 404)
        conn = db_store.connect_db(self.temp_dir.name)
        self.assertEqual(conn.execute("SELECT share_token FROM surf_experiences WHERE id = ?", (public_id,)).fetchone()[0], token)
        self.assertIsNone(conn.execute("SELECT share_token FROM surf_experiences WHERE id = ?", (other_public_id,)).fetchone()[0])
        conn.execute("UPDATE surf_experiences SET visibility = 'private' WHERE id = ?", (public_id,))
        conn.commit()
        conn.close()

        self._set_user(None)
        self.assertEqual(self.client.get(f"/share/experience/{token}").status_code, 404)

    def test_public_submission_without_public_username_uses_private_fallback(self) -> None:
        no_name_user_id: int
        conn = db_store.connect_db(self.temp_dir.name)
        no_name_user_id = db_store.create_user(conn, "private.login@example.com", portal._hash_password("test-password"))
        conn.close()
        unnamed_public = self._create_submission(
            no_name_user_id,
            "Secret Freeform Name",
            "2026-02-05",
            "public",
            rider_review="Public review without private name",
        )

        self._set_user(self.user_id)
        overview = self.client.get("/experiences?scope=all&sort=rider&dir=asc")
        self.assertEqual(overview.status_code, 200)
        self.assertIn(f'href="/experiences/{unnamed_public}"'.encode(), overview.data)
        self.assertIn(b"Unknown rider", overview.data)
        self.assertNotIn(b"private.login@example.com", overview.data)
        self.assertNotIn(b"Secret Freeform Name", overview.data)

        detail = self.client.get(f"/experiences/{unnamed_public}")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"<dt>Submitted by</dt><dd>Unknown rider</dd>", detail.data)
        self.assertNotIn(b"private.login@example.com", detail.data)

        conn = db_store.connect_db(self.temp_dir.name)
        share_token = db_store.create_or_get_surf_experience_share_token(conn, unnamed_public, no_name_user_id)
        conn.close()
        self.assertIsNotNone(share_token)

        self._set_user(None)
        public_share = self.client.get(f"/share/experience/{share_token}")
        self.assertEqual(public_share.status_code, 200)
        self.assertIn(b"<dt>Submitted by</dt><dd>Unknown rider</dd>", public_share.data)
        self.assertIn(b"Public review without private name", public_share.data)
        self.assertNotIn(b"private.login@example.com", public_share.data)
        self.assertNotIn(b"Secret Freeform Name", public_share.data)

    def test_activity_upload_requires_authenticated_owner(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-01", "private")

        response = self.client.post(
            f"/experiences/{experience_id}/activity-upload",
            data={"ActivityFile": self._activity_file()},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("login=1", response.headers["Location"])

        self._set_user(self.other_user_id)
        self.client.get("/experience/new")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        other_response = self.client.post(
            f"/experiences/{experience_id}/activity-upload",
            data={"_csrf_token": csrf_token, "ActivityFile": self._activity_file()},
        )
        self.assertEqual(other_response.status_code, 404)

    def test_invalid_activity_extension_is_rejected(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-02", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{experience_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        with patch.object(portal, "analyze_session_file") as mocked_analysis:
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._activity_file("bad.txt")},
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unsupported activity file. Please upload a FIT, TCX, GPX, KML, or ZIP file.", response.data)
        mocked_analysis.assert_not_called()
        conn = db_store.connect_db(self.temp_dir.name)
        self.assertIsNone(db_store.get_surf_experience_activity_analysis(conn, experience_id, self.user_id))
        conn.close()

    def test_tcx_activity_upload_is_accepted(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-02", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{experience_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        with patch.object(portal, "analyze_session_file", side_effect=self._mock_analysis_payload) as mocked_analysis:
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": (io.BytesIO(self._tcx_bytes()), "session.tcx")},
            )

        self.assertEqual(response.status_code, 302)
        mocked_analysis.assert_called_once()
        conn = db_store.connect_db(self.temp_dir.name)
        analysis = db_store.get_surf_experience_activity_analysis(conn, experience_id, self.user_id)
        conn.close()
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["file_type"], "tcx")

    def test_zip_activity_upload_with_one_supported_file_is_accepted(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-02", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{experience_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        gpx_bytes = self._activity_file()[0].getvalue()
        with patch.object(portal, "analyze_session_file", side_effect=self._mock_analysis_payload) as mocked_analysis:
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._zip_activity_file({"session.gpx": gpx_bytes})},
            )

        self.assertEqual(response.status_code, 302)
        mocked_analysis.assert_called_once()
        conn = db_store.connect_db(self.temp_dir.name)
        analysis = db_store.get_surf_experience_activity_analysis(conn, experience_id, self.user_id)
        conn.close()
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["file_type"], "zip")

    def test_zip_activity_upload_rejects_no_supported_file(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-02", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{experience_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        with patch.object(portal, "analyze_session_file") as mocked_analysis:
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._zip_activity_file({"notes.txt": b"not activity"})},
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ZIP uploads must contain exactly one supported activity file.", response.data)
        mocked_analysis.assert_not_called()

    def test_zip_activity_upload_rejects_multiple_supported_files(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-02", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{experience_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        gpx_bytes = self._activity_file()[0].getvalue()
        with patch.object(portal, "analyze_session_file") as mocked_analysis:
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._zip_activity_file({"a.gpx": gpx_bytes, "b.tcx": self._tcx_bytes()})},
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ZIP uploads must contain exactly one supported activity file.", response.data)
        mocked_analysis.assert_not_called()

    def test_zip_activity_upload_rejects_path_traversal(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-02", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{experience_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        with patch.object(portal, "analyze_session_file") as mocked_analysis:
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._zip_activity_file({"../session.gpx": self._activity_file()[0].getvalue()})},
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ZIP upload rejected because it contains unsafe paths or is too large when extracted.", response.data)
        mocked_analysis.assert_not_called()

    def test_activity_upload_above_twenty_mb_is_rejected(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-02", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{experience_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        with patch.object(portal, "_uploaded_stream_size", return_value=portal.MAX_ACTIVITY_UPLOAD_BYTES + 1), patch.object(portal, "analyze_session_file") as mocked_analysis:
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._activity_file("large.fit")},
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Activity file is too large. Please upload a file up to 20 MB.", response.data)
        mocked_analysis.assert_not_called()

    def test_existing_submission_activity_upload_persists_and_renders_artifacts(self) -> None:
        experience_id = self._create_submission(
            self.user_id,
            "Owner",
            "2026-04-03",
            "private",
            {"avg_wind_speed": 16.0, "mean_wind_dir": 225.0, "point_count": 4},
        )
        self._set_user(self.user_id)
        self.client.get("/experience/new")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        with patch.object(portal, "analyze_session_file", side_effect=self._mock_analysis_payload) as mocked_analysis:
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._activity_file("my-session.gpx")},
            )

        self.assertEqual(response.status_code, 302)
        mocked_analysis.assert_called_once()
        wind_context = mocked_analysis.call_args.kwargs["wind_context"]
        self.assertEqual(wind_context.spot_name, "Valkenburgse meer")
        self.assertEqual(wind_context.wind_speed_kts, 16.0)
        self.assertEqual(wind_context.wind_direction_deg, 225.0)

        conn = db_store.connect_db(self.temp_dir.name)
        analysis = db_store.get_surf_experience_activity_analysis(conn, experience_id, self.user_id)
        conn.close()
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["status"], "ok")
        self.assertEqual(analysis["original_filename"], "my-session.gpx")
        self.assertEqual(analysis["stats"]["run_count"], 13)
        self.assertEqual(analysis["artifacts"]["map_html"], "map.html")
        self.assertIn("timestamps are irregular", analysis["warnings"][0])

        detail = self.client.get(f"/experiences/{experience_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"Activity analysis", detail.data)
        self.assertIn(b"my-session.gpx", detail.data)
        self.assertIn(b"GPX", detail.data)
        self.assertIn(b"OK", detail.data)
        self.assertIn(b"Upload file", detail.data)
        self.assertIn(b"Upload a FIT, TCX, GPX, KML, or ZIP activity file.", detail.data)
        self.assertNotIn(b"<label for=\"ActivityFile\">Activity file</label>", detail.data)
        self.assertNotIn(b"Optional FIT, GPX or KML file from your session.", detail.data)
        self.assertNotIn(b"Current file:", detail.data)
        self.assertIn(b"GPS timestamps were somewhat irregular; speed and distance were reconstructed where needed.", detail.data)
        self.assertNotIn(b"median interval is 0.70s", detail.data)
        self.assertNotIn(b'class="activity-summary-stats"', detail.data)
        self.assertNotIn(b'class="activity-summary-stat"', detail.data)
        self.assertNotIn(b"Distance on foil", detail.data)
        self.assertNotIn(b"Avg run distance", detail.data)
        self.assertNotIn(b"Water time", detail.data)
        self.assertNotIn(b"20m 0s", detail.data)
        self.assertIn(b'class="activity-analysis-frame"', detail.data)
        self.assertIn(f"/experiences/{experience_id}/activity-artifact/map.html".encode(), detail.data)
        self.assertIn(b"Distance (m)", detail.data)
        self.assertIn(b"651 m", detail.data)
        self.assertIn(b"663 m", detail.data)
        self.assertIn(b"17.5 km/h", detail.data)
        self.assertIn(b"24.0 km/h", detail.data)
        self.assertNotIn(b"17.54 km/h", detail.data)
        self.assertNotIn(b"24.04 km/h", detail.data)
        self.assertNotIn(b"0.651 km", detail.data)
        self.assertIn(b'class="run-select-checkbox"', detail.data)
        self.assertIn(b'class="run-select-cell"', detail.data)
        self.assertIn(b'class="run-distance-cell"', detail.data)
        self.assertIn(b'data-run-id="13"', detail.data)
        self.assertIn(b'class="activity-sort-button"', detail.data)
        self.assertIn(b'data-sort-type="number"', detail.data)
        self.assertIn(b"localeCompare", detail.data)
        self.assertIn(b"wingfoil-run-selection", detail.data)
        self.assertEqual(detail.data.count(b">crosswind</td>"), 13)

        artifact = self.client.get(f"/experiences/{experience_id}/activity-artifact/map.svg")
        self.assertEqual(artifact.status_code, 200)
        self.assertIn(b"<svg", artifact.data)
        artifact.close()
        self.assertEqual(self.client.get(f"/experiences/{experience_id}/activity-artifact/../summary.json").status_code, 404)

    def test_optional_activity_upload_during_new_submission(self) -> None:
        self._set_user(self.user_id)
        self.client.get("/experience/new")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        form = self._valid_form("private")
        form["_csrf_token"] = csrf_token
        form["ActivityFile"] = self._activity_file("new-session.kml")

        with patch.object(portal, "analyze_session_file", side_effect=self._mock_analysis_payload):
            response = self.client.post("/experience/new", data=form)

        self.assertEqual(response.status_code, 302)
        conn = db_store.connect_db(self.temp_dir.name)
        rows = db_store.list_surf_experiences(conn, self.user_id)
        experience_id = rows[0]["id"]
        analysis = db_store.get_surf_experience_activity_analysis(conn, experience_id, self.user_id)
        conn.close()
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["file_type"], "kml")
        self.assertEqual(analysis["status"], "ok")

    def test_failed_activity_analysis_is_stored_without_crashing(self) -> None:
        experience_id = self._create_submission(self.user_id, "Owner", "2026-04-04", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{experience_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]

        def failed_analysis(*_args, **_kwargs):
            return {
                "status": "error",
                "analysis_version": "test-version",
                "input_type": "gpx",
                "error": "Could not parse activity file",
                "warnings": ["bad track"],
            }

        with patch.object(portal, "analyze_session_file", side_effect=failed_analysis):
            response = self.client.post(
                f"/experiences/{experience_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._activity_file("corrupt.gpx")},
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Could not parse activity file", response.data)
        conn = db_store.connect_db(self.temp_dir.name)
        analysis = db_store.get_surf_experience_activity_analysis(conn, experience_id, self.user_id)
        conn.close()
        self.assertEqual(analysis["status"], "error")
        self.assertEqual(analysis["errors"], ["Could not parse activity file"])
        self.assertIn("bad track", analysis["warnings"])

    def test_public_activity_analysis_renders_read_only_and_authorizes_artifacts(self) -> None:
        public_id = self._create_submission(self.user_id, "Owner", "2026-04-05", "public")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{public_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        with patch.object(portal, "analyze_session_file", side_effect=self._mock_analysis_payload):
            self.client.post(
                f"/experiences/{public_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._activity_file("public-session.gpx")},
            )

        conn = db_store.connect_db(self.temp_dir.name)
        analysis = db_store.get_surf_experience_activity_analysis(conn, public_id, self.user_id)
        token = db_store.create_or_get_surf_experience_share_token(conn, public_id, self.user_id)
        conn.close()
        self.assertIsNotNone(analysis)

        self._set_user(self.other_user_id)
        public_detail = self.client.get(f"/experiences/{public_id}")
        self.assertEqual(public_detail.status_code, 200)
        self.assertIn(b"Activity analysis", public_detail.data)
        self.assertIn(b'class="activity-analysis-frame"', public_detail.data)
        self.assertIn(f"/experiences/{public_id}/activity-artifact/map.html".encode(), public_detail.data)
        self.assertIn(b"Distance (m)", public_detail.data)
        self.assertIn(b"651 m", public_detail.data)
        self.assertIn(b"663 m", public_detail.data)
        self.assertIn(b"17.5 km/h", public_detail.data)
        self.assertIn(b"24.0 km/h", public_detail.data)
        self.assertNotIn(b"17.54 km/h", public_detail.data)
        self.assertIn(b'class="run-select-checkbox"', public_detail.data)
        self.assertIn(b'data-run-id="13"', public_detail.data)
        self.assertIn(b'class="activity-sort-button"', public_detail.data)
        self.assertIn(b'data-sort-value="663"', public_detail.data)
        self.assertIn(b"wingfoil-run-selection", public_detail.data)
        self.assertEqual(public_detail.data.count(b">crosswind</td>"), 13)
        self.assertIn(b"Run distance distribution", public_detail.data)
        self.assertIn(b"Run speed distribution", public_detail.data)
        self.assertIn(b"Run speed profile", public_detail.data)
        self.assertIn(b"GPS timestamps were somewhat irregular; speed and distance were reconstructed where needed.", public_detail.data)
        self.assertNotIn(b"median interval is 0.70s", public_detail.data)
        self.assertNotIn(b"Upload file", public_detail.data)
        self.assertNotIn(b"Re-upload", public_detail.data)
        self.assertNotIn(b"public-session.gpx", public_detail.data)
        self.assertNotIn(str(analysis["stored_filename"]).encode(), public_detail.data)

        artifact = self.client.get(f"/experiences/{public_id}/activity-artifact/map.svg")
        self.assertEqual(artifact.status_code, 200)
        self.assertIn(b"<svg", artifact.data)
        artifact.close()
        self.assertEqual(self.client.get(f"/experiences/{public_id}/activity-artifact/../summary.json").status_code, 404)
        self.assertEqual(self.client.get(f"/experiences/{public_id}/activity-artifact/{analysis['stored_filename']}").status_code, 404)

        self._set_user(None)
        public_share = self.client.get(f"/share/experience/{token}")
        self.assertEqual(public_share.status_code, 200)
        self.assertIn(b"Activity analysis", public_share.data)
        self.assertIn(b'class="activity-analysis-frame"', public_share.data)
        self.assertIn(f"/share/experience/{token}/activity-artifact/map.html".encode(), public_share.data)
        self.assertIn(b"Distance (m)", public_share.data)
        self.assertIn(b"663 m", public_share.data)
        self.assertIn(b"17.5 km/h", public_share.data)
        self.assertIn(b"24.0 km/h", public_share.data)
        self.assertNotIn(b"17.54 km/h", public_share.data)
        self.assertIn(b'class="run-select-checkbox"', public_share.data)
        self.assertIn(b'data-run-id="13"', public_share.data)
        self.assertIn(b'class="activity-sort-button"', public_share.data)
        self.assertIn(b"wingfoil-run-selection", public_share.data)
        self.assertEqual(public_share.data.count(b">crosswind</td>"), 13)
        self.assertIn(b"Run distance distribution", public_share.data)
        self.assertIn(b"Run speed distribution", public_share.data)
        self.assertIn(b"Run speed profile", public_share.data)
        self.assertNotIn(b"Upload file", public_share.data)
        self.assertNotIn(b"Re-upload", public_share.data)
        self.assertNotIn(b"public-session.gpx", public_share.data)
        self.assertNotIn(str(analysis["stored_filename"]).encode(), public_share.data)

        shared_map = self.client.get(f"/share/experience/{token}/activity-artifact/map.html")
        self.assertEqual(shared_map.status_code, 200)
        self.assertIn(b"map", shared_map.data)
        self.assertIn(b"applyWingfoilRunSelection", shared_map.data)
        self.assertIn(b"falls-toggle", shared_map.data)
        shared_map.close()
        shared_distance_plot = self.client.get(f"/share/experience/{token}/activity-artifact/run_distance_distribution.svg")
        self.assertEqual(shared_distance_plot.status_code, 200)
        self.assertIn(b"0-100 m", shared_distance_plot.data)
        shared_distance_plot.close()
        self.assertEqual(self.client.get(f"/share/experience/{token}/activity-artifact/../summary.json").status_code, 404)
        self.assertEqual(self.client.get(f"/share/experience/{token}/activity-artifact/{analysis['stored_filename']}").status_code, 404)
        self.assertEqual(self.client.get("/share/experience/not-a-real-token/activity-artifact/map.svg").status_code, 404)

    def test_private_activity_analysis_remains_hidden_from_public_and_other_users(self) -> None:
        private_id = self._create_submission(self.user_id, "Owner", "2026-04-06", "private")
        self._set_user(self.user_id)
        self.client.get(f"/experiences/{private_id}")
        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        with patch.object(portal, "analyze_session_file", side_effect=self._mock_analysis_payload):
            self.client.post(
                f"/experiences/{private_id}/activity-upload",
                data={"_csrf_token": csrf_token, "ActivityFile": self._activity_file("private-session.gpx")},
            )
        conn = db_store.connect_db(self.temp_dir.name)
        analysis = db_store.get_surf_experience_activity_analysis(conn, private_id, self.user_id)
        conn.close()
        self.assertIsNotNone(analysis)

        self._set_user(self.other_user_id)
        self.assertEqual(self.client.get(f"/experiences/{private_id}").status_code, 404)
        self.assertEqual(self.client.get(f"/experiences/{private_id}/activity-artifact/map.svg").status_code, 404)

        self._set_user(None)
        self.assertEqual(self.client.get(f"/share/experience/{private_id}").status_code, 404)
        self.assertEqual(self.client.get(f"/share/experience/{private_id}/activity-artifact/map.svg").status_code, 404)

    def test_native_run_direction_arrow_targets_follow_spacing_rules(self) -> None:
        cases = {
            42: [21.0],
            50: [25.0],
            74: [25.0],
            120: [25.0, 75.0],
            226: [25.0, 75.0, 125.0, 175.0],
            230: [25.0, 75.0, 125.0, 175.0],
            260: [25.0, 75.0, 125.0, 175.0, 225.0],
        }
        for distance_m, expected_targets in cases.items():
            with self.subTest(distance_m=distance_m):
                self.assertEqual(wingfoil_pipeline.run_direction_arrow_targets(distance_m), expected_targets)
        self.assertEqual(
            wingfoil_pipeline.run_direction_arrow_candidate_targets(260),
            [25.0, 50.0, 75.0, 125.0, 150.0, 175.0, 200.0, 225.0],
        )

    def test_native_run_direction_arrows_use_run_geometry_and_run_id(self) -> None:
        base_time = datetime(2026, 1, 20, 11, 0, tzinfo=timezone.utc)

        def make_samples(segment_distances: list[float]) -> list[wingfoil_pipeline.Sample]:
            return [
                wingfoil_pipeline.Sample(
                    index=index,
                    lat=52.0,
                    lon=4.0 + index * 0.0001,
                    ele_m=None,
                    time=base_time + timedelta(seconds=index),
                    dt_s=0.0 if index == 0 else 1.0,
                    segment_distance_m=segment_distance,
                    speed_mps=8.0 if index else 0.0,
                    smooth_speed_mps=8.0 if index else 0.0,
                    in_run=True,
                )
                for index, segment_distance in enumerate(segment_distances)
            ]

        short_samples = make_samples([0.0, 10.0, 10.0, 10.0, 12.0])
        short_run = wingfoil_pipeline.build_run(3, short_samples, 0, len(short_samples) - 1, min_run_duration_s=1)
        self.assertIsNotNone(short_run)
        short_arrows = wingfoil_pipeline.run_direction_arrows(short_samples, [short_run])
        self.assertEqual([arrow["target_distance_m"] for arrow in short_arrows], [21.0])
        self.assertEqual(short_arrows[0]["run_id"], 3)
        self.assertEqual(short_arrows[0]["run_distance_m"], 42.0)
        self.assertAlmostEqual(short_arrows[0]["bearing_deg"], 90.0, delta=1.0)

        long_samples = make_samples([0.0] + [10.0] * 26)
        long_run = wingfoil_pipeline.build_run(9, long_samples, 0, len(long_samples) - 1, min_run_duration_s=1)
        self.assertIsNotNone(long_run)
        long_arrows = wingfoil_pipeline.run_direction_arrows(long_samples, [long_run])
        self.assertEqual([arrow["target_distance_m"] for arrow in long_arrows], [25.0, 50.0, 75.0, 125.0, 150.0, 175.0, 200.0, 225.0])
        self.assertEqual({arrow["run_id"] for arrow in long_arrows}, {9})
        self.assertTrue(all(89.0 <= arrow["bearing_deg"] <= 91.0 for arrow in long_arrows))

    def test_tcx_parser_loads_namespaced_trackpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tcx_path = Path(temp_dir) / "session.tcx"
            tcx_path.write_bytes(self._tcx_bytes())

            df = wingfoil_pipeline.load_tcx_activity(tcx_path)
            samples, warnings = wingfoil_pipeline.load_activity(tcx_path, smooth_window=1)

        self.assertEqual(len(df), 2)
        self.assertEqual(int(df.iloc[0]["heart_rate_bpm"]), 101)
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[1].speed_mps, 6.5)
        self.assertAlmostEqual(samples[1].segment_distance_m, 65.0)
        self.assertEqual(warnings, [])

    def test_zip_parser_extracts_single_supported_activity_privately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "session.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("session.tcx", self._tcx_bytes())

            samples, warnings = wingfoil_pipeline.load_activity(zip_path, smooth_window=1)

        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[1].speed_mps, 6.5)
        self.assertEqual(warnings, [])

    def test_zip_parser_rejects_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "session.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("../session.tcx", self._tcx_bytes())

            with self.assertRaisesRegex(wingfoil_pipeline.AnalysisError, "unsafe paths"):
                wingfoil_pipeline.load_activity(zip_path, smooth_window=1)

    def test_generated_activity_artifacts_do_not_include_raw_source_filename(self) -> None:
        sensitive_name = "20260401T120000Z_private_raw_upload.tcx"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tcx_path = temp_path / sensitive_name
            output_dir = temp_path / "out"
            tcx_path.write_bytes(self._tcx_bytes())

            result = wingfoil_pipeline.analyze_activity(tcx_path)
            wingfoil_pipeline.write_analysis_outputs(result, output_dir)

            summary_text = (output_dir / "summary.json").read_text(encoding="utf-8")
            map_text = (output_dir / "map.html").read_text(encoding="utf-8")

        self.assertNotIn(sensitive_name, summary_text)
        self.assertNotIn(sensitive_name, map_text)
        self.assertIn('"run_detection_source": "native"', summary_text)
        self.assertIn('"run_arrows":', map_text)
        self.assertNotIn("leaflet-providers", map_text)
        self.assertNotIn("Stadia", map_text)
        self.assertNotIn("stadiamaps", map_text)
        self.assertIn("server.arcgisonline.com/ArcGIS/rest/services/World_Imagery", map_text)
        self.assertIn("tile.openstreetmap.org", map_text)
        self.assertIn("basemaps.cartocdn.com/light_all", map_text)
        self.assertIn("run-direction-arrow-svg", map_text)
        self.assertIn('viewBox="0 0 28 16"', map_text)
        self.assertIn('d="M2 8 H20 M14 2 L22 8 L14 14"', map_text)
        self.assertIn("arrowZoomStyle", map_text)
        self.assertIn("refreshDirectionArrows", map_text)
        self.assertIn("visibleArrowCount", map_text)
        self.assertIn("run_distance_m", map_text)


if __name__ == "__main__":
    unittest.main()
