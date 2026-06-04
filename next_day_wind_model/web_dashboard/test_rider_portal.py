from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import db_store
from next_day_wind_model.web_dashboard import app as portal


class RiderPortalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        portal.DATA_DIR = Path(self.temp_dir.name)
        portal.app.config.update(TESTING=True, SECRET_KEY="rider-portal-test-secret")

        conn = db_store.connect_db(self.temp_dir.name)
        db_store.init_db(conn)
        db_store.create_user(conn, "test-rider", portal._hash_password("test-password"))
        conn.close()
        self.client = portal.app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_my_sessions_login_flow_and_account_indicator(self) -> None:
        protected = self.client.get("/experiences")
        self.assertEqual(protected.status_code, 302)
        self.assertEqual(protected.headers["Location"], "/?login=1&next=/experiences")

        login_page = self.client.get(protected.headers["Location"])
        self.assertEqual(login_page.status_code, 200)
        self.assertIn(b'id="open-login">Login</button>', login_page.data)
        self.assertIn(b'name="next" value="/experiences"', login_page.data)

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
        self.assertIn(b'class="page-header-actions"', sessions_page.data)

        root_handoff = self.client.get("/?next=/experience/new")
        self.assertEqual(root_handoff.status_code, 302)
        self.assertEqual(root_handoff.headers["Location"], "/experience/new")

        new_submission = self.client.get("/experience/new")
        self.assertEqual(new_submission.status_code, 200)
        self.assertIn(b"<h2>New submission</h2>", new_submission.data)
        self.assertEqual(new_submission.data.count(b'href="/experiences"'), 1)
        self.assertEqual(new_submission.data.count(b'href="/experience/new"'), 0)

        with self.client.session_transaction() as current_session:
            csrf_token = current_session["_csrf_token"]
        logout_response = self.client.post("/logout", data={"_csrf_token": csrf_token})
        self.assertEqual(logout_response.status_code, 302)
        self.assertEqual(logout_response.headers["Location"], "/")

    def test_measured_report_min_max_trend_and_variability(self) -> None:
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
        self.assertGreater(measured["summary"]["wind_variability"], 4.0)
        self.assertEqual(
            measured["summary"]["wind_variability_kind"],
            "mean_15min_rolling_sample_standard_deviation",
        )
        sparse_conn = db_store.connect_db(self.temp_dir.name)
        sparse = db_store.get_measured_wind_for_session(
            sparse_conn,
            "Valkenburgse meer",
            start_ms,
            start_ms + 3 * 60 * 1000,
        )
        sparse_conn.close()
        self.assertIsNone(sparse["summary"]["wind_variability"])
        self.assertTrue(all("measured_wind_min" in record for record in measured["plot_records"]))
        self.assertTrue(all("measured_wind_max" in record for record in measured["plot_records"]))

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
        self.assertTrue(plot["trend_points"])

    def test_submission_detail_labels_units_and_primary_navigation_only(self) -> None:
        row = {
            "date": "2026-01-15",
            "spot": "Valkenburgse meer",
            "start_time": "12:00",
            "end_time": "13:00",
            "avg_forecast_temperature": 10.0,
            "session_rating": 4,
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
        }
        with portal.app.test_request_context("/experiences/1"):
            portal.session["user_id"] = 1
            detail = portal.render_template(
                "submission_detail.html",
                row=row,
                wind_plot={"available": False},
                current_day_archive_plot=None,
            ).encode()

        for label in (b"avg speed", b"max avg speed", b"min avg speed", b"max gust", b"wind variability", b"avg direction"):
            self.assertIn(label, detail)
        for value in (b"14.4 kts", b"20.0 kts", b"9.0 kts", b"30.0 kts", b"1.8 kts", b"SW (208 deg)"):
            self.assertIn(value, detail)
        self.assertNotIn(b'class="form-actions"', detail)
        self.assertEqual(detail.count(b'href="/experiences"'), 1)
        self.assertEqual(detail.count(b'href="/experience/new"'), 1)


if __name__ == "__main__":
    unittest.main()
