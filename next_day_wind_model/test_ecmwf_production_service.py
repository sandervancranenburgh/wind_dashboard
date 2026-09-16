from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from next_day_wind_model.shadow_config import load_shadow_config


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "manage_ecmwf_service.py"
SPEC = importlib.util.spec_from_file_location("ecmwf_production_service", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


class ProductionServiceTests(unittest.TestCase):
    def test_rendered_service_uses_dedicated_production_paths_and_hardening(self) -> None:
        units = service.render_units()
        rendered = units[service.SERVICE_NAME]
        self.assertIn(str(service.REPO_ROOT), rendered)
        self.assertIn("/.venv-ecmwf/bin/python", rendered)
        self.assertIn("/data/ecmwf_archive", rendered)
        self.assertIn("/config/ecmwf_production.json", rendered)
        self.assertIn("ProtectSystem=strict", rendered)
        self.assertIn("ProtectHome=read-only", rendered)
        self.assertIn("StandardOutput=journal", rendered)
        self.assertNotIn("--write-production", rendered)
        self.assertNotIn("@REPO_ROOT@", rendered)

    def test_timer_is_persistent_and_has_unique_production_name(self) -> None:
        units = service.render_units()
        timer = units[service.TIMER_NAME]
        self.assertEqual(service.TIMER_NAME, "wind-fetcher2-ecmwf.timer")
        self.assertIn("OnCalendar=*:0/15", timer)
        self.assertIn("Persistent=yes", timer)
        self.assertIn(service.SERVICE_NAME, timer)
        self.assertNotIn("shadow-dev", timer)

    def test_non_main_branch_is_rejected_for_production_installation(self) -> None:
        with mock.patch.object(service, "_git_output", return_value="development"):
            with self.assertRaisesRegex(service.ServiceSafetyError, "production branch"):
                service.validate_primary_checkout()

    def test_linked_worktree_is_rejected_for_production_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").write_text("gitdir: /tmp/fixture", encoding="utf-8")
            with (
                mock.patch.object(service, "REPO_ROOT", root),
                mock.patch.object(service, "_git_output", return_value="main"),
            ):
                with self.assertRaisesRegex(service.ServiceSafetyError, "linked worktree"):
                    service.validate_primary_checkout()

    def test_archive_and_environment_cannot_be_redirected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_path = root / "python"
            config_path = root / "config.json"
            python_path.touch()
            config_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(service.ServiceSafetyError):
                service.validate_runtime_paths(
                    python_path,
                    root / "archive",
                    config_path,
                )

    def test_production_config_is_continuous(self) -> None:
        config = load_shadow_config(service.DEFAULT_CONFIG)
        self.assertIsNone(config.experiment_end_utc)
        self.assertFalse(config.expired(datetime.now(timezone.utc)))
        self.assertIsNone(config.max_completed_runs_per_site)


if __name__ == "__main__":
    unittest.main()
