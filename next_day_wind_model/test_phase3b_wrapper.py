from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER = REPO_ROOT / "scripts" / "run_wind_dashboard_pipeline.sh"


class Phase3BWrapperTests(unittest.TestCase):
    def test_wrapper_uses_site_isolated_artifacts(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('SITE_ID="valkenburgsemeer"', text)
        self.assertIn('SITE_ARTIFACT_DIR="${LEGACY_ARTIFACT_DIR}/${SITE_ID}"', text)
        self.assertIn('--out-dir "${MODEL_ARTIFACT_DIR}"', text)
        self.assertIn('--model-artifact-dir "${MODEL_ARTIFACT_DIR}"', text)

    def test_wrapper_has_explicit_legacy_rollback_switch(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('WIND_USE_LEGACY_MODEL_ARTIFACTS', text)
        self.assertIn('MODEL_ARTIFACT_DIR="${LEGACY_ARTIFACT_DIR}"', text)

    def test_wrapper_does_not_enable_oostvoorne(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn('--site oostvoorne', text)


if __name__ == "__main__":
    unittest.main()
