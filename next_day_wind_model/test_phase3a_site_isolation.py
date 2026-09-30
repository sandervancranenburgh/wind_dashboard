from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from next_day_wind_model.migrate_site_artifacts import _assert_disjoint
from next_day_wind_model.site_model_orchestrator import build_site_command, enabled_model_site_ids
from next_day_wind_model.site_paths import (
    CHECKPOINT_SCHEMA_VERSION,
    build_artifact_manifest,
    resolve_site_paths,
    validate_artifact_manifest,
    validate_checkpoint_identity,
    write_artifact_manifest,
)
from next_day_wind_model.update_model_and_predict import _site_publication_allowed


class Phase3ASiteIsolationTests(unittest.TestCase):
    def test_site_paths_route_default_and_block_unpublished_site(self) -> None:
        root = Path("isolated-artifacts")
        web = Path("isolated-web")
        valkenburg = resolve_site_paths("valkenburgsemeer", artifact_root=root, web_root=web)
        oostvoorne = resolve_site_paths("oostvoorne", artifact_root=root, web_root=web)
        self.assertEqual(valkenburg.artifact_dir, root / "valkenburgsemeer")
        self.assertEqual(valkenburg.web_dir, web)
        self.assertEqual(oostvoorne.artifact_dir, root / "oostvoorne")
        self.assertIsNone(oostvoorne.web_dir)
        self.assertTrue(_site_publication_allowed("valkenburgsemeer"))
        self.assertFalse(_site_publication_allowed("oostvoorne"))

    def test_orchestrator_excludes_and_rejects_disabled_site(self) -> None:
        self.assertEqual(enabled_model_site_ids(), ("valkenburgsemeer",))
        with self.assertRaisesRegex(ValueError, "disabled"):
            build_site_command(
                "oostvoorne",
                db_path=Path("backup.db"),
                artifact_root=Path("artifacts"),
                web_root=Path("web"),
            )

    def test_manifest_checksum_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"verified")
            manifest = build_artifact_manifest(
                artifact_dir=root,
                site_id="oostvoorne",
                forecast_model="HARMONIE",
                status="experimental",
                production_eligible=False,
                files=[artifact],
            )
            write_artifact_manifest(root, manifest)
            validate_artifact_manifest(
                root,
                expected_site_id="oostvoorne",
                expected_forecast_model="HARMONIE",
            )
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_artifact_manifest(
                    root,
                    expected_site_id="oostvoorne",
                    expected_forecast_model="HARMONIE",
                )

    def test_checkpoint_site_mismatch_is_rejected(self) -> None:
        checkpoint = {
            "artifact_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "site_id": "valkenburgsemeer",
            "forecast_model": "HARMONIE",
        }
        with self.assertRaisesRegex(ValueError, "site mismatch"):
            validate_checkpoint_identity(
                checkpoint,
                expected_site_id="oostvoorne",
                expected_forecast_model="HARMONIE",
            )

    def test_partially_identified_checkpoint_is_not_treated_as_legacy(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks"):
            validate_checkpoint_identity(
                {"site_id": "valkenburgsemeer"},
                expected_site_id="valkenburgsemeer",
                expected_forecast_model="HARMONIE",
                allow_legacy_identity=True,
            )

    def test_migration_rejects_overlapping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "disjoint"):
                _assert_disjoint(source, source / "nested")


if __name__ == "__main__":
    unittest.main()
