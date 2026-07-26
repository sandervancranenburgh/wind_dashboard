import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_rider_portal_preview.sh"
CORE_PATHS = (
    "data/wind_data_all_sites.db",
    "data/rider_activity_uploads",
    "data/rider_activity_analysis",
    "next_day_wind_model/artifacts/current_day_plot_archive",
)
SECRET_PATH = "data/.wind_dashboard_secret"
ALL_PATHS = (CORE_PATHS[0], SECRET_PATH, *CORE_PATHS[1:])


class SetupRiderPortalPreviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.preview = self.root / "preview"
        self.production = self.root / "production"
        self.home = self.root / "home"

        self._init_repo(self.preview)
        self._init_repo(self.production)
        self.home.mkdir()
        self._create_production_data(self.production)

    @staticmethod
    def _init_repo(path: Path, *, main_branch: bool = False) -> None:
        path.mkdir()
        command = ["git", "init", "-q"]
        if main_branch:
            command.extend(["-b", "main"])
        command.append(str(path))
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def _configure_git_identity(path: Path) -> None:
        subprocess.run(["git", "-C", str(path), "config", "user.name", "Preview Test"], check=True)
        subprocess.run(
            ["git", "-C", str(path), "config", "user.email", "preview@example.invalid"],
            check=True,
        )

    @staticmethod
    def _create_production_data(path: Path) -> None:
        (path / "data").mkdir(exist_ok=True)
        (path / "data" / "wind_data_all_sites.db").write_bytes(b"production database")
        (path / "data" / ".wind_dashboard_secret").write_text(
            "production secret", encoding="utf-8"
        )
        (path / "data" / "rider_activity_uploads").mkdir(exist_ok=True)
        (path / "data" / "rider_activity_uploads" / "upload.fit").write_bytes(b"upload")
        (path / "data" / "rider_activity_analysis").mkdir(exist_ok=True)
        (path / "data" / "rider_activity_analysis" / "analysis.json").write_text(
            "{}", encoding="utf-8"
        )
        archive = path / "next_day_wind_model" / "artifacts" / "current_day_plot_archive"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "plot.png").write_bytes(b"plot")

    def _run(self, *arguments: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["HOME"] = str(self.home)
        return subprocess.run(
            [str(SCRIPT), *arguments],
            cwd=cwd or self.preview,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def _assert_links_configured(
        self,
        paths: tuple[str, ...] = CORE_PATHS,
        *,
        preview: Path | None = None,
        production: Path | None = None,
    ) -> None:
        preview_root = preview or self.preview
        production_root = production or self.production
        for relative_path in paths:
            destination = preview_root / relative_path
            source = production_root / relative_path
            self.assertTrue(destination.is_symlink(), relative_path)
            self.assertEqual(destination.resolve(), source.resolve())

    def test_help_and_default_production_path(self) -> None:
        help_result = self._run("--help")

        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--link-secret", help_result.stdout)
        self.assertIn("--undo", help_result.stdout)
        self.assertIn("--dry-run", help_result.stdout)
        self.assertIn("--force", help_result.stdout)

        default_production = self.home / "Documents" / "repos" / "wind_fetcher2"
        default_production.parent.mkdir(parents=True)
        self.production.rename(default_production)
        self.production = default_production

        default_result = self._run()

        self.assertEqual(default_result.returncode, 0, default_result.stderr)
        self.assertIn("default fallback", default_result.stdout)
        self._assert_links_configured()
        self.assertFalse((self.preview / SECRET_PATH).exists())

    def test_auto_detects_the_single_other_main_worktree(self) -> None:
        main_checkout = self.root / "auto-main"
        preview_worktree = self.root / "auto-preview"
        self._init_repo(main_checkout, main_branch=True)
        self._configure_git_identity(main_checkout)
        (main_checkout / "tracked.txt").write_text("fixture", encoding="utf-8")
        subprocess.run(["git", "-C", str(main_checkout), "add", "tracked.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(main_checkout), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(main_checkout),
                "worktree",
                "add",
                "-q",
                "-b",
                "preview-validation",
                str(preview_worktree),
            ],
            check=True,
        )
        self._create_production_data(main_checkout)

        result = self._run(cwd=preview_worktree)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("auto-detected main worktree", result.stdout)
        self.assertIn(str(main_checkout), result.stdout)
        self._assert_links_configured(preview=preview_worktree, production=main_checkout)

    def test_configures_from_nested_directory_and_is_idempotent_without_secret(self) -> None:
        nested = self.preview / "some" / "nested" / "directory"
        nested.mkdir(parents=True)

        first = self._run(str(self.production), cwd=nested)

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn(f"Preview worktree: {self.preview}", first.stdout)
        self.assertIn("Configuration complete: 4 link(s) updated", first.stdout)
        self.assertIn("Using a different Flask secret is harmless", first.stdout)
        self._assert_links_configured()
        self.assertFalse((self.preview / SECRET_PATH).exists())

        second = self._run(str(self.production), cwd=nested)

        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn(
            "Already configured: all 4 selected Rider Portal preview links are correct.",
            second.stdout,
        )
        self._assert_links_configured()

    def test_secret_is_opt_in_and_undo_restores_tracked_secret(self) -> None:
        preview_secret = self.preview / SECRET_PATH
        preview_secret.parent.mkdir()
        preview_secret.write_text("preview secret", encoding="utf-8")
        self._configure_git_identity(self.preview)
        subprocess.run(["git", "-C", str(self.preview), "add", SECRET_PATH], check=True)
        subprocess.run(
            ["git", "-C", str(self.preview), "commit", "-q", "-m", "fixture secret"],
            check=True,
        )

        default_result = self._run(str(self.production))

        self.assertEqual(default_result.returncode, 0, default_result.stderr)
        self.assertFalse(preview_secret.is_symlink())
        self.assertEqual(preview_secret.read_text(encoding="utf-8"), "preview secret")
        self._assert_links_configured()

        without_force = self._run("--link-secret", str(self.production))

        self.assertNotEqual(without_force.returncode, 0)
        self.assertIn("without --force", without_force.stderr)
        self.assertFalse(preview_secret.is_symlink())

        with_force = self._run("--link-secret", "--force", str(self.production))

        self.assertEqual(with_force.returncode, 0, with_force.stderr)
        self._assert_links_configured(ALL_PATHS)

        undo_dry_run = self._run("--undo", "--dry-run")

        self.assertEqual(undo_dry_run.returncode, 0, undo_dry_run.stderr)
        self.assertIn("Dry-run undo summary", undo_dry_run.stdout)
        self._assert_links_configured(ALL_PATHS)

        undo = self._run("--undo")

        self.assertEqual(undo.returncode, 0, undo.stderr)
        self.assertIn("Restored tracked path with git restore", undo.stdout)
        self.assertFalse(preview_secret.is_symlink())
        self.assertEqual(preview_secret.read_text(encoding="utf-8"), "preview secret")
        for relative_path in CORE_PATHS:
            self.assertFalse((self.preview / relative_path).is_symlink())
            self.assertFalse((self.preview / relative_path).exists())
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(self.preview), "status", "--porcelain"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout,
            "",
        )

        second_undo = self._run("--undo")

        self.assertEqual(second_undo.returncode, 0, second_undo.stderr)
        self.assertIn("no setup-managed Rider Portal links were found", second_undo.stdout)

    def test_undo_leaves_preexisting_correct_symlink_untouched(self) -> None:
        database = self.preview / CORE_PATHS[0]
        database.parent.mkdir()
        database.symlink_to(self.production / CORE_PATHS[0])

        setup = self._run(str(self.production))
        self.assertEqual(setup.returncode, 0, setup.stderr)

        undo = self._run("--undo")

        self.assertEqual(undo.returncode, 0, undo.stderr)
        self.assertTrue(database.is_symlink())
        self.assertEqual(database.resolve(), (self.production / CORE_PATHS[0]).resolve())
        for relative_path in CORE_PATHS[1:]:
            self.assertFalse((self.preview / relative_path).exists())

    def test_refuses_to_configure_production_repository_itself(self) -> None:
        result = self._run(str(self.production), cwd=self.production)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to configure the production repository itself", result.stderr)
        self.assertEqual(
            (self.production / "data" / "wind_data_all_sites.db").read_bytes(),
            b"production database",
        )
        self.assertFalse((self.production / "data" / "wind_data_all_sites.db").is_symlink())

    def test_refuses_missing_repository_or_required_source(self) -> None:
        missing_repository = self._run(str(self.root / "missing-production"))
        self.assertNotEqual(missing_repository.returncode, 0)
        self.assertIn("Production repository does not exist", missing_repository.stderr)

        (self.production / "data" / "wind_data_all_sites.db").unlink()
        missing_source = self._run(str(self.production))
        self.assertNotEqual(missing_source.returncode, 0)
        self.assertIn("required production file is missing", missing_source.stderr)
        for relative_path in CORE_PATHS:
            self.assertFalse((self.preview / relative_path).exists())
            self.assertFalse((self.preview / relative_path).is_symlink())

    def test_dry_run_and_force_protect_existing_preview_data(self) -> None:
        preview_data = self.preview / "data"
        preview_data.mkdir()
        database = preview_data / "wind_data_all_sites.db"
        database.write_bytes(b"preview database")
        uploads = preview_data / "rider_activity_uploads"
        uploads.mkdir()
        (uploads / "keep.fit").write_bytes(b"keep")

        dry_run = self._run("--dry-run", str(self.production))

        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("Existing non-symlinks requiring --force: 2", dry_run.stdout)
        self.assertEqual(database.read_bytes(), b"preview database")
        self.assertEqual((uploads / "keep.fit").read_bytes(), b"keep")
        self.assertFalse((preview_data / "rider_activity_analysis").exists())

        without_force = self._run(str(self.production))

        self.assertNotEqual(without_force.returncode, 0)
        self.assertIn("without --force", without_force.stderr)
        self.assertIn("No changes were made", without_force.stderr)
        self.assertEqual(database.read_bytes(), b"preview database")
        self.assertEqual((uploads / "keep.fit").read_bytes(), b"keep")
        self.assertFalse((preview_data / "rider_activity_analysis").exists())

        with_force = self._run("--force", str(self.production))

        self.assertEqual(with_force.returncode, 0, with_force.stderr)
        self._assert_links_configured()
        self.assertEqual(
            (self.production / "data" / "wind_data_all_sites.db").read_bytes(),
            b"production database",
        )
        self.assertEqual(
            (self.production / "data" / "rider_activity_uploads" / "upload.fit").read_bytes(),
            b"upload",
        )

    def test_replaces_only_incorrect_symlinks_without_force(self) -> None:
        wrong_target = self.root / "wrong-database"
        wrong_target.write_bytes(b"wrong")
        (self.preview / "data").mkdir()
        database = self.preview / "data" / "wind_data_all_sites.db"
        database.symlink_to(wrong_target)

        result = self._run(str(self.production))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Will replace incorrect symlink", result.stdout)
        self._assert_links_configured()
        self.assertEqual(wrong_target.read_bytes(), b"wrong")

    def test_refuses_symlinked_preview_parent_directory(self) -> None:
        (self.preview / "data").symlink_to(self.production / "data", target_is_directory=True)

        result = self._run("--force", str(self.production))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to use symlinked preview parent directory", result.stderr)
        self.assertEqual(
            (self.production / "data" / "wind_data_all_sites.db").read_bytes(),
            b"production database",
        )


if __name__ == "__main__":
    unittest.main()
