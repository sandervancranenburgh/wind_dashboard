from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from next_day_wind_model import update_model_and_predict


class SourceRefreshInterpreterTests(unittest.TestCase):
    def test_fetch_uses_the_updaters_selected_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            out_data_dir = repo_root / "data"
            with mock.patch.object(update_model_and_predict.subprocess, "run") as run:
                update_model_and_predict._run_fetch_script(repo_root, out_data_dir)

        run.assert_called_once_with(
            [sys.executable, str(repo_root / "source_fetch.py"), str(out_data_dir)],
            cwd=str(repo_root),
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
