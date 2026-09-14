from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from patent_coating.patent_rag_app import PATH_KEYS, load_config


class LoadConfigTests(unittest.TestCase):
    def test_resolves_every_relative_path_against_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configuration"
            other_cwd = root / "elsewhere"
            config_dir.mkdir()
            other_cwd.mkdir()
            (config_dir / "secrets").mkdir()
            (config_dir / "secrets" / ".env").write_text("", encoding="utf-8")
            config_path = config_dir / "rag.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    paths:
                      mineru_root: corpus/mineru
                      parsed_dir: generated/parsed
                      index_dir: generated/index
                      workspace_root: generated/workspaces
                      env_file: secrets/.env
                    models: {}
                    parsing: {}
                    index: {}
                    retrieval:
                      candidate_patents: 12
                      final_patents: 5
                    agent: {}
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            previous_cwd = Path.cwd()
            try:
                os.chdir(other_cwd)
                config = load_config(config_path)
            finally:
                os.chdir(previous_cwd)

            expected = {
                "mineru_root": config_dir / "corpus" / "mineru",
                "parsed_dir": config_dir / "generated" / "parsed",
                "index_dir": config_dir / "generated" / "index",
                "workspace_root": config_dir / "generated" / "workspaces",
                "env_file": config_dir / "secrets" / ".env",
            }
            self.assertEqual(set(expected), set(PATH_KEYS))
            for key, expected_path in expected.items():
                self.assertEqual(Path(config["paths"][key]), expected_path.resolve())
            self.assertEqual(Path(config["_config_path"]), config_path.resolve())


if __name__ == "__main__":
    unittest.main()
