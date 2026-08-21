from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ecospec_kg.cli import main
from ecospec_kg.io_utils import read_json, read_jsonl


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "mini_chunks.jsonl"


class CliIntegrationTests(unittest.TestCase):
    def test_module_entrypoint_runs_train_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "training.jsonl"
            output = root / "run"
            prepared.write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ecospec_kg.cli",
                    "train",
                    "--prepared",
                    str(prepared),
                    "--out",
                    str(output),
                    "--model",
                    "unused-in-preflight",
                    "--trainer",
                    "swift",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_json(output / "run_manifest.json")["status"], "prepared")

    def test_fixture_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            annotations = root / "annotations.csv"
            index_dir = root / "index"
            ablations = root / "ablations.json"
            self.assertEqual(
                main(["predict", "--chunks", str(FIXTURE), "--out", str(predictions)]),
                0,
            )
            self.assertTrue(read_jsonl(predictions))
            self.assertEqual(
                main(
                    [
                        "annotate",
                        "--chunks",
                        str(FIXTURE),
                        "--relations",
                        str(predictions),
                        "--out",
                        str(annotations),
                    ]
                ),
                0,
            )
            self.assertTrue(annotations.exists())
            self.assertEqual(
                main(
                    [
                        "index",
                        "--adapter",
                        "native",
                        "--chunks",
                        str(FIXTURE),
                        "--relations",
                        str(predictions),
                        "--out",
                        str(index_dir),
                    ]
                ),
                0,
            )
            self.assertTrue((index_dir / "index.json").exists())
            self.assertEqual(main(["ablate", "--out", str(ablations)]), 0)
            self.assertEqual(read_json(ablations)["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
