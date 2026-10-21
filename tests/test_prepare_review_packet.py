import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/prepare_review_packet_20261021.py"
spec = importlib.util.spec_from_file_location("review_packet", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReviewPacketTests(unittest.TestCase):
    def make_source(self, root):
        (root / "packet/sources").mkdir(parents=True)
        (root / "protocol").mkdir()
        sources = []
        for code in ("A", "B"):
            path = root / f"packet/sources/{code}.pdf"
            path.write_bytes(b"%PDF-test-" + code.encode())
            sources.append({"id": code, "file": f"sources/{code}.pdf",
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        meta = {"source_count": 2, "sources": sources,
                "gold_available": False, "inference_ready": False,
                "project_exposure_human_confirmation": "pending"}
        (root / "packet/source_manifest.json").write_text(json.dumps(meta), encoding="utf-8")
        self.refresh_lock(root)

    def refresh_lock(self, root):
        files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((root / "packet").rglob("*")) if p.is_file()}
        (root / "protocol/packet_lock.json").write_text(json.dumps({"files": files}), encoding="utf-8")

    def test_copy_preserves_bytes_and_does_not_mark_gold_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.make_source(root)
            out = Path(tmp) / "review"
            result = module.prepare(root, out)
            self.assertEqual(result["source_count"], 2)
            self.assertIs(result["gold_available"], False)
            self.assertIs(result["inference_ready"], False)
            self.assertIs(result["model_calls"], 0)
            for rel in json.loads((root / "protocol/packet_lock.json").read_text())["files"]:
                dest = out / Path(rel).relative_to("packet")
                self.assertEqual(dest.read_bytes(), (root / rel).read_bytes())
            self.assertEqual(result["required_next_action"], "human_blind_annotation_and_review")

    def test_tampered_source_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.make_source(root)
            (root / "packet/sources/A.pdf").write_bytes(b"%PDF-changed")
            out = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.prepare(root, out)
            self.assertFalse(out.exists())

    def test_manifest_path_cannot_escape_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.make_source(root)
            lock = root / "protocol/packet_lock.json"
            data = json.loads(lock.read_text())
            data["files"]["packet/../outside"] = "0" * 64
            lock.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "path"):
                module.prepare(root, Path(tmp) / "out")

    def test_unlisted_payload_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.make_source(root)
            (root / "packet/prediction.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory"):
                module.prepare(root, Path(tmp) / "out")

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.make_source(root)
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                module.prepare(root, out)
            self.assertEqual((out / "keep.txt").read_text(), "keep")

    def test_declared_readiness_and_mismatched_source_hash_are_rejected(self):
        for change in ("ready", "hash"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "repo"
                self.make_source(root)
                path = root / "packet/source_manifest.json"
                meta = json.loads(path.read_text())
                if change == "ready":
                    meta["gold_available"] = True
                else:
                    meta["sources"][0]["sha256"] = "0" * 64
                path.write_text(json.dumps(meta))
                self.refresh_lock(root)
                with self.assertRaises(ValueError):
                    module.prepare(root, Path(tmp) / "out")

    def test_real_cli_marks_only_packet_integrity_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.make_source(root)
            run = subprocess.run([sys.executable, str(TOOL), "--root", str(root),
                                  "--out", str(Path(tmp) / "out")], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("BLIND_PACKET_OK=true", run.stdout)
            self.assertIn("INFERENCE_READY=false", run.stdout)
            self.assertIn("MODEL_CALLS=0", run.stdout)
            self.assertNotIn("RESULTS_UPLOAD_OK", run.stdout)


if __name__ == "__main__":
    unittest.main()
