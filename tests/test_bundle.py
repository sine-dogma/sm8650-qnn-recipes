"""Verify exported license scope and public metadata. SPDX-License-Identifier: CC0-1.0."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sm8650_qnn import bundle
from sm8650_qnn.prepared import write
from sm8650_qnn.settings import PROJECT


class BundleTest(unittest.TestCase):
    def test_model_license_is_retained_and_private_source_paths_are_removed(self):
        parent = PROJECT / "work" / "unit-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary, ExitStack() as stack:
            root = Path(temporary).resolve()
            model = copy.deepcopy(bundle.MODEL)
            model.update(repo_id="source/fixture", revision="pinned", expected_contexts=1, expected_graphs=1)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "LICENSE").write_text("Original external model license\n", encoding="utf-8")
            write(snapshot / "config.json", {"text_config": {"model_type": "fixture"}})
            write(snapshot / "tokenizer_config.json", {"model_max_length": 2048})
            source = root / "model-source.json"
            write(source, {"repo_id": model["repo_id"], "revision": model["revision"], "snapshot_path": str(snapshot)})
            upstream = root / "third_party/ai-hub-models/src"
            upstream.mkdir(parents=True)
            (upstream / "LICENSE").write_text("External BSD license\n", encoding="utf-8")
            base = root / "prepared"
            write(base / "calibration-provenance.json", {"dataset": "fixture", "raw_examples_in_public_package": False})
            for precision in bundle.VARIANTS:
                directory = root / "variants" / precision
                (directory / "context").mkdir(parents=True)
                binary = directory / "context/decoder.bin"
                binary.write_bytes(b"generated test context")
                write(directory / "native-artifact-report.json", {
                    "complete": True, "contexts": {"decoder": {"file": str(binary), "bytes": binary.stat().st_size,
                        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}},
                    "precision": {"decoder_s1": {}}, "total_context_bytes": binary.stat().st_size,
                })
            for name, value in {"PROJECT": root, "BASE": base, "MODEL": model, "SOURCE_RECORD": source,
                                "variant_dir": lambda precision: root / "variants" / precision}.items():
                stack.enter_context(patch.object(bundle, name, value))
            destination = root / "output"
            bundle.package(destination)
            self.assertEqual((destination / "LICENSE").read_text(), "Original external model license\n")
            self.assertIn("license: apache-2.0", (destination / "README.md").read_text())
            for path in destination.rglob("*.json"):
                text = path.read_text().replace("\\\\", "/")
                self.assertNotIn(root.as_posix(), text)
            for line in (destination / "SHA256SUMS").read_text().splitlines():
                expected, relative = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((destination / relative).read_bytes()).hexdigest(), expected)
            self.assertFalse(destination.with_name(destination.name + ".pending").exists())


if __name__ == "__main__":
    unittest.main()
