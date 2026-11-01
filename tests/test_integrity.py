"""Regression tests for stale outputs, native input data and locks. SPDX-License-Identifier: CC0-1.0."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sm8650_qnn import convert_local, local_tools, prepared
from sm8650_qnn import calibration_xlam
from sm8650_qnn.locking import workflow_lock
from sm8650_qnn.settings import PROJECT


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        parent = PROJECT / "work" / "unit-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

    def substitute(self, module, **values):
        for name, value in values.items():
            self.stack.enter_context(patch.object(module, name, value))


class ConversionCommitTest(WorkspaceTest):
    def test_success_without_new_output_cannot_promote_an_old_dlc(self):
        self.substitute(local_tools, ROOT=self.root)
        source = self.root / "float"
        source.mkdir()
        (source / "model.onnx").write_bytes(b"first source graph")
        graph = {"validated": True, "directory": "float", "inputs": {}, "outputs": []}
        destination = self.root / "dlc"
        def sdk_success(command, *args):
            Path(command[command.index("--output_path")+1]).write_bytes(b"complete DLC")
        with patch.object(local_tools, "_run", sdk_success):
            output = local_tools.local_convert("decoder_s1", graph, destination, {})
        old_marker = (destination / "decoder_s1.convert.json").read_bytes()
        (source / "model.onnx").write_bytes(b"second source graph")
        with patch.object(local_tools, "_run", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "not produced"):
                local_tools.local_convert("decoder_s1", graph, destination, {})
        self.assertEqual(output.read_bytes(), b"complete DLC")
        self.assertEqual((destination / "decoder_s1.convert.json").read_bytes(), old_marker)

    def test_same_size_output_tampering_invalidates_reuse(self):
        output = self.root / "graph.dlc"
        marker = self.root / "marker.json"
        output.write_bytes(b"valid")
        signature = {"graph": "example"}
        marker.write_text(json.dumps({"status": "COMPLETE", "signature": signature, "bytes": 5, "sha256": local_tools.digest(output)}))
        self.assertTrue(local_tools._reusable(marker, signature, output))
        output.write_bytes(b"wrong")
        self.assertFalse(local_tools._reusable(marker, signature, output))


class CalibrationIntegrityTest(WorkspaceTest):
    def setUp(self):
        super().setUp()
        self.base = self.root / "prepared"
        self.base.mkdir()
        (self.base / "calibration-provenance.json").write_text("{}")
        self.source = self.base / "captures"
        self.source.mkdir()
        self.graph = {"calibration_directory": "prepared/captures", "inputs": {"hidden_states": {"shape": [1, 2], "dtype": "float32"}}}
        self.values = np.array([[1.0, 2.0]], dtype=np.float32)
        np.savez_compressed(self.source / "000.npz", hidden_states=self.values)
        self.substitute(convert_local, ROOT=self.root, BASE=self.base, BATCH=self.root / "local", SAMPLES=1)

    def test_same_size_raw_corruption_is_repaired_from_original_capture(self):
        listing, _ = convert_local.calibration("decoder_s1", self.graph)
        raw = Path(listing.read_text().strip().split(":=", 1)[1])
        raw.write_bytes(b"\x00" * self.values.nbytes)
        convert_local.calibration("decoder_s1", self.graph)
        np.testing.assert_array_equal(np.fromfile(raw, dtype=np.float32).reshape(1, 2), self.values)

    def test_changed_capture_with_preserved_timestamp_cannot_reuse_staged_data(self):
        convert_local.calibration("decoder_s1", self.graph)
        capture = self.source / "000.npz"
        previous = capture.stat()
        np.savez_compressed(capture, hidden_states=self.values + 1)
        os.utime(capture, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with self.assertRaisesRegex(RuntimeError, "source calibration changed"):
            convert_local.calibration("decoder_s1", self.graph)


class PathAndLockTest(WorkspaceTest):
    def test_manifest_rejects_windows_paths_and_parent_segments(self):
        self.substitute(prepared, ROOT=self.root)
        for value in ("../outside", "/outside", "Z:/outside", "Z:outside", "prepared\\graph"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepared.work_path(value)

    def test_concurrent_writer_is_rejected_and_lock_is_released(self):
        path = self.root / "workflow.lock"
        with workflow_lock(path):
            with self.assertRaises(OSError):
                with workflow_lock(path):
                    self.fail("A second writer acquired the workflow lock")
        with workflow_lock(path):
            pass


class SelectionAndWorkerTest(WorkspaceTest):
    def test_changed_selection_with_same_count_is_rejected(self):
        selected = self.root / "selection.json"
        self.substitute(calibration_xlam, OUT=self.root, SELECTED=selected, COUNT=2)
        prepared.write(selected, [{"tokens": [1, 2]}, {"tokens": [3, 4]}])
        prepared.write(self.root / "calibration-provenance.json", {
            "seed": calibration_xlam.SEED, "selection_identity": calibration_xlam.selection_identity(),
            "selected_examples_sha256": prepared.digest(selected),
        })
        calibration_xlam.select()
        prepared.write(selected, [{"tokens": [1, 9]}, {"tokens": [3, 4]}])
        with self.assertRaisesRegex(RuntimeError, "selection changed"):
            calibration_xlam.select()

    def test_worker_network_guard_blocks_connection_before_network_io(self):
        env = os.environ.copy()
        env["SM8650_OFFLINE"] = "1"
        env["PYTHONPATH"] = str(PROJECT / "src")
        child = subprocess.run([sys.executable, "-c", "import sm8650_qnn,socket; socket.socket().connect(('127.0.0.1',1))"],
                               capture_output=True, text=True, env=env)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("Network disabled for local conversion", child.stderr)

    @unittest.skipUnless(os.name == "posix", "Linux venv symlink regression")
    def test_sdk_interpreter_retains_venv_symlink_location(self):
        directory = self.root / "venv" / "bin"
        directory.mkdir(parents=True)
        interpreter = directory / "python"
        interpreter.symlink_to(sys.executable)
        env = os.environ.copy()
        env["QAIRT_PYTHON"] = str(interpreter)
        env["PYTHONPATH"] = str(PROJECT / "src")
        child = subprocess.run([sys.executable, "-c", "from sm8650_qnn.settings import SDK_PYTHON; print(SDK_PYTHON)"],
                               capture_output=True, text=True, env=env, check=True)
        self.assertEqual(child.stdout.strip(), str(interpreter.absolute()))


if __name__ == "__main__":
    unittest.main()
