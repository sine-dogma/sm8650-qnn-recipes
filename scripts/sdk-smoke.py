#!/usr/bin/env python3
"""Exercise every recipe on a generated graph, without a model download.

Requires the SDK and requirements/sdk.txt plus the SDK's Python dependencies.
This checks conversion and native metadata, not quantized phone inference.
SPDX-License-Identifier: CC0-1.0
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--sdk-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--work-dir", type=Path, default=PROJECT / "work" / "sdk-smoke")
    args = parser.parse_args()
    root = args.work_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh smoke-test work directory")
    root.mkdir(parents=True, exist_ok=True)
    model = json.loads((PROJECT / "models" / "qwen3.5-0.8b.json").read_text(encoding="utf-8"))
    model.update(id="sdk-smoke", repo_id="local/normalization-smoke", revision="synthetic-v1", license="cc0-1.0", adapter="fixture",
                 expected_graphs=1, expected_contexts=1, pilot_part="decoder")
    model["calibration"].update(profile="synthetic-smoke", samples=16, repo_id="local/synthetic",
                                revision="synthetic-v1", filename="generated", sha256="0"*64, license="cc0-1.0", seed=2075)
    recipe = root / "model.json"
    recipe.write_text(json.dumps(model, indent=2), encoding="utf-8")
    os.environ.update(SM8650_MODEL_FILE=str(recipe), SM8650_WORK_DIR=str(root),
                      QAIRT_SDK_ROOT=str(args.sdk.expanduser().resolve()),
                      QAIRT_PYTHON=str(args.sdk_python.expanduser().absolute()))
    from sm8650_qnn.network import disable_network

    disable_network()
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    from sm8650_qnn.prepared import seal_prepared, write

    directory = root / "prepared" / "float" / "decoder_s1"
    captures = root / "prepared" / "calibration" / "decoder_s1"
    directory.mkdir(parents=True)
    captures.mkdir(parents=True)
    rng = np.random.default_rng(2075)
    weights = rng.normal(0, .1, (32, 32)).astype(np.float32)
    gamma = np.linspace(.8, 1.2, 32, dtype=np.float32)
    initializers = [
        numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(gamma, "gamma"),
        numpy_helper.from_array(np.array(1e-6, dtype=np.float32), "epsilon"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "axes"),
    ]
    nodes = [
        helper.make_node("Mul", ["hidden_states", "hidden_states"], ["squared"]),
        helper.make_node("ReduceMean", ["squared", "axes"], ["mean"], keepdims=1),
        helper.make_node("Add", ["mean", "epsilon"], ["variance"]),
        helper.make_node("Sqrt", ["variance"], ["denominator"]),
        helper.make_node("Div", ["hidden_states", "denominator"], ["normalized"]),
        helper.make_node("Mul", ["normalized", "gamma"], ["scaled"]),
        helper.make_node("MatMul", ["scaled", "weights"], ["hidden_states_out"]),
    ]
    graph = helper.make_graph(nodes, "decoder_s1", [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, 1, 32])],
                              [helper.make_tensor_value_info("hidden_states_out", TensorProto.FLOAT, [1, 1, 32])], initializers)
    network = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(network)
    onnx.save(network, directory / "model.onnx")
    session = ort.InferenceSession(str(directory / "model.onnx"), providers=["CPUExecutionProvider"])
    metrics = []
    for index in range(16):
        x = (np.linspace(-1e-4, 1e-4, 32, dtype=np.float32).reshape(1, 1, 32)
             if index == 0 else rng.normal(0, .2, (1, 1, 32)).astype(np.float32))
        np.savez_compressed(captures / f"{index:03d}.npz", hidden_states=x)
        expected = (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-6)) * gamma) @ weights
        actual = session.run(None, {"hidden_states": x})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        if index in (0, 15):
            delta = actual.astype(np.float64) - expected.astype(np.float64)
            metrics.append({"sample": index, "output": "hidden_states_out", "max_absolute_error": float(np.max(np.abs(delta))),
                            "rms_error": float(np.sqrt(np.mean(delta*delta)))})
    base = root / "prepared"
    write(base / "float-manifest.json", {
        "model": model["repo_id"], "calibration_profile": "synthetic-smoke", "graphs": {
            "decoder_s1": {"part": "decoder", "sequence_length": 1, "validated": True,
                "directory": "prepared/float/decoder_s1", "calibration_directory": "prepared/calibration/decoder_s1",
                "inputs": {"hidden_states": {"shape": [1, 1, 32], "dtype": "float32"}}, "outputs": ["hidden_states_out"]}
        },
    })
    write(base / "calibration-summary.json", {"decoder_s1": 16})
    write(base / "calibration-provenance.json", {"source": "deterministically generated synthetic test tensors", "seed": 2075, "example_count": 16})
    write(base / "xlam-float-validation.json", {"decoder_s1": metrics})
    write(base / "float-reference-check.json", [{"max_absolute_error": max(m["max_absolute_error"] for m in metrics),
                                               "rms_error": max(m["rms_error"] for m in metrics), "reference_backend": "NumPy"}])
    seal_prepared(origin="synthetic_sdk_smoke")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT / "src")
    subprocess.run([sys.executable, "-m", "sm8650_qnn", "--model-file", str(recipe), "--work-dir", str(root),
                    "--sdk", str(args.sdk.resolve()), "--sdk-python", str(args.sdk_python.absolute()), "build"],
                   env=env, check=True)
    summary = {}
    for precision in ("w4a8", "w4a16", "w8a8", "w8a16", "w16a16"):
        report = json.loads((root / "variants" / precision / "native-artifact-report.json").read_text(encoding="utf-8"))
        if not report["complete"]:
            raise RuntimeError("Incomplete smoke inspection")
        summary[precision] = {"complete": True, "contexts": len(report["contexts"]), "graphs": len(report["precision"]),
                              "float_max_absolute_error": report["local_float_dlc_validation"]["decoder_s1"]["max_absolute_error"]}
    write(root / "sdk-smoke-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
