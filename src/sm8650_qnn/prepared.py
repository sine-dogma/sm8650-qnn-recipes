"""Validate and seal prepared graphs and calibration captures. SPDX-License-Identifier: CC0-1.0."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from .settings import BASE, MODEL, ROOT, SAMPLES


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    pending.replace(path)


def work_path(value):
    """Only portable relative paths resolving inside this model's work directory."""
    if (not isinstance(value, str) or "\\" in value
            or PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts):
        raise ValueError("Manifest paths must be portable relative paths inside the work directory")
    result = (ROOT / value).resolve()
    if result == ROOT or not result.is_relative_to(ROOT):
        raise ValueError("Manifest path escapes the work directory")
    return result


def contract(manifest):
    if manifest.get("model") != MODEL["repo_id"]:
        raise ValueError("Prepared model does not match the selected recipe")
    if manifest.get("calibration_profile") != MODEL["calibration"]["profile"]:
        raise ValueError("Prepared calibration profile differs")
    graphs = manifest.get("graphs", {})
    if len(graphs) != MODEL["expected_graphs"]:
        raise ValueError("Prepared graph inventory is incomplete")
    parts = set()
    for key, graph in graphs.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) or not graph.get("validated"):
            raise ValueError("Graph key is invalid or the source graph was not validated")
        part = graph["part"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", part):
            raise ValueError("Invalid partition name")
        parts.add(part)
        for field in ("directory", "calibration_directory"):
            work_path(graph[field])
        if not graph.get("inputs") or not graph.get("outputs"):
            raise ValueError("Graph I/O contract is empty")
        names = list(graph["inputs"]) + graph["outputs"]
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name) for name in names):
            raise ValueError("Tensor names must be safe identifiers for native input lists")
        if len(set(graph["outputs"])) != len(graph["outputs"]):
            raise ValueError("Duplicate graph outputs")
        for spec in graph["inputs"].values():
            if spec["dtype"] not in {"float32", "int32"} or not spec["shape"] or any(
                type(dimension) is not int or dimension < 1 for dimension in spec["shape"]
            ):
                raise ValueError("A native input has an unsupported dtype or dynamic/invalid shape")
    if len(parts) != MODEL["expected_contexts"] or MODEL["pilot_part"] not in parts:
        raise ValueError("Prepared partition inventory is incomplete")
    return graphs


def inventory():
    """Hash every graph, external weight, capture and validation record."""
    manifest = read(BASE / "float-manifest.json")
    graphs = contract(manifest)
    counts = read(BASE / "calibration-summary.json")
    checks = read(BASE / "xlam-float-validation.json")
    if set(counts) != set(graphs) or set(counts.values()) != {SAMPLES} or set(checks) != set(graphs):
        raise ValueError("Complete calibration counts and float validations are required")
    reference = read(BASE / "float-reference-check.json")
    if not reference or any(
        not math.isfinite(item.get("max_absolute_error", math.inf))
        or not math.isfinite(item.get("rms_error", math.inf))
        or item["max_absolute_error"] > .02 or item["rms_error"] > .002
        for item in reference
    ):
        raise ValueError("Source-model float reference validation did not pass")
    files = [BASE / name for name in (
        "float-manifest.json", "calibration-summary.json", "calibration-provenance.json",
        "float-reference-check.json", "xlam-float-validation.json"
    )]
    for key, graph in graphs.items():
        rows = checks[key]
        if not rows or {item["sample"] for item in rows} != {0, SAMPLES - 1}:
            raise ValueError(f"First and last calibration samples were not validated: {key}")
        if any(not math.isfinite(item["max_absolute_error"]) for item in rows):
            raise ValueError(f"Nonfinite source float comparison: {key}")
        directory = work_path(graph["directory"])
        files += [directory / "model.onnx", *sorted(directory.glob("*.data"))]
        captures = sorted(work_path(graph["calibration_directory"]).glob("*.npz"))
        if [p.name for p in captures] != [f"{i:03d}.npz" for i in range(SAMPLES)]:
            raise ValueError(f"Missing or stale calibration capture files: {key}")
        files += captures
    result = {}
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Prepared file is missing or escapes the work directory")
        result[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return {"model_recipe": MODEL, "files": result}


def seal_prepared(origin="local_preparation"):
    record = {"status": "COMPLETE", "origin": origin, "inventory": inventory()}
    write(BASE / "prepared-seal.json", record)
    print("Prepared graphs and calibration captures sealed with SHA-256.", flush=True)


def verify_prepared():
    path = BASE / "prepared-seal.json"
    if not path.is_file():
        raise RuntimeError("Run prepare or import-prepared before building")
    record = read(path)
    if record.get("status") != "COMPLETE" or record.get("inventory") != inventory():
        raise RuntimeError("Prepared inputs changed; use a fresh work directory or repeat preparation and validation")
