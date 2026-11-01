"""Recipe configuration shared by local worker processes. SPDX-License-Identifier: CC0-1.0."""
import json
import os
from pathlib import Path
import re
import sys

from .host import Host

PROJECT = Path(__file__).resolve().parents[2]
MODEL_ID = os.environ.get("SM8650_MODEL", "qwen3.5-0.8b")
if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", MODEL_ID):
    raise ValueError("Invalid model recipe identifier")
MODEL_FILE = Path(os.environ.get("SM8650_MODEL_FILE", PROJECT / "models" / f"{MODEL_ID}.json")).expanduser().resolve()
if not MODEL_FILE.is_file():
    raise ValueError(f"No model recipe named {MODEL_ID}")
MODEL = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
MODEL_ID = MODEL["id"]
if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", MODEL_ID):
    raise ValueError("Invalid model recipe identifier")
TARGET = json.loads((PROJECT / "targets" / f"{MODEL['target']}.json").read_text(encoding="utf-8"))
ROOT = Path(os.environ.get("SM8650_WORK_DIR", PROJECT / "work" / MODEL_ID)).expanduser().resolve()
OUT = BASE = ROOT / "prepared"
BATCH = ROOT / "local"
SOURCE_RECORD = ROOT / "model-source.json"
DATASET_RECORD = ROOT / "dataset-source.json"
REPO_ID = MODEL["repo_id"]
SAMPLES = MODEL["calibration"]["samples"]
SDK = Path(os.environ.get("QAIRT_SDK_ROOT", PROJECT / "sdks" / "qairt")).expanduser().resolve()
HOST = Host.current(SDK)
BIN, LIB = HOST.bin, HOST.lib
# Preserve venv symlinks: resolving bin/python to the base executable loses sys.prefix.
SDK_PYTHON = str(Path(os.environ.get("QAIRT_PYTHON", sys.executable)).expanduser().absolute())
EXPORT_PYTHON = str(Path(os.environ.get("SM8650_EXPORT_PYTHON", sys.executable)).expanduser().absolute())
SDK_VERSION = TARGET["qairt_version"]
RECIPES = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((PROJECT / "recipes").glob("*.json"))}
VARIANTS = {key: (value["weight_bits"], value["activation_bits"]) for key, value in RECIPES.items()}


def variant_dir(precision):
    if precision not in VARIANTS:
        raise ValueError(f"Unknown precision: {precision}")
    return ROOT / "variants" / precision


def require_sdk():
    metadata = SDK / "sdk.yaml"
    if not metadata.is_file():
        raise RuntimeError("Set QAIRT_SDK_ROOT to the extracted QAIRT SDK directory containing sdk.yaml")
    text = metadata.read_text(encoding="utf-8")
    version = re.search(r"^version:\s*(\S+)", text, re.MULTILINE)
    build = re.search(r"^build_id:\s*(\S+)", text, re.MULTILINE)
    if not version or version[1] != SDK_VERSION or not build or build[1] != TARGET["qairt_build_id"]:
        raise RuntimeError("SDK version/build differs from the pinned target profile; review the profile and normalization guard before changing it")
    required = [
        BIN / "qairt-converter", BIN / "qairt-quantizer", BIN / "qairt-dlc-to-json",
        HOST.executable("qnn-context-binary-generator"), HOST.executable("qnn-context-binary-utility"),
        HOST.executable("qnn-net-run"), HOST.library("QnnHtp"), HOST.library("QnnCpu"),
        HOST.library("QnnModelDlc"), HOST.library("QnnHtpNetRunExtensions"),
        Path(SDK_PYTHON),
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("The SDK host tools, host libraries, or QAIRT_PYTHON interpreter are missing")
    if any(char.isspace() for char in str(ROOT)):
        raise RuntimeError("Use a work directory without whitespace: QNN named raw input lists use whitespace separators")


def sdk_environment():
    env = HOST.environment(sdk_python=True)
    if HOST.system == "Linux":
        executable = Path(SDK_PYTHON)
        prefix = executable.resolve().parent.parent
        config = executable.parent.parent / "pyvenv.cfg"
        if config.is_file():
            home = re.search(r"^home\s*=\s*(.+)$", config.read_text(encoding="utf-8"), re.MULTILINE)
            if home:
                prefix = Path(home[1].strip()).parent
        # Standalone Python distributions may keep libpython outside ldconfig's paths.
        directory = prefix / "lib"
        if directory.is_dir():
            env["LD_LIBRARY_PATH"] = os.pathsep.join([str(HOST.lib), str(directory), env.get("LD_LIBRARY_PATH", "")])
    return env


def export_environment():
    return HOST.environment(sdk_python=False)
