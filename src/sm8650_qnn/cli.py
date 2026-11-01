"""Command line entry point. SPDX-License-Identifier: CC0-1.0."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def parser():
    result = argparse.ArgumentParser(description="Local QAIRT recipes for the SM8650 HTP")
    result.add_argument("--model", default="qwen3.5-0.8b", help="Model recipe identifier")
    result.add_argument("--model-file", type=Path, help="An explicit model recipe JSON file")
    result.add_argument("--work-dir", type=Path, help="Private work directory for this model")
    result.add_argument("--sdk", type=Path, help="Extracted SDK; defaults to QAIRT_SDK_ROOT")
    result.add_argument("--sdk-python", type=Path, help="SDK interpreter; defaults to QAIRT_PYTHON")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List model recipes and quantizations")
    commands.add_parser("doctor", help="Check the pinned SDK and both Python environments")
    commands.add_parser("fetch", help="Download the pinned model and calibration dataset")
    prep = commands.add_parser("prepare", help="Export FP32 graphs and capture calibration inputs locally")
    prep.add_argument("--stage", choices=["all", "bootstrap", "select", "capture", "validate"], default="all")
    build = commands.add_parser("build", help="Convert, validate, quantize and generate native contexts")
    build.add_argument("--variants", nargs="+", default=["all"], choices=["all", "w4a8", "w4a16", "w8a8", "w8a16", "w16a16"])
    build.add_argument("--pilot-only", action="store_true", help="Stop after the paired W4A16 decoder pilot")
    inspect = commands.add_parser("inspect", help="Inspect an existing variant without phone inference")
    inspect.add_argument("--precision", required=True, choices=["w4a8", "w4a16", "w8a8", "w8a16", "w16a16"])
    inspect.add_argument("--parts", nargs="+")
    commands.add_parser("verify", help="Verify the prepared-input seal")
    commands.add_parser("status", help="Show the current private workflow status")
    package = commands.add_parser("package", help="Package complete variants and shared tokenizer assets locally")
    package.add_argument("--output", type=Path, help="Destination; defaults to the ignored outputs/ directory")
    return result


def configure(args):
    os.environ["SM8650_MODEL"] = args.model
    for name, value in (
        ("SM8650_MODEL_FILE", args.model_file), ("SM8650_WORK_DIR", args.work_dir),
        ("QAIRT_SDK_ROOT", args.sdk), ("QAIRT_PYTHON", args.sdk_python),
    ):
        if value is not None:
            expanded = value.expanduser()
            os.environ[name] = str(expanded.absolute() if name == "QAIRT_PYTHON" else expanded.resolve())


def worker(module, arguments=(), sdk=False):
    from .settings import EXPORT_PYTHON, ROOT, export_environment, sdk_environment

    ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [EXPORT_PYTHON, "-u", "-m", f"sm8650_qnn.{module}", *arguments],
        env=sdk_environment() if sdk else export_environment(), cwd=ROOT, check=True,
    )


def doctor():
    from .settings import HOST, MODEL, SDK_PYTHON, SDK_VERSION, TARGET, require_sdk, sdk_environment

    require_sdk()
    expected = {
        "torch": "2.10.0", "transformers": "5.7.0", "onnx": "1.18.0",
        "onnxruntime": "1.22.1", "onnxscript": "0.7.0", "numpy": "2.4.4",
        "qai-hub-models": "0.64.0",
    }
    installed = {}
    for package, version in expected.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"Missing export dependency: {package}; run the setup instructions") from None
        if actual.split("+")[0] != version:
            raise RuntimeError(f"Export dependency {package} is {actual}; the recipe pins {version}")
        installed[package] = actual
    probe = subprocess.run(
        [SDK_PYTHON, "-c", "import json,sys,numpy,onnx; from qti.aisw.converters.common.converter_ir import op_graph_optimizations; from qti.aisw.converters.onnx.onnx_to_ir import OnnxConverterFrontend; print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'onnx':onnx.__version__}))"],
        env=sdk_environment(), capture_output=True, text=True, check=True,
    )
    sdk_packages = json.loads(probe.stdout.strip().splitlines()[-1])
    if sdk_packages["numpy"] != "1.26.4" or sdk_packages["onnx"] != "1.18.0":
        raise RuntimeError("SDK dependencies differ from requirements/sdk.txt")
    print(json.dumps({
        "status": "READY", "model": MODEL["repo_id"], "host": HOST.tag,
        "sdk_version": SDK_VERSION, "sdk_build_id": TARGET["qairt_build_id"],
        "soc_model": TARGET["soc_model"], "dsp_arch": TARGET["dsp_arch"],
        "export_packages": installed, "sdk_python": sdk_packages,
        "phone_required_for_conversion": False,
    }, indent=2))


def prepare(stage):
    from .locking import workflow_lock, keep_awake
    from .prepared import seal_prepared, verify_prepared
    from .settings import BASE, BATCH, DATASET_RECORD, MODEL, SOURCE_RECORD

    if not SOURCE_RECORD.is_file() or not DATASET_RECORD.is_file():
        raise RuntimeError("Run fetch before prepare")
    adapter = MODEL["adapter"]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", adapter):
        raise ValueError("Invalid adapter module name")
    with workflow_lock(BATCH / "workflow.lock"), keep_awake():
        seal = BASE / "prepared-seal.json"
        if stage == "all" and seal.is_file():
            verify_prepared()
            print("Reusing verified prepared inputs.")
            return
        seal.unlink(missing_ok=True)
        if stage in {"all", "bootstrap"}:
            worker(f"adapters.{adapter}")
        if stage == "all":
            worker("calibration_xlam")
        elif stage != "bootstrap":
            worker("calibration_xlam", ["--stage", stage])
        if stage in {"all", "validate"}:
            seal_prepared()


def main(argv=None):
    args = parser().parse_args(argv)
    configure(args)
    from . import settings

    try:
        if args.command == "list":
            print(json.dumps({"models": [p.stem for p in sorted((settings.PROJECT / "models").glob("*.json"))], "quantizations": settings.VARIANTS}, indent=2))
            return
        if args.command == "fetch":
            from .fetch import fetch

            fetch()
            return
        from .network import disable_network

        disable_network()
        if args.command == "doctor":
            doctor()
        elif args.command == "prepare":
            prepare(args.stage)
        elif args.command in {"build", "inspect"}:
            settings.require_sdk()
            if args.command == "build":
                selected = args.variants
                if len(set(selected)) != len(selected) or ("all" in selected and len(selected) != 1):
                    raise ValueError("Use all alone, or a list of distinct quantizations")
                arguments = [] if selected == ["all"] else ["--variants", *selected]
                if args.pilot_only:
                    arguments.append("--pilot-only")
                worker("convert_local", arguments)
            else:
                arguments = ["--precision", args.precision]
                if args.parts:
                    arguments += ["--parts", *args.parts]
                worker("inspect_native", arguments)
        elif args.command == "verify":
            from .prepared import verify_prepared

            verify_prepared()
            print("Prepared-input checksums match.")
        elif args.command == "status":
            path = settings.BATCH / "workflow-status.json"
            print(path.read_text(encoding="utf-8") if path.is_file() else "No conversion has started.")
        elif args.command == "package":
            from .bundle import package

            package(args.output)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
