"""Package inspected variants locally. SPDX-License-Identifier: CC0-1.0."""
import json
from pathlib import Path
import shutil

from .prepared import digest, read, write
from .settings import BASE, MODEL, MODEL_ID, PROJECT, SOURCE_RECORD, TARGET, VARIANTS, variant_dir


def package(output=None):
    output = (output or PROJECT / "outputs" / f"{MODEL_ID}-qnn-sm8650-xlam-local").expanduser().resolve()
    if output.exists():
        raise FileExistsError("Package destination already exists; choose a new --output directory")
    source = read(SOURCE_RECORD)
    if source["repo_id"] != MODEL["repo_id"] or source["revision"] != MODEL["revision"]:
        raise ValueError("Tokenizer source differs from the pinned model")
    snapshot = Path(source["snapshot_path"])
    original_license = snapshot / "LICENSE"
    if not original_license.is_file():
        raise FileNotFoundError("The downloaded model LICENSE is required for packaging")
    upstream_license = PROJECT / "third_party" / "ai-hub-models" / "src" / "LICENSE"
    if not upstream_license.is_file():
        raise FileNotFoundError("Pinned AI Hub Models source LICENSE is required; run the setup instructions")
    reports = {}
    for precision in VARIANTS:
        directory = variant_dir(precision)
        report = read(directory / "native-artifact-report.json")
        if (not report.get("complete") or len(report["contexts"]) != MODEL["expected_contexts"]
                or len(report["precision"]) != MODEL["expected_graphs"]):
            raise ValueError(f"A complete native inspection is required for {precision}")
        for part, entry in report["contexts"].items():
            binary = directory / "context" / f"{part}.bin"
            if not binary.is_file() or digest(binary) != entry["sha256"]:
                raise ValueError(f"Inspected context changed: {precision}/{part}")
        reports[precision] = report
    # Stage beside the destination and publish the directory only after all copies verify.
    staging = output.with_name(output.name + ".pending")
    if staging.exists():
        raise FileExistsError("An incomplete package staging directory exists; inspect it or choose a new destination")
    staging.mkdir(parents=True)
    shutil.copyfile(original_license, staging / "LICENSE")
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "tokenizer.model", "special_tokens_map.json"):
        if (snapshot / filename).is_file():
            shutil.copyfile(snapshot / filename, staging / filename)
    config = read(snapshot / "config.json")
    write(staging / "text_config.json", config.get("text_config", config))
    provenance = read(BASE / "calibration-provenance.json")
    write(staging / "calibration-provenance.json", provenance)
    catalog = {}
    for precision, report in reports.items():
        destination = staging / precision
        destination.mkdir()
        for part, entry in report["contexts"].items():
            target = destination / f"{part}.bin"
            shutil.copyfile(variant_dir(precision) / "context" / target.name, target)
            if digest(target) != entry["sha256"]:
                raise RuntimeError("Copied context checksum differs")
            entry["file"] = f"{precision}/{target.name}"
        write(destination / "model-info.json", report)
        catalog[precision] = {
            "requested_weight_bits": VARIANTS[precision][0], "requested_activation_bits": VARIANTS[precision][1],
            "context_files": [entry["file"] for entry in report["contexts"].values()],
            "total_context_bytes": report["total_context_bytes"],
            "mixed_precision_details": f"{precision}/model-info.json",
        }
    write(staging / "variants.json", {
        "model": MODEL["repo_id"], "revision": MODEL["revision"], "target": TARGET,
        "build_origin": "local_qairt", "variants": catalog,
        "phone_inference_tested": False, "speed_tested": False,
    })
    dataset = MODEL["calibration"]
    (staging / "NOTICE").write_text(
        f"Base model: {MODEL['repo_id']} at {MODEL['revision']} ({MODEL['license']}).\n"
        f"Calibration source: {dataset['repo_id']} at {dataset['revision']} ({dataset['license']}).\n"
        f"Dataset: https://huggingface.co/datasets/{dataset['repo_id']}\n"
        "Dataset license: https://creativecommons.org/licenses/by/4.0/\n"
        "Attribution: Salesforce APIGen / xLAM dataset authors; APIGen (2024), https://arxiv.org/abs/2406.18518\n"
        "Changes for calibration: seeded selection, normalized tool descriptions, chat-template rendering, tokenization and state capture.\n"
        "No fine-tuning was performed. Raw dataset records are not included.\n"
        "The conversion uses Qualcomm AI Hub Models (BSD-3-Clause) and separately licensed QAIRT tools.\n"
        "The CC0 license of the recipe repository does not relicense this model or external materials.\n",
        encoding="utf-8", newline="\n",
    )
    licenses = staging / "licenses"
    licenses.mkdir()
    shutil.copyfile(upstream_license, licenses / "AI-Hub-Models-BSD-3-Clause.txt")
    (staging / "README.md").write_text(
        f"---\nlicense: {MODEL['license']}\nbase_model: {MODEL['repo_id']}\n"
        "library_name: qnn\ntags:\n- qnn\n- qairt\n- sm8650\n---\n\n"
        f"# {MODEL['repo_id'].split('/')[-1]} — local QNN variants for SM8650\n\n"
        "Five post-training quantization recipes share a pinned xLAM calibration selection. "
        "Choose one precision folder and use all its context binaries together. "
        "The root tokenizer assets are shared. These files require an application that manages the model's "
        "tensor inputs, outputs and persistent states; this is not a standalone Android application.\n\n"
        f"Target: SM8650, SoC {TARGET['soc_model']}, HTP v{TARGET['dsp_arch']}. "
        f"Context length {MODEL['context_length']}; batch 1; prefill {MODEL['prefill_sequence_length']}; decode 1. "
        "Native token inputs are INT32 and state/hidden I/O is FP32.\n\n"
        "Requested precision names describe the conversion recipe. Embeddings and some operators can use "
        "other widths; each model-info.json records the inspected encodings and tensor contracts.\n\n"
        "Source float adaptation, ONNX exports and host float DLC outputs were checked. "
        "Quantized model accuracy, phone inference, decoding speed, peak RAM and power remain untested. "
        "See NOTICE for source attribution and separate license scopes.\n",
        encoding="utf-8", newline="\n",
    )
    lines = [f"{digest(path)}  {path.relative_to(staging).as_posix()}" for path in sorted(staging.rglob("*")) if path.is_file()]
    (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    staging.replace(output)
    print(f"Local package complete: {output.name}. No upload performed.")
