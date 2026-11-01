"""Explicit, pinned asset downloads. SPDX-License-Identifier: CC0-1.0."""
from .prepared import digest, write
from .settings import DATASET_RECORD, MODEL, ROOT, SOURCE_RECORD


def fetch():
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot = snapshot_download(
        MODEL["repo_id"], revision=MODEL["revision"],
        cache_dir=ROOT / "cache" / "huggingface",
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "LICENSE*", "NOTICE*"],
    )
    write(SOURCE_RECORD, {
        "repo_id": MODEL["repo_id"], "revision": MODEL["revision"],
        "license": MODEL["license"], "snapshot_path": snapshot,
    })
    dataset = MODEL["calibration"]
    filename = hf_hub_download(
        dataset["repo_id"], dataset["filename"], repo_type="dataset", revision=dataset["revision"],
        cache_dir=ROOT / "cache" / "huggingface",
    )
    if digest(filename) != dataset["sha256"]:
        raise RuntimeError("Dataset bytes differ from the pinned source hash")
    write(DATASET_RECORD, {
        key: dataset[key] for key in ("repo_id", "revision", "license", "sha256")
    } | {"files": {dataset["filename"]: filename}})
    print("Pinned model and dataset downloaded. Subsequent preparation and conversion stages are local.")
