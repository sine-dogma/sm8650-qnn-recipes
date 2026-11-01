#!/usr/bin/env bash
# SPDX-License-Identifier: CC0-1.0
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${QAIRT_SDK_ROOT:?Set QAIRT_SDK_ROOT to the extracted SDK containing sdk.yaml}"
export QAIRT_SDK_ROOT
recipe_python="${RECIPE_PYTHON:-python3.12}"
sdk_setup_python="${SDK_SETUP_PYTHON:-$recipe_python}"
"$recipe_python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
"$recipe_python" -m venv .venv-export
"$sdk_setup_python" -m venv .venv-sdk
.venv-export/bin/python -m pip install --upgrade pip
.venv-export/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
.venv-export/bin/python -m pip install -r requirements/export.txt
upstream_commit=d075bf442f46b58bfffdc79b3eaaf62636acd79b
if [ -d third_party/ai-hub-models/.git ]; then
    if [ "$(git -C third_party/ai-hub-models rev-parse HEAD)" != "$upstream_commit" ]; then
        echo 'Existing AI Hub Models checkout differs from the pinned revision; use a fresh checkout.' >&2
        exit 1
    fi
else
    mkdir -p third_party
    git clone --filter=blob:none --no-checkout https://github.com/qualcomm/ai-hub-models.git third_party/ai-hub-models
    git -C third_party/ai-hub-models checkout --detach "$upstream_commit"
fi
# The exact Git revision is authoritative; fix development-package metadata as well.
SETUPTOOLS_SCM_PRETEND_VERSION=0.64.0 .venv-export/bin/python -m pip install -c requirements/export.txt -e third_party/ai-hub-models/src
.venv-export/bin/python -m pip install -e .
.venv-sdk/bin/python -m pip install --upgrade pip
.venv-sdk/bin/python "$QAIRT_SDK_ROOT/bin/check-python-dependency"
.venv-sdk/bin/python -m pip install -r requirements/sdk.txt
echo 'Setup complete. Activate .venv-export and set QAIRT_PYTHON to .venv-sdk/bin/python.'
