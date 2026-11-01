# Linux setup

The documented setup uses **Ubuntu 24.04 x86-64 and Python 3.12**. Windows users run these steps inside WSL2. Qualcomm documents supported host/Python combinations in its [Linux setup guide](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/linux_setup.html?product=1601111740010412).

Allow at least 32 GiB of host RAM and about 150 GB of free working storage for the initial model and all five variants. These are host planning estimates, not phone memory measurements. The 256-example captures expand to roughly 37 GB of native raw inputs; exports, compressed captures and generated models need additional storage. Larger models need a separate resource assessment.

1. Install the basic host tools:

   ```bash
   sudo apt update
   sudo apt install git unzip python3.12 python3.12-venv python3.12-dev build-essential
   ```

   Then clone the repository using its GitHub **Code** URL and enter its folder:

   ```bash
   cd sm8650-qnn-recipes
   ```

2. Download QAIRT **2.51.40**, build **260930230134**, from Qualcomm under its SDK terms. Extract it separately. Set the root to the directory containing `sdk.yaml`, using your installation location. A conventional example is:

   ```bash
   export QAIRT_SDK_ROOT="$HOME/sdks/qairt/2.51.40.260930"
   test -f "$QAIRT_SDK_ROOT/sdk.yaml"
   ```

   Review the SDK's host dependency check, then install its required OS packages using the SDK instructions:

   ```bash
   bash "$QAIRT_SDK_ROOT/bin/check-linux-dependency.sh" --dry-run
   ```

   This repository does not distribute the SDK. A different SDK build requires reviewing the target profile and repeating the normalization and native-context checks before changing the version pin.

3. Install the two Python environments and pinned upstream model dependency:

   ```bash
   bash scripts/bootstrap-linux.sh
   source .venv-export/bin/activate
   export QAIRT_PYTHON="$PWD/.venv-sdk/bin/python"
   sm8650-qnn doctor
   ```

   `.venv-export` contains PyTorch, Transformers and ONNX Runtime for model preparation. `.venv-sdk` contains the QAIRT dependencies, including NumPy 1.26.4 and the `distutils` compatibility supplied by setuptools. The Python development package supplies the shared Python library needed by the SDK. The two environments' package paths are kept separate. The script imports AI Hub Models for its local model math; it does not configure or use Workbench.

4. Open the [xLAM dataset page](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) and accept its access terms if requested. Authenticate locally with a read-capable Hugging Face token:

   ```bash
   hf auth login
   sm8650-qnn fetch
   ```

   The official client handles token entry. Tokens and account configuration are not part of the repository. `fetch` downloads the pinned model and dataset revisions and verifies the dataset file's SHA-256.

5. Continue with [the conversion recipes](recipes.md). Repeat the two `export` commands and activate `.venv-export` when opening a new shell. No phone, ADB connection, Android NDK or Workbench account is needed for context generation with this DLC path.

For Ubuntu 22.04, install Python 3.12 for export using a suitable Python installer and retain the SDK-supported Python 3.10 environment. Then select the interpreters explicitly:

```bash
sudo apt install python3.10-venv python3.10-dev
RECIPE_PYTHON=python3.12 SDK_SETUP_PYTHON=python3.10 bash scripts/bootstrap-linux.sh
```

The shared CLI requires Python 3.12. The SDK subprocess interpreter can differ. The standard setup above uses Ubuntu 24.04 to avoid this additional interpreter installation.
