# SM8650 QNN recipes

Local QAIRT conversion recipes for the Snapdragon Galaxy S24, using the SM8650-AC profile: QNN SoC **57**, HTP **v75**. The initial model is the text decoder of **Qwen3.5-0.8B**, with post-training calibration using a pinned selection from the Salesforce xLAM dataset.

Windows uses WSL2 and the same Linux setup and conversion commands. Model export, calibration, DLC conversion, quantization and context generation run locally. AI Hub Workbench accounts, jobs and model uploads are not used. Setup and the explicit `fetch` command download dependencies and source assets.

Start with [Windows / WSL setup](docs/setup-windows.md) or [Linux setup](docs/setup-linux.md). Then follow [the conversion recipes](docs/recipes.md). See [validation and limits](docs/validation.md) and [adding another model](docs/adding-models.md).

| Recipe | Requested weight bits | Requested activation bits |
| --- | ---: | ---: |
| W4A8 | 4 | 8 |
| W4A16 | 4 | 16 |
| W8A8 | 8 | 8 |
| W8A16 | 8 | 16 |
| W16A16 | 16 | 16 |

W16A16 uses fixed-point quantization. It is distinct from FP16. Native model inputs/outputs retain their original types: INT32 token IDs and FP32 hidden/state tensors. Some operators and embedding tables use other widths; inspection records the actual encodings.

After setup, activate the export environment and run:

```bash
source .venv-export/bin/activate
export QAIRT_PYTHON="$PWD/.venv-sdk/bin/python"
sm8650-qnn doctor
sm8650-qnn fetch
sm8650-qnn prepare
sm8650-qnn build --pilot-only
sm8650-qnn build --variants all
sm8650-qnn package
```

Set `QAIRT_SDK_ROOT` as described in the setup guide before running these commands. xLAM access may require accepting its dataset terms and signing in locally with `hf auth login`.

The initial model uses batch 1, a 2,048-token context, 64-token prefill and single-token decode. Each precision produces **15 graphs in 8 context binaries**. A packaged bundle contains five precision folders and shared tokenizer assets. It requires an application that manages tensor inputs, outputs and persistent states; an Android inference application is outside this repository's scope.

The repository contains source and recipes. Downloaded weights, SDK files, raw dataset records, calibration tensors, generated models and private build logs are excluded from Git. New model adapters can reuse the quantization and SM8650 target code.

Original project code, recipes and documentation are dedicated under **[CC0-1.0](LICENSE)**. External models, datasets, dependencies and Qualcomm tools retain their own licenses; see [NOTICE](NOTICE). CC0 does not relicense generated model weights.
