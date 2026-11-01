# Validation and current limits

Conversion success, model correctness and phone performance are separate checks.

The full-model reference workflow generated all five Qwen3.5-0.8B xLAM variants with QAIRT 2.51.40 on a Windows x86-64 host. It inspected **40 context binaries containing 75 graphs**. Comparisons of the **15 float DLCs** with their ONNX sources had a largest absolute error of approximately **2.96e-5**. These measurements validate float conversion and artifact structure, not quantized model quality.

The reusable repository adds source regression tests and a small generated-graph SDK test. The SDK test uses 16 synthetic captures and checks epsilon-inside-square-root normalization, a matrix operation, float output agreement, all five quantization recipes and native SM8650/HTP v75 metadata. It does not use xLAM or download model weights.

Fresh checks of the shared implementation:

| Check | Result |
| --- | --- |
| Source regressions, Windows / Python 3.12.10 | 9 passed; 1 Linux-specific test skipped |
| Source regressions, WSL Ubuntu 22.04 / Python 3.12.16 | 10 passed |
| Generated-graph SDK test, native Windows | All five recipes generated and inspected successfully |
| Generated-graph SDK test, WSL Linux | All five recipes generated and inspected successfully |
| Qwen adapter versus original model, prefill 64 and decode 1 | Passed; largest absolute error about 4.44e-5 |
| Fresh Qwen ONNX exports | Embedding at 1/64, first four-layer decoder at 1/64, and output head at 1 all passed source comparisons |
| Fresh xLAM selection | All 256 records, token sequences and capture positions matched the reference selection |

The SDK checks use the pinned build and generated tensors. They do not establish full-model Linux conversion coverage or quantized inference accuracy.

Run the source tests after installing the test dependencies:

```bash
python -m pip install -r requirements/test.txt
python -m unittest discover -s tests -v
python scripts/audit-source.py
```

To test an installed SDK, use an interpreter with ONNX Runtime, the SDK Python dependencies and `requirements/sdk.txt`. Give the SDK subprocess interpreter explicitly:

```bash
python scripts/sdk-smoke.py --sdk "$QAIRT_SDK_ROOT" --sdk-python "$QAIRT_PYTHON" --work-dir work/sdk-smoke
```

Use a fresh smoke-test directory. The test records a private `sdk-smoke-summary.json` after all five variants pass. CI runs the source tests on Linux and Windows; it does not redistribute or download the proprietary SDK.

The newly packaged CLI has not yet completed a fresh full-model run covering all 256 captures, all 15 graphs and all five variants together. Full-model reference results, fresh partial real-model exports and small SDK tests are different levels of coverage.

No phone inference, quantized-model accuracy, decoding throughput, power consumption or peak device RAM has been measured here. The repository does not claim a fastest quantization or guaranteed fit within a phone's available RAM. Runtime behavior needs testing on the intended Snapdragon device with compatible QNN libraries and an application that manages the complete tensor/state contract.
