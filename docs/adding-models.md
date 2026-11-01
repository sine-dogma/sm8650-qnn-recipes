# Adding another model for SM8650

Model preparation is separate from the five quantization recipes and the device profile. Adding a model does not require duplicating the QAIRT conversion, CPU float validation, quantization, context generation or inspection stages.

1. Add a JSON entry under `models/`, using the initial recipe as a reference. Pin the model revision, calibration source and hash, license, context and sequence sizes, partition span, expected graph/context counts and pilot partition. Keep machine-specific configuration in CLI arguments or environment variables.

2. Add an adapter under `src/sm8650_qnn/adapters/`. The `adapter` field selects its module. The current xLAM capture harness expects `load_pipeline()`, `check_reference(pipeline)`, `Pipeline.parts`, `Pipeline.names(part)`, `Pipeline.run(tokens, sequence_length, capture)` and `Pipeline.reset()`. The module also exposes its PyTorch/NumPy imports to the harness. The pipeline provides a tokenizer through `pipeline.base.tokenizer`.

3. The adapter's module entry point prepares FP32 ONNX graphs and export fixtures. The existing Qwen3.5 adapter shows strict text-weight loading, state layout and source-reference comparisons. An architecture with different attention, normalization or state semantics needs its own implementation and validation. Changing a model identifier alone does not establish support.

4. Emit `prepared/float-manifest.json` in the model work directory. Each graph needs a safe identifier, `part`, `sequence_length`, relative `directory`, `inputs` with static shapes/dtypes, ordered `outputs`, relative `calibration_directory` and `validated=true`. ONNX files are named `model.onnx`; external weights use `.data` files in the same directory. Paths use forward slashes and remain inside the work directory.

5. Supply one NPZ capture per selected example per graph, plus calibration counts/provenance, source float-reference comparisons and first/last ONNX-versus-adapter checks. The current native interface supports FP32 and INT32 inputs and FP32 outputs. The xLAM harness uses sequence axis 1, single-token decode, a partition named `lm_head` for the decode-only output head, and `embedding` for embedding-width exceptions. Adapters with another contract must update the harness or add a preparation path deliberately.

6. Run the source-reference checks, ONNX comparisons, prepared-input sealing, the W4A16 pilot and all requested recipes. Check actual tensor encodings and native metadata. Add the model's licenses and source attribution, and document which platforms and device behavior were actually tested.

The `primitive_rmsnorm` setting is model-specific. It enables the tested primitive normalization lowering and associated inspection guard. Compare the source math with the SDK output before changing it for a new adapter or SDK build.

Use a different work directory for each model or changed calibration profile. The engine also accepts an explicit recipe file for development:

```bash
sm8650-qnn --model-file models/new-model.json --work-dir work/new-model doctor
```

The SM8650 profile remains shared. A different SoC needs its own profile and validation; changing that target is outside the initial recipes.
