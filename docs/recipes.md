# Local conversion recipes

The model recipe pins both the checkpoint and xLAM revision. The target profile maps SM8650-AC to QNN's **SM8650 SoC identifier 57 and HTP v75**. Qualcomm's [target table](https://docs.qualcomm.com/nav/home/QNN_general_overview.html?product=924033590759186372) documents the QNN mapping; its [Snapdragon 8 Gen 3 product brief](https://docs.qualcomm.com/doc/87-71408-1/87-71408-1_REV_G_Snapdragon_8_gen_3_Mobile_Platform_Product_Brief.pdf) lists SM8650 part variants. This profile does not apply a separate AC-specific clock configuration.

With the export environment active and both SDK variables set:

```bash
sm8650-qnn doctor
sm8650-qnn fetch
sm8650-qnn prepare
sm8650-qnn build --pilot-only
sm8650-qnn build --variants all
sm8650-qnn package
```

`prepare` strictly loads the text weights, checks the adapted model against the original implementation, exports fixed-shape FP32 ONNX graphs, captures xLAM state inputs and validates the first and last captured examples. It then seals the prepared graphs, external weights and calibration inputs with content hashes. Preparation and conversion use Python network guards and offline Hugging Face settings; the native QAIRT tools run on the host.

The calibration selection contains **256 existing xLAM examples**, using seed **20261030**. A seeded shuffle forms a candidate pool, followed by selection across available 512-token length buckets and single/multiple reference-call groups. Complete rendered conversations must fit the 2,048-token context. Queries and reference answers come from the dataset. Function descriptions are normalized for the model's original chat template, rendered in non-thinking mode and tokenized. No fine-tuning or synthetic example generation is performed.

Each example supplies one selected 64-token prefill capture and one teacher-forced reference-answer decode capture per applicable graph. Earlier tokens execute to produce real recurrent and attention states. The output head is captured at decode. Selection IDs, transformations, versions and source hashes are recorded in the private work directory; raw records and token captures are not committed.

Every build starts with the W4A16 decoder pilot. A successful pilot is reused during the full build. The remaining work runs sequentially to limit host memory use. To build selected variants instead:

```bash
sm8650-qnn build --variants w4a8 w8a8
```

The five JSON recipes request these options:

| Option | Value |
| --- | --- |
| Weight / activation bits | From the selected recipe |
| Bias / floating-point precision | 32 / 32 bits |
| Parameter / activation calibration | min-max / min-max |
| Per-channel / per-row quantization | Disabled |
| Native input files | Enabled |
| Original I/O datatypes | Preserved |
| W16 parameter schema | Symmetric |
| W16 restricted steps | `-0x8000 0x7F7F` |

Conversion preserves ONNX output order and skips ONNX simplification. For this Qwen adapter, the process-local converter wrapper disables `MATCH_RMSNORM`. The validated source computes epsilon inside the square root; the tested SDK's fused CPU normalization changed that behavior. The wrapper retains primitive operations, leaves SDK files untouched, and inspection rejects fused `RmsNorm` in the decoder and head DLCs.

Before quantization, every float DLC runs one captured input on the host QNN CPU backend and is compared with its ONNX source. All outputs must be finite, with maximum absolute error at most 0.02 and RMS error at most 0.002. Quantization uses the same complete captured inputs for each variant. Requested bit widths do not imply uniform encoding of every tensor; the inspector checks the main matrices and records exceptions and actual activation widths.

Context generation uses the host `QnnModelDlc` plugin and `QnnHtp` backend, with optimization 3, weight sharing, `vtcm_mb=0`, relaxed FP16 precision enabled and `short_depth_conv_on_hmx_off=true`. Prefill/decode graphs for each partition share a context binary. Native inspection checks target identity, graph inventory, tensor shapes/types, encodings and checksums.

The initial model produces embedding, six four-layer decoder partitions and output-head contexts: **8 `.bin` files per variant**. Use all eight files from one precision together. Hidden and state I/O remains FP32; token IDs are INT32. The reference adapter describes left padding, masks, rotary inputs, recurrent-state replacement and full-attention cache updates.

Re-run the same build command to resume. Success records bind source files, recipes, calibration tensors and actual output bytes. Interrupted outputs are staged before promotion. A changed prepared input fails verification rather than silently reusing a prior conversion. Use a fresh `--work-dir` for a changed recipe or calibration selection:

```bash
sm8650-qnn --model qwen3.5-0.8b --work-dir work/new-run fetch
sm8650-qnn --model qwen3.5-0.8b --work-dir work/new-run prepare
```

Global options such as `--model`, `--work-dir`, `--sdk` and `--sdk-python` go before the subcommand. `status` reports progress and `verify` checks the prepared-input seal. Logs and intermediate files remain under the ignored `work/` directory.

`package` requires all five complete inspections. It creates an ignored `outputs/` bundle with context files, shared tokenizer assets, source licenses, attribution, metadata and SHA-256 checksums. It does not upload anything or create a Git repository for model weights. Choose a new `package --output` destination if a bundle already exists.
