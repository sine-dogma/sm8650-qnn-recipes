# SPDX-License-Identifier: CC0-1.0
"""Prepare static Qwen3.5 text graphs and real-state calibration samples.

This script performs local correctness checks, not performance measurements.
It submits no cloud jobs. The published Qualcomm Qwen3.5 family adaptations
are used for the attention math; their scan is unrolled for fixed-size graphs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

from ..settings import ROOT, OUT, REPO_ID, SOURCE_RECORD, MODEL
from ..prepared import digest
PREFILL = MODEL["prefill_sequence_length"]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import onnx
import torch
import psutil
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from qai_hub_models.models.templates.qwen3_5.model import Qwen3_5Base
from qai_hub_models.models.templates.qwen3_5 import model_adaptations as attention
from qai_hub_models.models.templates.llm import model as llm_template



def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        raise TypeError(f'Cannot serialize {type(obj).__name__}')
    path.write_text(json.dumps(value, indent=2, default=encode) + '\n', encoding='utf-8')


def static_scan(fn, carry, xs):
    """Unroll the upstream recurrence for a fixed export sequence length."""
    outputs = []
    for index in range(int(xs[0].shape[0])):
        carry, output = fn(carry, [x[index] for x in xs])
        outputs.append(output)
    return carry, torch.stack(outputs)


attention.scan = static_scan


def memory_check(recommended_gib):
    # Portable RAM detection for Linux, WSL and Windows.
    total = psutil.virtual_memory().total / 2**30
    if total < recommended_gib:
        raise RuntimeError(f'Export recommends {recommended_gib} GiB RAM; found {total:.1f}')
    return True


llm_template.has_recommended_memory = memory_check


class StrictTextModel(Qwen3_5ForCausalLM):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # The original multimodal checkpoint prefixes the text decoder with
        # model.language_model. Do not silently initialize unmatched weights.
        kwargs['key_mapping'] = {r'^model\.language_model\.': 'model.'}
        kwargs['output_loading_info'] = True
        model, info = super().from_pretrained(*args, **kwargs)
        unexpected = [key for key in info.get('unexpected_keys', [])
                      if not key.startswith(('model.visual.', 'mtp.'))]
        if info.get('missing_keys') or info.get('mismatched_keys') or unexpected:
            raise RuntimeError(f'Text checkpoint did not load strictly: {info}')
        save_json(OUT / 'checkpoint-load.json', info)
        return model


class QwenNative(Qwen3_5Base):
    LMClass = StrictTextModel
    hf_repo_name = REPO_ID
    min_memory_recommended = MODEL.get("minimum_export_ram_gib", 24)


class EmbeddingPart(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.embedding = base.model.model.embed_tokens

    def forward(self, input_ids):
        return self.embedding(input_ids)


class DecoderPart(nn.Module):
    def __init__(self, base, start, end):
        super().__init__()
        self.start, self.end = start, end
        self.layers = nn.ModuleList(list(base.model.model.layers[start:end]))
        self.descs = base.layer_cache_descriptors[start:end]
        self.sha = nn.ModuleDict({str(i): base.sha_attention[str(i)]
                                 for i in range(start, end) if str(i) in base.sha_attention})

    def forward(self, hidden_states, attention_mask, position_ids_cos, position_ids_sin, *states):
        seq = hidden_states.shape[1]
        attends = (attention_mask[:, 0, :, -seq:] == 0).to(torch.float32)
        valid = (attends * torch.eye(seq, device=hidden_states.device)).sum(-1)
        outputs = []
        for offset, (desc, layer) in enumerate(zip(self.descs, self.layers, strict=True)):
            a, b = states[offset * 2:offset * 2 + 2]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            if desc.attention_type.value == 'linear_attention':
                hidden_states, a, b = attention.linear_attention_forward(
                    layer.linear_attn, hidden_states, valid, a, b)
            else:
                hidden_states, a, b = self.sha[str(desc.layer_idx)](
                    hidden_states, attention_mask, position_ids_cos, position_ids_sin, a, b)
            hidden_states = residual + hidden_states
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
            outputs.extend([a, b])
        return (hidden_states, *outputs)


class HeadPart(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.norm, self.head = base.model.model.norm, base.model.lm_head

    def forward(self, hidden_states):
        return self.head(self.norm(hidden_states))


class Pipeline:
    def __init__(self, base, context):
        self.base, self.context = base, context
        self.parts = [('embedding', EmbeddingPart(base))]
        span = MODEL['layers_per_partition']
        if base.llm_config.num_hidden_layers % span:
            raise ValueError('Layer count must be divisible by the configured partition span')
        for start in range(0, base.llm_config.num_hidden_layers, span):
            self.parts.append((f'layers_{start:02d}_{start + span - 1:02d}', DecoderPart(base, start, start + span)))
        self.parts.append(('lm_head', HeadPart(base)))
        for _, part in self.parts:
            part.eval()
        self.reset()

    def reset(self):
        self.seen = 0
        self.states = []
        for desc in self.base.layer_cache_descriptors:
            if desc.attention_type.value == 'linear_attention':
                shapes = desc.dummy_state_shapes(1, self.context, 1)
                self.states.append(tuple(torch.zeros(shape) for shape in shapes))
            else:
                self.states.append((torch.zeros(desc.num_kv_heads, 1, desc.head_dim, 0),
                                    torch.zeros(desc.num_kv_heads, 1, 0, desc.head_dim)))

    def state_inputs(self, seq):
        values = []
        for desc, (a, b) in zip(self.base.layer_cache_descriptors, self.states, strict=True):
            if desc.attention_type.value == 'linear_attention':
                values.extend([a, b])
            else:
                count = min(a.shape[-1], self.context - seq)
                a = a[..., -count:] if count else a[..., :0]
                b = b[..., -count:, :] if count else b[..., :0, :]
                pad = self.context - seq - count
                values.extend([torch.nn.functional.pad(a, (pad, 0)),
                               torch.nn.functional.pad(b, (0, 0, pad, 0))])
        return values

    def names(self, part):
        if isinstance(part, EmbeddingPart):
            return ['input_ids'], ['hidden_states_out']
        if isinstance(part, HeadPart):
            return ['hidden_states'], ['logits']
        inputs = ['hidden_states', 'attention_mask', 'position_ids_cos', 'position_ids_sin']
        outputs = ['hidden_states_out']
        for desc in part.descs:
            names = (['conv_state', 'recurrent_state'] if desc.attention_type.value == 'linear_attention'
                     else ['past_key', 'past_value'])
            inputs.extend(f'{name}_{desc.layer_idx}_in' for name in names)
            outputs.extend(f'{name}_{desc.layer_idx}_out' for name in names)
        return inputs, outputs

    @torch.no_grad()
    def run(self, tokens, seq, capture=None):
        if not 0 < len(tokens) <= seq or self.seen + len(tokens) > self.context:
            raise ValueError('Invalid token/context length')
        count, left = len(tokens), seq - len(tokens)
        ids = torch.zeros((1, seq), dtype=torch.int32)
        ids[0, left:] = torch.tensor(tokens, dtype=torch.int32)
        positions = torch.zeros((1, seq), dtype=torch.int64)
        positions[0, left:] = torch.arange(self.seen, self.seen + count)
        cos, sin = self.base.embedding.get_embedding(positions)
        mask = torch.full((1, 1, seq, self.context), -10000.0)
        past = min(self.seen, self.context - seq)
        for index in range(left, seq):
            if past:
                mask[0, 0, index, self.context - seq - past:self.context - seq] = 0
            mask[0, 0, index, self.context - seq + left:self.context - seq + index + 1] = 0
        state_inputs = self.state_inputs(seq)
        hidden = None
        new_states = []
        for name, part in self.parts:
            if isinstance(part, EmbeddingPart):
                inputs = (ids,)
            elif isinstance(part, HeadPart):
                inputs = (hidden[:, -1:, :].contiguous(),)
            else:
                inputs = (hidden, mask, cos, sin, *state_inputs[part.start * 2:part.end * 2])
            input_names, _ = self.names(part)
            if capture:
                capture(name, 1 if isinstance(part, HeadPart) else seq,
                        dict(zip(input_names, inputs, strict=True)))
            result = part(*inputs)
            if isinstance(part, DecoderPart):
                hidden = result[0]
                new_states.extend(zip(result[1::2], result[2::2], strict=True))
            else:
                hidden = result
        for i, (desc, (a, b)) in enumerate(zip(self.base.layer_cache_descriptors, new_states, strict=True)):
            if desc.attention_type.value == 'linear_attention':
                self.states[i] = (a, b)
            else:
                old_a, old_b = self.states[i]
                self.states[i] = (torch.cat([old_a, a[..., left:]], -1)[..., -(self.context - 1):],
                                  torch.cat([old_b, b[..., left:, :]], -2)[..., -(self.context - 1):, :])
        self.seen += count
        return hidden


def check_reference(pipeline):
    text = ('Explain how computers represent numbers, and why careful numerical '
            'precision matters when running a language model. ') * max(12, PREFILL)
    ids = pipeline.base.tokenizer(text, add_special_tokens=False)['input_ids'][:(PREFILL + 1)]
    if len(ids) != (PREFILL + 1):
        raise RuntimeError('Reference fixture too short')
    pipeline.reset()
    results = []
    cache = None
    with torch.no_grad():
        for tokens, seq in [(ids[:PREFILL], PREFILL), (ids[PREFILL:], 1)]:
            reference = pipeline.base.model(input_ids=torch.tensor([tokens]), past_key_values=cache,
                                            use_cache=True, logits_to_keep=1)
            cache = reference.past_key_values
            actual = pipeline.run(tokens, seq)
            diff = (actual - reference.logits).float()
            metric = {'sequence_length': seq, 'max_absolute_error': diff.abs().max().item(),
                      'rms_error': diff.square().mean().sqrt().item()}
            results.append(metric)
            print('Float reference comparison:', metric, flush=True)
            if metric['max_absolute_error'] > 0.02 or metric['rms_error'] > 0.002:
                raise RuntimeError('Float adaptation differs from the original model; refusing export')
    save_json(OUT / 'float-reference-check.json', results)
    pipeline.reset()
    del cache
    gc.collect()


def bootstrap(pipeline):
    """Create export fixtures; these are not used to calibrate the final quantizers."""
    text = ('Explain how computers represent numbers and why numerical precision matters. ') * max(12, PREFILL)
    tokens = pipeline.base.tokenizer(text, add_special_tokens=False)['input_ids'][:PREFILL + 1]
    if len(tokens) != PREFILL + 1:
        raise RuntimeError('Export fixture is too short')
    pipeline.reset()
    def capture(name, seq, values):
        directory = OUT / 'bootstrap' / f'{name}_s{seq}'
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / '000.pending'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **{key: value.detach().cpu().numpy() for key, value in values.items()})
        temporary.replace(directory / '000.npz')
    pipeline.run(tokens[:PREFILL], PREFILL, capture)
    pipeline.run(tokens[PREFILL:], 1, capture)
    pipeline.reset()


def export_graphs(pipeline, only):
    import onnxruntime as ort
    manifest_path = OUT / 'float-manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'graphs': {}}
    manifest.update(model=REPO_ID, context_length=pipeline.context,
                    text_only=True, state_io_dtype='float32', requested_precision='float32',
                    checkpoint={key: MODEL[key] for key in ('repo_id', 'revision', 'license')})
    for name, part in pipeline.parts:
        if only and name not in only:
            continue
        for seq in ([1] if isinstance(part, HeadPart) else [1, PREFILL]):
            key = f'{name}_s{seq}'
            previous = manifest['graphs'].get(key, {})
            source_files = previous.get('source_files', {})
            reusable = previous.get('validated') and source_files and all(
                (ROOT / previous['directory'] / filename).is_file()
                and digest(ROOT / previous['directory'] / filename) == checksum
                for filename, checksum in source_files.items())
            if reusable:
                print('Already validated:', key, flush=True)
                continue
            sample_files = sorted((OUT / 'bootstrap' / key).glob('*.npz'))
            if not sample_files:
                raise RuntimeError(f'No real calibration samples for {key}')
            with np.load(sample_files[-1], allow_pickle=False) as sample:
                values = {key: sample[key].copy() for key in sample.files}
            input_names, output_names = pipeline.names(part)
            args = tuple(torch.from_numpy(values[key]) for key in input_names)
            directory = OUT / 'float' / key
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'model.onnx'
            print('Exporting static graph:', key, flush=True)
            with torch.no_grad():
                expected = part(*args)
                if isinstance(expected, torch.Tensor):
                    expected = (expected,)
                expected = [value.numpy().copy() for value in expected]
                torch.onnx.export(part, args, str(path), input_names=input_names,
                                  output_names=output_names, opset_version=18,
                                  dynamo=True, external_data=True)
            previous_data_files = list(directory.glob('*.data'))
            model = onnx.load(str(path))
            graph_inputs = [value.name for value in model.graph.input]
            if graph_inputs != input_names:
                raise RuntimeError(f'Graph pruned/reordered inputs: {graph_inputs}; expected {input_names}')
            bad_ops = sorted({node.op_type for node in model.graph.node} & {'Loop', 'Scan', 'If'})
            if bad_ops:
                raise RuntimeError(f'Unexpected control-flow operations: {bad_ops}')
            weights_path = directory / 'weights.data'
            if weights_path.exists():
                weights_path.unlink()
            onnx.save_model(model, str(path), save_as_external_data=True,
                            all_tensors_to_one_file=True, location='weights.data', size_threshold=1024)
            for previous in previous_data_files:
                if previous != weights_path:
                    previous.unlink()
            del model
            gc.collect()
            onnx.checker.check_model(str(path))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            actual = session.run(None, values)
            errors = []
            for output_name, ref, got in zip(output_names, expected, actual, strict=True):
                error = float(np.abs(ref - got).max())
                errors.append({'output': output_name, 'max_absolute_error': error})
                if not np.allclose(ref, got, atol=0.01, rtol=0.002):
                    raise RuntimeError(f'ONNX correctness check failed: {key}/{output_name}: {error}')
            manifest['graphs'][key] = {
                'part': name, 'sequence_length': seq,
                'directory': directory.relative_to(ROOT).as_posix(),
                'source_files': {file.name: digest(file) for file in [path, *sorted(directory.glob('*.data'))]},
                'inputs': {key: {'shape': list(values[key].shape), 'dtype': str(values[key].dtype)} for key in input_names},
                'outputs': output_names, 'validated': True, 'comparison': errors,
                'calibration_directory': (OUT / 'calibration' / key).relative_to(ROOT).as_posix(),
            }
            save_json(manifest_path, manifest)
            print('Validated ONNX:', key, flush=True)
            del session, args, expected, actual
            gc.collect()


def load_pipeline():
    torch.set_num_threads(6)
    torch.manual_seed(0)
    source = json.loads(SOURCE_RECORD.read_text(encoding='utf-8'))
    if source['repo_id'] != REPO_ID or source['revision'] != MODEL['revision']:
        raise RuntimeError('Checkpoint record does not match the pinned model recipe')
    base = QwenNative.from_pretrained(checkpoint=source['snapshot_path'], sequence_length=PREFILL,
        context_length=MODEL['context_length'], host_device=torch.device('cpu'))
    return Pipeline(base, MODEL['context_length'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['check', 'bootstrap', 'export', 'all'], default='all')
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    pipeline = load_pipeline()
    check_reference(pipeline)
    if args.stage in ('bootstrap', 'all'):
        bootstrap(pipeline)
    if args.stage in ('export', 'all'):
        export_graphs(pipeline, args.only)


if __name__ == '__main__':
    main()
