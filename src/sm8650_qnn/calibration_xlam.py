# SPDX-License-Identifier: CC0-1.0
"""Select xLAM conversations and capture real model states for QNN PTQ."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
from .settings import ROOT, OUT, SOURCE_RECORD, DATASET_RECORD, MODEL

SEED = MODEL['calibration']['seed']
COUNT = MODEL['calibration']['samples']
PREFILL = MODEL['prefill_sequence_length']
CONTEXT = MODEL['context_length']
SELECTED = OUT / 'xlam-selected-examples.json'


def selection_identity():
    return {'model': MODEL['repo_id'], 'model_revision': MODEL['revision'],
            'dataset_revision': MODEL['calibration']['revision'],
            'dataset_sha256': MODEL['calibration']['sha256'], 'seed': SEED,
            'example_count': COUNT, 'context_length': CONTEXT,
            'prefill_sequence_length': PREFILL, 'rendering_version': 1}


def validate_selection():
    from .prepared import digest
    provenance = read(OUT / 'calibration-provenance.json')
    if (provenance.get('selection_identity') != selection_identity()
            or provenance.get('selected_examples_sha256') != digest(SELECTED)):
        raise RuntimeError('Calibration selection changed; use a fresh work directory')
    return provenance


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def parsed(value):
    return json.loads(value) if isinstance(value, str) else value


def schema(value):
    aliases = {'str': 'string', 'int': 'integer', 'float': 'number',
               'bool': 'boolean', 'list': 'array', 'dict': 'object'}
    if not isinstance(value, dict):
        raise ValueError('Invalid tool parameter description')
    value = dict(value)
    if 'type' in value:
        value['type'] = aliases.get(value['type'], value['type'])
    if 'properties' in value:
        value['properties'] = {name: schema(item) for name, item in value['properties'].items()}
    if 'items' in value and isinstance(value['items'], dict):
        value['items'] = schema(value['items'])
    return value


def normalized_tools(items):
    result = []
    for tool in parsed(items):
        function = dict(tool.get('function', tool))
        parameters = function.get('parameters', {})
        if 'properties' not in parameters and parameters.get('type') != 'object':
            required = [name for name, spec in parameters.items() if spec.get('required', False)]
            parameters = {'type': 'object', 'properties': {
                name: {key: value for key, value in spec.items() if key != 'required'}
                for name, spec in parameters.items()}, 'required': required}
        function['parameters'] = schema(parameters)
        result.append({'type': 'function', 'function': function})
    return result


def select():
    if SELECTED.exists():
        data = read(SELECTED)
        validate_selection()
        if len(data) != COUNT or read(OUT / 'calibration-provenance.json')['seed'] != SEED:
            raise RuntimeError('Unexpected previous calibration selection')
        print(f'Reusing the pinned selection of {COUNT} examples.', flush=True)
        return
    from transformers import AutoTokenizer
    source, dataset = read(SOURCE_RECORD), read(DATASET_RECORD)
    tokenizer = AutoTokenizer.from_pretrained(source['snapshot_path'], is_fast=False, local_files_only=True)
    from .prepared import digest
    source_file = Path(dataset['files'][MODEL['calibration']['filename']])
    if dataset['revision'] != MODEL['calibration']['revision'] or digest(source_file) != MODEL['calibration']['sha256']:
        raise RuntimeError('Dataset record differs from the pinned source bytes')
    rows = read(source_file)
    if not isinstance(rows, list) or len(rows) != 60000:
        raise RuntimeError('Unexpected official xLAM dataset layout')
    order = list(range(len(rows)))
    rng = random.Random(SEED)
    rng.shuffle(order)
    buckets, rejected = defaultdict(list), Counter()
    for position, row_index in enumerate(order):
        row = rows[row_index]
        try:
            tools = normalized_tools(row['tools'])
            answers = parsed(row['answers'])
            names = {item['function']['name'] for item in tools}
            if not tools or not answers or any(answer['name'] not in names for answer in answers):
                raise ValueError('Unmatched reference calls')
            calls = [{'type': 'function', 'function': {'name': answer['name'],
                      'arguments': parsed(answer['arguments'])}} for answer in answers]
            messages = [{'role': 'user', 'content': row['query']}]
            prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
            conversation = tokenizer.apply_chat_template(
                messages + [{'role': 'assistant', 'content': '', 'tool_calls': calls}],
                tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
            tokens = tokenizer(conversation, add_special_tokens=False)['input_ids']
            if tokens[:len(prompt_ids)] != prompt_ids:
                raise ValueError('Chat-template prefixes differ')
            if not PREFILL + 1 <= len(prompt_ids) < len(tokens) <= CONTEXT:
                rejected['length'] += 1
                continue
            # Decode a real reference-answer token, never a token cut from the user prompt.
            decode_at = rng.randrange(len(prompt_ids), len(tokens))
            blocks = (decode_at + PREFILL - 1) // PREFILL
            capture_block = rng.randrange(blocks)
            # A PREFILL-token graph must fit in the context even when the final block is padded.
            if capture_block * PREFILL + PREFILL > CONTEXT:
                capture_block -= 1
            length_bin = min((len(tokens) - 1) // 512, (CONTEXT - 1) // 512)
            key = (length_bin, min(len(calls), 2))
            buckets[key].append({'row_index': row_index, 'id': row.get('id', row_index),
                'tokens': tokens, 'prompt_tokens': len(prompt_ids), 'decode_at': decode_at,
                'prefill_capture_block': capture_block, 'tool_count': len(tools),
                'reference_call_count': len(calls), 'length_bucket': length_bin})
        except (KeyError, ValueError, TypeError):
            rejected['format'] += 1
        if sum(map(len, buckets.values())) >= 2048:
            break
    chosen = []
    keys = sorted(buckets)
    while len(chosen) < COUNT:
        progressed = False
        for key in keys:
            if buckets[key]:
                chosen.append(buckets[key].pop())
                progressed = True
                if len(chosen) == COUNT:
                    break
        if not progressed:
            raise RuntimeError('Not enough complete conversations fit the static context')
    rng.shuffle(chosen)
    lengths = sorted(len(item['tokens']) for item in chosen)
    provenance = {'dataset': dataset['repo_id'], 'revision': dataset['revision'],
        'source_file_sha256': dataset['sha256'], 'license': dataset['license'],
        'source_url': 'https://huggingface.co/datasets/' + dataset['repo_id'],
        'citation': 'APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets (2024), arXiv:2406.18518',
        'seed': SEED, 'example_count': COUNT, 'source_row_count': len(rows),
        'selection': 'Seeded shuffle; equal-turn selection across available 512-token length buckets and single/multiple reference-call groups; complete examples only',
        'rendering': 'Pinned model chat template, normalized function descriptions, native assistant tool_calls, non-thinking mode',
        'capture': f'One selected {PREFILL}-token prefill input and one teacher-forced reference-answer decode input per example per applicable graph; output head captured at decode',
        'row_indices': [item['row_index'] for item in chosen],
        'example_ids': [item['id'] for item in chosen],
        'complete_conversation_tokens': {'minimum': lengths[0], 'median': lengths[len(lengths)//2], 'maximum': lengths[-1]},
        'length_bucket_counts': dict(Counter(str(item['length_bucket']) for item in chosen)),
        'reference_call_count_distribution': dict(Counter(str(item['reference_call_count']) for item in chosen)),
        'rejected_candidates': dict(rejected), 'raw_examples_in_public_package': False,
        'fine_tuning_performed': False}
    save(SELECTED, chosen)
    from .prepared import digest
    provenance.update(selection_identity=selection_identity(), selected_examples_sha256=digest(SELECTED))
    save(OUT / 'calibration-provenance.json', provenance)
    print(json.dumps(provenance, indent=2), flush=True)


def implementation():
    import importlib
    import re
    if not re.fullmatch(r'[a-z][a-z0-9_]*', MODEL['adapter']):
        raise ValueError('Invalid adapter module name')
    return importlib.import_module(f'.adapters.{MODEL["adapter"]}', package=__package__)


def capture(limit=None, reset=False):
    validate_selection()
    native = implementation()
    torch, np = native.torch, native.np
    torch.set_num_threads(6)
    torch.manual_seed(0)
    pipeline = native.load_pipeline()
    native.check_reference(pipeline)
    # Calibration executes the same established float graphs that QAIRT consumes.
    # ORT avoids the per-head Python/Conv dispatch cost during long CPU prefixes.
    import onnxruntime as ort
    original_manifest = read(OUT / 'float-manifest.json')
    from .prepared import digest
    capture_signature = hashlib.sha256(json.dumps({
        'selection': digest(SELECTED), 'model_recipe': MODEL,
        'sources': {key: {file.name: digest(file) for file in
            [ROOT / graph['directory'] / 'model.onnx', *sorted((ROOT / graph['directory']).glob('*.data'))]}
            for key, graph in original_manifest['graphs'].items()}
    }, sort_keys=True).encode()).hexdigest()
    sessions = {}
    for name, part in pipeline.parts:
        input_names, _ = pipeline.names(part)
        for seq in ([1] if name == 'lm_head' else [1, PREFILL]):
            graph = original_manifest['graphs'][f'{name}_s{seq}']
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            sessions[name, seq] = ort.InferenceSession(
                str(ROOT / graph['directory'] / 'model.onnx'), sess_options=options,
                providers=['CPUExecutionProvider'])

        def forward_factory(part_name, names):
            def forward(*arguments):
                seq = int(arguments[0].shape[1])
                values = {name: value.detach().cpu().numpy() for name, value in zip(names, arguments, strict=True)}
                outputs = sessions[part_name, seq].run(None, values)
                result = tuple(torch.from_numpy(value) for value in outputs)
                return result[0] if len(result) == 1 else result
            return forward

        part.forward = forward_factory(name, input_names)
    native.check_reference(pipeline)
    provenance = read(OUT / 'calibration-provenance.json')
    provenance['capture_backend'] = 'ONNX Runtime CPU using validated FP32 float exports; checked against the source reference implementation'
    save(OUT / 'calibration-provenance.json', provenance)
    print('All float ONNX sessions loaded; reference equivalence passed.', flush=True)
    chosen = read(SELECTED)
    progress_path = OUT / 'calibration-progress.json'
    previous = read(progress_path) if progress_path.exists() and not reset else None
    if previous and previous.get('capture_signature') != capture_signature:
        raise RuntimeError('Capture inputs changed; use a fresh work directory or explicitly reset capture')
    done = previous['completed_examples'] if previous else 0
    stop = min(limit or COUNT, COUNT)
    expected_keys = set(original_manifest['graphs'])
    for index in range(done, stop):
        example = chosen[index]
        pipeline.reset()
        captured = set()

        def record(phase):
            def callback(name, seq, values):
                if phase == 'prefill' and name == 'lm_head':
                    return
                key = f'{name}_s{seq}'
                if key in captured:
                    raise RuntimeError('Duplicate per-example graph capture')
                directory = OUT / 'calibration' / key
                directory.mkdir(parents=True, exist_ok=True)
                arrays = {name: value.detach().cpu().numpy() for name, value in values.items()}
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise RuntimeError('Non-finite calibration inputs')
                target = directory / f'{index:03d}.npz'
                temporary = target.with_suffix('.pending')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(target)
                captured.add(key)
            return callback

        prefix = example['tokens'][:example['decode_at']]
        for block, start in enumerate(range(0, len(prefix), PREFILL)):
            pipeline.run(prefix[start:start+PREFILL], PREFILL,
                record('prefill') if block == example['prefill_capture_block'] else None)
        pipeline.run([example['tokens'][example['decode_at']]], 1, record('decode'))
        if captured != expected_keys:
            raise RuntimeError(f'Incomplete per-example calibration: {captured ^ expected_keys}')
        save(progress_path, {'completed_examples': index + 1, 'target_examples': COUNT,
                            'capture_signature': capture_signature,
                            'last_example_id': example['id'], 'status': 'CAPTURING'})
        print(f'Captured example {index + 1}/{COUNT}; prefix={len(prefix)} tokens; id={example["id"]}', flush=True)
    if stop < COUNT:
        return
    counts = {}
    for key in sorted(expected_keys):
        files = sorted((OUT / 'calibration' / key).glob('*.npz'))
        if [file.name for file in files] != [f'{i:03d}.npz' for i in range(COUNT)]:
            raise RuntimeError('Missing or stale calibration sample files')
        counts[key] = len(files)
    save(OUT / 'calibration-summary.json', counts)
    manifest = original_manifest
    manifest.update(calibration_profile=MODEL['calibration']['profile'],
                    calibration_provenance='calibration-provenance.json')
    for key, graph in manifest['graphs'].items():
        graph['calibration_directory'] = (OUT / 'calibration' / key).relative_to(ROOT).as_posix()
    save(OUT / 'float-manifest.json', manifest)
    save(progress_path, {'completed_examples': COUNT, 'target_examples': COUNT,
                        'capture_signature': capture_signature, 'status': 'COMPLETE'})


def validate():
    native = implementation()
    import onnxruntime as ort
    import numpy as np
    torch = native.torch
    torch.set_num_threads(6)
    pipeline = native.load_pipeline()
    manifest = read(OUT / 'float-manifest.json')
    validations = {}
    for name, part in pipeline.parts:
        for seq in ([1] if name == 'lm_head' else [1, PREFILL]):
            key = f'{name}_s{seq}'
            graph = manifest['graphs'][key]
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session = ort.InferenceSession(str(ROOT / graph['directory'] / 'model.onnx'),
                      sess_options=options, providers=['CPUExecutionProvider'])
            metrics = []
            for index in (0, COUNT-1):
                with np.load(OUT / 'calibration' / key / f'{index:03d}.npz', allow_pickle=False) as data:
                    values = {name: data[name].copy() for name in graph['inputs']}
                if any(list(values[name].shape) != spec['shape'] or str(values[name].dtype) != spec['dtype']
                       for name, spec in graph['inputs'].items()):
                    raise RuntimeError('Calibration tensor contract mismatch')
                with torch.no_grad():
                    expected = part(*(torch.from_numpy(values[name]) for name in graph['inputs']))
                if isinstance(expected, torch.Tensor):
                    expected = (expected,)
                actual = session.run(None, values)
                for output_name, reference, got in zip(graph['outputs'], expected, actual, strict=True):
                    reference = reference.numpy()
                    if not np.isfinite(got).all() or not np.allclose(reference, got, atol=.01, rtol=.002):
                        raise RuntimeError(f'xLAM ONNX comparison failed: {key}/{output_name}')
                    metrics.append({'sample': index, 'output': output_name,
                                    'max_absolute_error': float(np.abs(reference - got).max())})
            validations[key] = metrics
            print(f'Validated xLAM tensor inputs and float outputs: {key}', flush=True)
            del session
    save(OUT / 'xlam-float-validation.json', validations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('select', 'capture', 'validate', 'all'), default='all')
    parser.add_argument('--limit', type=int, help='Capture a small initial subset for preparation diagnostics.')
    parser.add_argument('--reset-capture', action='store_true', help='Restart captured samples using the pinned selection.')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ('select', 'all'):
        select()
    if args.stage in ('capture', 'all'):
        capture(args.limit, args.reset_capture)
    if args.stage in ('validate', 'all') and args.limit is None:
        validate()
