# SPDX-License-Identifier: CC0-1.0
"""Compare locally converted FP32 DLC outputs against their original ONNX.

Runs one existing calibration row on the host QNN CPU backend only. This is
converter correctness validation, not quantized-model evaluation, phone
inference, or a performance benchmark. No cloud clients or download APIs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import onnx
import onnxruntime as ort

from .settings import (ROOT, BASE, BATCH, SDK, BIN, LIB, SAMPLES, HOST,
                       SDK_PYTHON, sdk_environment)
STATUS = BATCH / 'local-float-validation.json'
MAX_ABSOLUTE = .02
MAX_RMS = .002
VERSION = 2


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def first_inputs(graph_key, graph):
    source_samples = sorted((ROOT / graph['calibration_directory']).glob('*.npz'))
    require(len(source_samples) == SAMPLES, f'Expected all configured source captures for {graph_key}')
    input_list = BATCH / 'calibration' / graph_key / 'input-list.txt'
    rows = [line for line in input_list.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')]
    require(len(rows) == SAMPLES, f'Expected all configured staged rows for {graph_key}')
    fields = {}
    for entry in rows[0].split():
        require(':=' in entry, 'Calibration rows must use named native raw files')
        name, filename = entry.split(':=', 1)
        require(name not in fields, f'Duplicate raw calibration input: {name}')
        fields[name] = Path(filename)
    require(set(fields) == set(graph['inputs']), 'Staged calibration input names differ')
    values = {}
    with np.load(source_samples[0], allow_pickle=False) as source:
        require(set(source.files) == set(graph['inputs']), 'Source calibration input names differ')
        for name, spec in graph['inputs'].items():
            value = source[name].copy()
            require(list(value.shape) == spec['shape'] and str(value.dtype) == spec['dtype'],
                    f'Original input tensor contract differs: {graph_key}/{name}')
            require(np.isfinite(value).all(), f'Nonfinite source input: {graph_key}/{name}')
            raw = fields[name]
            require(raw.is_file() and raw.stat().st_size == value.nbytes,
                    f'Staged input byte count differs: {graph_key}/{name}')
            actual = np.fromfile(raw, dtype=value.dtype).reshape(value.shape)
            require(np.array_equal(value, actual),
                    f'Staged raw data differs from original NPZ: {graph_key}/{name}')
            values[name] = value
    return source_samples[0], rows[0], values


def native_contract(graph_key, graph, dlc, directory, env):
    """Read the real DLC I/O contract, without executing a graph."""
    description = directory / 'float-dlc.json'
    logfile = directory / 'float-dlc-metadata.log'
    metadata_env = sdk_environment()
    command = [SDK_PYTHON, str(BIN / 'qairt-dlc-to-json'),
               '--input_dlc', str(dlc), '--output_json', str(description)]
    with logfile.open('w', encoding='utf-8') as stream:
        process = subprocess.run(command, cwd=directory, env=metadata_env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    require(process.returncode == 0, f'Cannot inspect float DLC I/O contract: {logfile}')
    tensors = read_json(description)['graph']['tensors']
    inputs = {name: value for name, value in tensors.items() if value.get('type') == 0}
    outputs = {name: value for name, value in tensors.items() if value.get('type') == 1}
    require(set(inputs) == set(graph['inputs']), 'Float DLC native input names differ')
    for name, spec in graph['inputs'].items():
        dtype = {'float32': 0x0232, 'int32': 0x0032}[spec['dtype']]
        require(inputs[name]['dims'] == spec['shape'] and inputs[name]['data_type'] == dtype,
                f'Float DLC native input shape/dtype differs: {graph_key}/{name}')
    original = onnx.load(str(ROOT / graph['directory'] / 'model.onnx'), load_external_data=False)
    source_outputs = original.graph.output
    require([tensor.name for tensor in source_outputs] == graph['outputs'],
            'ONNX output inventory differs from manifest')
    mapping = {}
    for index, tensor in enumerate(source_outputs):
        name = tensor.name
        native = next((candidate for candidate in (name, f'output_{index}', f'Output_{index}')
                       if candidate in outputs), None)
        require(native is not None and native not in mapping.values(),
                f'Cannot map native DLC output name: {graph_key}/{name}')
        shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
        require(tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
                and outputs[native]['dims'] == shape
                and outputs[native]['data_type'] == 0x0232,
                f'Float DLC native output shape/dtype differs: {graph_key}/{name}')
        mapping[name] = native
    require(set(mapping.values()) == set(outputs), 'A native DLC output was not mapped')
    return mapping


def validate(graph_key, graph, previous):
    float_directory = BATCH / 'float-dlc' / graph_key
    dlc = float_directory / f'{graph_key}.dlc'
    marker = read_json(float_directory / f'{graph_key}.convert.json')
    require(marker.get('status') == 'COMPLETE' and marker['signature']['graph'] == graph_key,
            f'Local float conversion is incomplete: {graph_key}')
    recipe = marker['signature']['recipe']
    require(recipe.get('tool') == 'qairt-converter' and recipe.get('preserve_io_datatype') == 'all'
            and recipe.get('preserve_onnx_output_order'), 'Unexpected local float conversion recipe')
    require(dlc.stat().st_size == marker['bytes'] and digest(dlc) == marker['sha256'],
            'The local float DLC differs from its successful conversion record')
    capture, row, values = first_inputs(graph_key, graph)
    model_path = ROOT / graph['directory'] / 'model.onnx'
    signature = {'version': VERSION, 'float_dlc_sha256': marker['sha256'],
                 'source_onnx_sha256': digest(model_path),
                 'capture_sha256': digest(capture),
                 'conversion_signature': marker['signature'],
                 'maximum_absolute_tolerance': MAX_ABSOLUTE, 'rms_tolerance': MAX_RMS}
    if previous and previous.get('validated') and previous.get('signature') == signature:
        print(f'Reusing host float-DLC comparison: {graph_key}', flush=True)
        return previous
    directory = BATCH / 'float-validation-runs' / graph_key / uuid.uuid4().hex
    directory.mkdir(parents=True)
    single_row = directory / 'input-list.txt'
    single_row.write_text(row + '\n', encoding='utf-8')
    output_directory = directory / 'outputs'
    log_path = directory / 'qnn-cpu.log'
    env = sdk_environment()
    output_mapping = native_contract(graph_key, graph, dlc, directory, env)
    command = [str(HOST.executable('qnn-net-run')), '--model', str(HOST.library('QnnModelDlc')),
               '--backend', str(HOST.library('QnnCpu')), '--dlc_path', str(dlc),
               '--input_list', str(single_row), '--output_dir', str(output_directory),
               '--num_inferences', '1', '--use_native_input_files', '--log_level', 'info']
    print(f'Host QNN CPU float correctness check: {graph_key}', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        log.write(json.dumps({'command': command}) + '\n')
        log.flush()
        process = subprocess.run(command, cwd=directory, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        print('\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-30:]),
              flush=True)
        raise RuntimeError(f'Host QNN CPU float execution failed ({process.returncode}): {log_path}')
    result_directories = list(output_directory.rglob('Result_0'))
    require(len(result_directories) == 1, 'Expected exactly one graph execution result directory')
    files = {path.stem: path for path in result_directories[0].glob('*.raw')}
    require(len(files) == len(graph['outputs']), 'The native float output count differs')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    output_names = [value.name for value in session.get_outputs()]
    require(output_names == graph['outputs'], 'Original ONNX output order differs from manifest')
    reference = session.run(output_names, values)
    del session
    metrics, used_names = [], set()
    for index, (name, expected) in enumerate(zip(output_names, reference, strict=True)):
        require(expected.dtype == np.float32 and np.isfinite(expected).all(),
                f'Unexpected reference output dtype or values: {graph_key}/{name}')
        native = output_mapping[name]
        require(native is not None and native not in used_names,
                f'Could not map native output: {graph_key}/{name}')
        require(native in files, f'Native output file is missing: {graph_key}/{native}')
        used_names.add(native)
        require(files[native].stat().st_size == expected.nbytes,
                f'Native output shape/byte count differs: {graph_key}/{name}')
        actual = np.fromfile(files[native], dtype=np.float32).reshape(expected.shape)
        require(np.isfinite(actual).all(), f'Nonfinite QNN CPU output: {graph_key}/{name}')
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        maximum = float(np.max(np.abs(difference)))
        rms = float(np.sqrt(np.mean(difference * difference)))
        metrics.append({'sample': 0, 'output': name, 'native_output': native,
                        'shape': list(expected.shape), 'max_absolute_error': maximum,
                        'rms_error': rms})
    require(used_names == set(files), 'A native output was not compared')
    maximum = max(item['max_absolute_error'] for item in metrics)
    rms = max(item['rms_error'] for item in metrics)
    result = {'validated': maximum <= MAX_ABSOLUTE and rms <= MAX_RMS,
              'signature': signature, 'sample_index': 0,
              'backend': 'QNN_CPU', 'reference_backend': 'ONNXRuntime_CPU',
              'max_absolute_error': maximum, 'rms_error': rms, 'outputs': metrics,
              'native_io_shapes_and_dtypes_checked': True,
              'sample': capture.relative_to(ROOT).as_posix(),
              'execution_log': log_path.relative_to(ROOT).as_posix(),
              'cpu_float_dlc_inference_tested': True,
              'phone_inference_tested': False, 'quantized_model_accuracy_tested': False,
              'speed_tested': False,
              'validated_utc': datetime.now(timezone.utc).isoformat()}
    print(f'Float DLC versus ONNX {graph_key}: max_abs={maximum:.8g}, max_rms={rms:.8g}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs', nargs='+', default=['layers_00_03_s1', 'layers_00_03_s64'])
    args = parser.parse_args()
    manifest = read_json(BASE / 'float-manifest.json')
    require(len(set(args.graphs)) == len(args.graphs), 'Duplicate requested graph keys')
    require(set(args.graphs) <= set(manifest['graphs']), 'Unknown requested graph key')
    checks = read_json(STATUS) if STATUS.exists() else {}
    for key in args.graphs:
        result = validate(key, manifest['graphs'][key], checks.get(key))
        checks[key] = result
        write_json(STATUS, checks)
        require(result['validated'], f'Local float DLC numerical comparison failed for {key}')


if __name__ == '__main__':
    main()
