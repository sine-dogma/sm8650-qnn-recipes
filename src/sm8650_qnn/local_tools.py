# SPDX-License-Identifier: CC0-1.0
"""Local QAIRT conversion and calibrated quantization; no service clients.

The caller stages raw calibration inputs and controls scheduling. Success
markers bind the inputs, recipe and actual output bytes, so interrupted tool
runs cannot be mistaken for reusable converted models. Markers and command
logs are private build records; ``recipe_metadata`` contains public-safe data.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .settings import (ROOT, SDK, VARIANTS, MODEL, SAMPLES, SDK_VERSION, TARGET,
                       SDK_PYTHON, HOST, RECIPES)
RECIPE_VERSION = 3


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def _directory(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Local QAIRT outputs must be inside the workspace')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _key(value):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', value):
        raise ValueError('Invalid graph key')
    return value


def converter_recipe_metadata():
    return {'version': RECIPE_VERSION, 'tool': 'qairt-converter',
            'sdk_version': SDK_VERSION, 'sdk_build_id': TARGET['qairt_build_id'],
            'host': HOST.tag, 'preserve_io_datatype': 'all',
            'preserve_onnx_output_order': True, 'onnx_skip_simplification': True,
            'io_layout': 'original_model_layout', 'graph_name': 'output_file_stem',
            'normalization_lowering': ('primitive_ops_epsilon_inside_sqrt' if MODEL.get('primitive_rmsnorm') else 'sdk_default'),
            'disabled_optimizer_passes': ['MATCH_RMSNORM'] if MODEL.get('primitive_rmsnorm') else []}


def recipe_metadata(precision):
    precision = precision.lower()
    if precision not in RECIPES:
        raise ValueError(f'Unsupported local precision: {precision}')
    return {'version': RECIPE_VERSION, 'tool': 'qairt-quantizer',
            'sdk_version': SDK_VERSION, 'sdk_build_id': TARGET['qairt_build_id'],
            'host': HOST.tag, **RECIPES[precision]}


def _reusable(marker, signature, output):
    if not marker.exists() or not output.exists() or output.stat().st_size == 0:
        return False
    try:
        record = json.loads(marker.read_text(encoding='utf-8'))
        return (record.get('status') == 'COMPLETE' and record.get('signature') == signature
                and record.get('bytes') == output.stat().st_size
                and record.get('sha256') == digest(output))
    except (OSError, ValueError):
        return False


def _run(command, env, logfile, cwd):
    name = command[2] if command[1] == '-m' else Path(command[1]).name
    print(f'Local QAIRT: {name} ({logfile.name})', flush=True)
    with logfile.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps({'command': command}) + '\n')
        stream.flush()
        result = subprocess.run(command, env=env, cwd=cwd,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = logfile.read_text(encoding='utf-8', errors='replace').splitlines()[-30:]
        print('\n'.join(tail), flush=True)
        raise RuntimeError(f'Local QAIRT exited {result.returncode}; see {logfile}')


def _complete(marker, signature, output, command, log):
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError('QAIRT reported success without producing a nonempty DLC')
    _write_json(marker, {'status': 'COMPLETE', 'signature': signature,
                        'bytes': output.stat().st_size, 'sha256': digest(output),
                        'file': output.relative_to(ROOT).as_posix(),
                        'command': command, 'log': log.relative_to(ROOT).as_posix(),
                        'completed_utc': datetime.now(timezone.utc).isoformat()})


def local_convert(graph_key, graph_spec, target_directory, env):
    """Convert one validated local ONNX graph and return its float DLC Path."""
    graph_key = _key(graph_key)
    if not graph_spec.get('validated'):
        raise ValueError(f'Float graph has not been validated: {graph_key}')
    source = (ROOT / graph_spec['directory']).resolve()
    if not source.is_relative_to(ROOT) or not (source / 'model.onnx').is_file():
        raise ValueError('The validated local ONNX graph is unavailable')
    directory = _directory(target_directory)
    output = directory / f'{graph_key}.dlc'
    marker = directory / f'{graph_key}.convert.json'
    log = directory / f'{graph_key}.convert.log'
    source_files = [source / 'model.onnx', *sorted(source.glob('*.data'))]
    signature = {'graph': graph_key, 'source_files': {
                    path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                    for path in source_files},
                 'inputs': graph_spec['inputs'], 'outputs': graph_spec['outputs'],
                 'recipe': converter_recipe_metadata()}
    if _reusable(marker, signature, output):
        print(f'Reusing local float DLC: {graph_key}', flush=True)
        return output
    work = _directory(directory / '.work' / graph_key)
    pending_output = work / f'{graph_key}.dlc'
    pending_output.unlink(missing_ok=True)
    command = [SDK_PYTHON, '-m', 'sm8650_qnn.converter_shim',
               '--input_network', str(source / 'model.onnx'),
               '--output_path', str(pending_output), '--preserve_io_datatype',
               '--preserve_onnx_output_order', '--onnx_skip_simplification']
    _run(command, dict(env), log, work)
    if not pending_output.is_file() or pending_output.stat().st_size == 0:
        raise RuntimeError(f'Local float DLC was not produced: {graph_key}')
    pending_output.replace(output)
    _complete(marker, signature, output, command, log)
    return output


def local_quantize(precision, graph_key, float_dlc, raw_input_list,
                   output_directory, env, *, calibration_signature):
    """Calibrate one graph entirely locally and return the quantized DLC Path.

    The required calibration signature binds all staged raw tensors and the
    source capture hashes. A changed tensor cannot reuse a previous result.
    """
    graph_key = _key(graph_key)
    recipe = recipe_metadata(precision)
    if not re.fullmatch(r'[0-9a-f]{64}', calibration_signature):
        raise ValueError('A SHA-256 calibration identity is required')
    float_dlc = Path(float_dlc).resolve()
    raw_input_list = Path(raw_input_list).resolve()
    if not float_dlc.is_file() or not raw_input_list.is_file():
        raise ValueError('Local float DLC and raw calibration input-list are required')
    rows = [line for line in raw_input_list.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')]
    if len(rows) != SAMPLES:
        raise ValueError(f'The complete {SAMPLES}-example calibration is required')
    directory = _directory(output_directory)
    output = directory / f'{graph_key}.dlc'
    if output == float_dlc:
        raise ValueError('Quantization must preserve the source float DLC')
    marker = directory / f'{graph_key}.quantize.json'
    log = directory / f'{graph_key}.quantize.log'
    signature = {'graph': graph_key, 'float_dlc_sha256': digest(float_dlc),
                 'input_list_sha256': digest(raw_input_list), 'samples': len(rows),
                 'calibration_signature': calibration_signature, 'recipe': recipe}
    if _reusable(marker, signature, output):
        print(f'Reusing local {precision.upper()} DLC: {graph_key}', flush=True)
        return output
    work = _directory(directory / '.work' / graph_key)
    pending_output = work / f'{graph_key}.dlc'
    pending_output.unlink(missing_ok=True)
    command = [SDK_PYTHON, str(HOST.bin / 'qairt-quantizer'),
               '--input_dlc', str(float_dlc), '--output_dlc', str(pending_output),
               '--input_list', str(raw_input_list), '--weights_bitwidth', str(recipe['weight_bits']),
               '--act_bitwidth', str(recipe['activation_bits']), '--bias_bitwidth', str(recipe['bias_bits']),
               '--float_bitwidth', str(recipe['float_bits']), '--preserve_io_datatype', '--use_native_input_files',
               '--param_quantizer_calibration', recipe['parameter_calibration'],
               '--act_quantizer_calibration', recipe['activation_calibration']]
    if recipe['per_channel_quantization']:
        command += ['--use_per_channel_quantization']
    if recipe['per_row_quantization']:
        command += ['--use_per_row_quantization']
    if recipe['parameter_schema'] != 'sdk_default':
        command += ['--param_quantizer_schema', recipe['parameter_schema']]
    if recipe['activation_schema'] != 'sdk_default':
        command += ['--act_quantizer_schema', recipe['activation_schema']]
    if recipe.get('restricted_quantization_steps'):
        command += ['--restrict_quantization_steps', ' '.join(recipe['restricted_quantization_steps'])]
    _run(command, dict(env), log, work)
    if not pending_output.is_file() or pending_output.stat().st_size == 0:
        raise RuntimeError(f'Local quantized DLC was not produced: {graph_key}')
    pending_output.replace(output)
    _complete(marker, signature, output, command, log)
    return output


def inspect_dlc(dlc, output_json, env):
    """Export local DLC metadata without inference or network access."""
    dlc, output_json = Path(dlc).resolve(), Path(output_json).resolve()
    _directory(output_json.parent)
    command = [SDK_PYTHON, str(HOST.bin / 'qairt-dlc-to-json'),
               '--input_dlc', str(dlc), '--output_json', str(output_json)]
    _run(command, dict(env), output_json.with_suffix('.log'), output_json.parent)
    return json.loads(output_json.read_text(encoding='utf-8'))
