# SPDX-License-Identifier: CC0-1.0
"""Inspect locally built QAIRT DLCs and SM8650 contexts without inference or network I/O."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import onnx

from .settings import (ROOT, BATCH, SDK, BIN, LIB, VARIANTS, MODEL, TARGET,
                       HOST, SDK_PYTHON, sdk_environment, variant_dir)
PUBLIC_OPTION_KEYS = {'optimization', 'O', 'weight_sharing', 'weight_sharing_enabled',
    'short_depth_conv_on_hmx_off', 'advanced_activation_fusion', 'vtcm_mb',
    'fp16_relaxed_precision', 'graph_names', 'soc_model', 'dsp_arch',
    'weights_bitwidth', 'act_bitwidth', 'bias_bitwidth', 'preserve_io_datatype',
    'use_native_input_files', 'restrict_quantization_steps', 'param_quantizer_schema',
    'act_quantizer_schema', 'param_quantizer_calibration', 'act_quantizer_calibration',
    'use_per_channel_quantization', 'pack_4_bit_weights',
    'weight_bits', 'activation_bits', 'bias_bits', 'float_bits', 'native_input_files',
    'parameter_calibration', 'activation_calibration', 'per_channel_quantization',
    'per_row_quantization', 'parameter_schema', 'activation_schema',
    'restricted_quantization_steps', 'precision', 'tool', 'sdk_version', 'version',
    'calibration_samples', 'preserve_onnx_output_order', 'onnx_skip_simplification',
    'io_layout', 'graph_name', 'normalization_lowering', 'disabled_optimizer_passes',
    'sdk_build_id', 'host'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_artifact(entry, expected):
    require(entry.get('status') == 'SUCCESS' and entry.get('file'),
            f'Local artifact has not completed: {expected.name}')
    path = (ROOT / entry['file']).resolve()
    require(path == expected.resolve(), f'Unexpected local artifact location: {expected.name}')
    require(path.is_file() and path.stat().st_size > 0, f'Local artifact is missing: {expected.name}')
    require(digest(path) == entry.get('sha256'), f'Local artifact checksum differs: {expected.name}')
    return path


def quantization(tensor):
    """Read scalar or per-axis encodings; zero means unquantized."""
    widths, fixed = set(), []
    def walk(value):
        if isinstance(value, dict):
            width = value.get('bitwidth', 0)
            if isinstance(width, int) and width > 0:
                widths.add(width)
                fixed.append(value.get('is_fixed_point', True))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(tensor.get('quant_params', {}))
    require(len(widths) <= 1, 'A tensor has inconsistent quantization widths')
    return (next(iter(widths)) if widths else 0), bool(widths) and all(fixed)


def public_options(entry):
    values = entry.get('options', {})
    if not isinstance(values, dict):
        return {'build_origin': 'local_qairt'}
    return {'build_origin': 'local_qairt', **{
        key: value for key, value in values.items() if key in PUBLIC_OPTION_KEYS}}


def normalization_guard(description, key):
    nodes = description['graph'].get('nodes')
    require(isinstance(nodes, (dict, list)) and nodes, f'Operator inventory missing: {key}')
    values = list(nodes.values()) if isinstance(nodes, dict) else nodes
    require(all(isinstance(node, dict) and isinstance(node.get('type'), str) for node in values),
            f'Unexpected DLC operator schema: {key}')
    counts = Counter(node['type'] for node in values)
    rmsnorm_count = sum(count for name, count in counts.items()
                        if 'rmsnorm' in re.sub(r'[^a-z]', '', name.casefold()))
    required = bool(MODEL.get('primitive_rmsnorm')) and not key.startswith('embedding_')
    require(not required or rmsnorm_count == 0,
            f'{key}: fused RmsNorm is present; primitive normalization is required')
    return {'operator_count': len(values), 'operators_by_type': dict(counts),
            'primitive_normalization_guard': {'required': required, 'passed': True,
                'rmsnorm_operator_count': rmsnorm_count,
                'normalization_denominator': ('sqrt(mean(x*x) + epsilon)'
                    if MODEL.get('primitive_rmsnorm') else 'model_defined')}}


def public_float_checks(out, graphs, compile_state, full):
    """Bind host float comparisons to the exact source DLC used for quantization."""
    source = BATCH / 'local-float-validation.json'
    if not source.exists():
        require(not full, 'Full inspection requires all local float-DLC comparisons')
        return {}
    checks = read_json(source)
    result = {}
    for key, graph in graphs.items():
        if key not in checks:
            require(not full, f'Local float-DLC comparison is missing: {key}')
            continue
        check = checks[key]
        require(check.get('validated') is True and check.get('sample_index') == 0
                and check.get('backend') == 'QNN_CPU'
                and check.get('reference_backend') == 'ONNXRuntime_CPU',
                f'Local float-DLC comparison is incomplete: {key}')
        signature = check.get('signature', {})
        float_hash = signature.get('float_dlc_sha256', '')
        require(re.fullmatch(r'[0-9a-f]{64}', float_hash), f'Local float DLC identity is missing: {key}')
        marker = read_json(out / 'dlc' / f'{key}.quantize.json')
        require(marker.get('status') == 'COMPLETE'
                and marker.get('sha256') == compile_state[key]['sha256']
                and marker['signature'].get('float_dlc_sha256') == float_hash,
                f'Quantized DLC did not use the validated float DLC: {key}')
        for name, threshold in (('max_absolute_error', .02), ('rms_error', .002)):
            value = check.get(name)
            require(isinstance(value, (int, float)) and math.isfinite(value)
                    and 0 <= value <= threshold, f'Local float-DLC tolerance failed: {key}/{name}')
        metrics = check.get('outputs', [])
        require({item.get('output') for item in metrics} == set(graph['outputs'])
                and len(metrics) == len(graph['outputs']), f'Local float output comparisons differ: {key}')
        allowed = {'sample', 'output', 'native_output', 'shape', 'max_absolute_error', 'rms_error'}
        result[key] = {name: check[name] for name in (
            'validated', 'sample_index', 'backend', 'reference_backend',
            'max_absolute_error', 'rms_error')}
        result[key].update(float_dlc_sha256=float_hash,
            outputs=[{name: value for name, value in item.items() if name in allowed} for item in metrics],
            cpu_float_dlc_inference_tested=True, phone_inference_tested=False,
            quantized_model_accuracy_tested=False, speed_tested=False,
            maximum_absolute_tolerance=.02, rms_tolerance=.002)
    require(not full or len(result) == MODEL['expected_graphs'], 'Full inspection requires every local float-DLC comparison')
    return result


def inspect(precision, parts=None):
    weights, activations = VARIANTS[precision]
    out = variant_dir(precision)
    manifest = read_json(out / 'float-manifest.json')
    state = read_json(out / 'local-build.json')
    require(manifest['model'] == MODEL['repo_id']
            and manifest['calibration_profile'] == MODEL['calibration']['profile']
            and manifest['requested_precision'] == precision, 'Source manifest differs')
    require(state['precision'] == precision and state.get('build_origin') == 'local_qairt'
            and state.get('uploads_performed') is False, 'Build is not a local-only QAIRT run')
    require(not any(key.startswith('hub_') for graph in manifest['graphs'].values() for key in graph),
            'The local manifest contains cloud reuse instructions')
    encoded = json.dumps(state)
    require(not re.search(r'"(?:job_id|calibration_dataset_id|target_model_id)"\s*:', encoded),
            'Cloud job state is not accepted for a local build')
    all_parts = {graph['part'] for graph in manifest['graphs'].values()}
    selected_parts = set(parts) if parts else all_parts
    require(selected_parts and selected_parts <= all_parts, 'Unknown inspection partition')
    graphs = {key: value for key, value in manifest['graphs'].items() if value['part'] in selected_parts}
    env = sdk_environment()
    directory = out / 'inspection'
    directory.mkdir(parents=True, exist_ok=True)
    def run(name, command):
        with (directory / f'{name}.log').open('w', encoding='utf-8') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    report = {'complete': False, 'selection_complete': False, 'requested_precision': precision,
              'build_origin': 'local_qairt', 'uploads_performed': False,
              'contexts': {}, 'precision': {}, 'calibration_signatures': {},
              'phone_inference_tested': False, 'quantized_model_accuracy_tested': False,
              'speed_tested': False, 'peak_memory_measured': False}
    for part in sorted(selected_parts):
        entry = state['link'].get(part, {})
        binary = checked_artifact(entry, out / 'context' / f'{part}.bin')
        metadata = directory / f'{part}.context.json'
        run(part, [str(HOST.executable('qnn-context-binary-utility')), '--context_binary', str(binary),
                   '--json_file', str(metadata)])
        info = read_json(metadata)['info']
        require((info['backendId'], info['socModel'], info['contextMetadata']['info']['dspArch'])
                == (TARGET['backend_id'], TARGET['soc_model'], TARGET['dsp_arch']), f'Unexpected backend/SoC/architecture in {part}')
        expected = {key for key, graph in graphs.items() if graph['part'] == part}
        actual = {graph['info']['graphName']: graph['info'] for graph in info['graphs']}
        require(set(actual) == expected, f'Graph inventory differs in {part}')
        contracts = {}
        for key, graph in actual.items():
            source = graphs[key]
            inputs = {tensor['info']['name']: tensor['info'] for tensor in graph['graphInputs']}
            outputs = {tensor['info']['name']: tensor['info'] for tensor in graph['graphOutputs']}
            require(set(inputs) == set(source['inputs']), f'Input contract differs: {key}')
            for name, spec in source['inputs'].items():
                require(inputs[name]['dimensions'] == spec['shape'], f'Input shape differs: {key}/{name}')
                dtype = {'float32': 'QNN_DATATYPE_FLOAT_32', 'int32': 'QNN_DATATYPE_INT_32'}[spec['dtype']]
                require(inputs[name]['dataType'] == dtype, f'Input dtype differs: {key}/{name}')
            mapping = {name: name if name in outputs else f'output_{index}'
                       for index, name in enumerate(source['outputs'])}
            require(set(mapping.values()) == set(outputs), f'Output inventory differs: {key}')
            model = onnx.load(str(ROOT / source['directory'] / 'model.onnx'), load_external_data=False)
            for tensor in model.graph.output:
                target = outputs[mapping[tensor.name]]
                shape = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
                require(target['dimensions'] == shape, f'Output shape differs: {key}/{tensor.name}')
                dtype = {onnx.TensorProto.FLOAT: 'QNN_DATATYPE_FLOAT_32',
                         onnx.TensorProto.INT32: 'QNN_DATATYPE_INT_32'}[tensor.type.tensor_type.elem_type]
                require(target['dataType'] == dtype, f'Output dtype differs: {key}/{tensor.name}')
            contracts[key] = {'inputs': inputs, 'outputs': outputs, 'source_to_native_output': mapping}
        report['contexts'][part] = {'file': binary.relative_to(ROOT).as_posix(),
            'bytes': binary.stat().st_size, 'sha256': entry['sha256'], 'qairt_build': info['buildId'],
            'soc_model': TARGET['soc_model'], 'dsp_arch': TARGET['dsp_arch'], 'link_options': public_options(entry), 'graphs': contracts}
    for key in sorted(graphs):
        entry = state['compile'].get(key, {})
        dlc = checked_artifact(entry, out / 'dlc' / f'{key}.dlc')
        signature = entry.get('calibration_signature', '')
        require(re.fullmatch(r'[0-9a-f]{64}', signature), f'Calibration identity missing: {key}')
        report['calibration_signatures'][key] = signature
        description = directory / f'{key}.dlc.json'
        run(key, [SDK_PYTHON, str(BIN / 'qairt-dlc-to-json'), '--input_dlc', str(dlc),
                  '--output_json', str(description)])
        dlc_description = read_json(description)
        operators = normalization_guard(dlc_description, key)
        tensors = dlc_description['graph']['tensors']
        constants, activation_counts = Counter(), Counter()
        for tensor in tensors.values():
            width, _ = quantization(tensor)
            if tensor.get('type') == 4:
                constants[str(width)] += math.prod(tensor.get('dims', []))
            else:
                activation_counts[str(width)] += 1
        static = [tensor for tensor in tensors.values() if tensor.get('type') == 4]
        require(static, f'No static weights found: {key}')
        largest = max(static, key=lambda tensor: math.prod(tensor.get('dims', [])))
        width, fixed = quantization(largest)
        exceptions = []
        if key.startswith('embedding_') and width != weights:
            exceptions.append('Embedding lookup table uses a different inspected quantization width.')
        if constants.get('0'):
            exceptions.append('Some static tensors remain unquantized.')
        if any(int(value) not in (0, activations) for value in activation_counts):
            exceptions.append('Some nonconstant tensors use other quantization widths.')
        report['precision'][key] = {
            'constant_elements_by_quantized_bitwidth': dict(constants),
            'activation_tensor_count_by_quantized_bitwidth': dict(activation_counts),
            'largest_constant_quantized_bitwidth': width, 'largest_constant_is_fixed_point': fixed,
            'zero_bitwidth_meaning': 'unquantized tensor; see DLC JSON data_type',
            'mixed_precision_notes': exceptions, 'quantization_options': public_options(entry),
            'conversion_options': public_options({'options': entry.get('converter_options', {})}),
            **operators}
        if not key.startswith('embedding_'):
            require(width == weights and fixed,
                    f'{key}: main matrix is not fixed-point {weights}-bit quantized')
            require(activation_counts.get(str(activations)),
                    f'{key}: requested {activations}-bit activation quantization is absent')
    report['local_float_dlc_validation'] = public_float_checks(
        out, graphs, state['compile'], selected_parts == all_parts)
    report['local_float_dlc_validation_count'] = len(report['local_float_dlc_validation'])
    report['selection_complete'] = set(report['contexts']) == selected_parts and set(report['precision']) == set(graphs)
    report['complete'] = report['selection_complete'] and selected_parts == all_parts
    report['total_context_bytes'] = sum(item['bytes'] for item in report['contexts'].values())
    pending = out / 'native-artifact-report.pending'
    pending.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    pending.replace(out / 'native-artifact-report.json')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('contexts', 'precision', 'calibration_signatures',
                                     'local_float_dlc_validation')}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', required=True, choices=VARIANTS)
    parser.add_argument('--parts', nargs='+')
    args = parser.parse_args()
    inspect(args.precision, args.parts)
