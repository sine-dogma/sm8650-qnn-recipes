# SPDX-License-Identifier: CC0-1.0
"""Convert five xLAM-calibrated QNN variants using local QAIRT tools only."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .settings import (ROOT, BASE, BATCH, SDK, BIN, LIB, MODEL, SAMPLES, TARGET,
                       HOST, EXPORT_PYTHON, sdk_environment, export_environment, variant_dir)
from .locking import workflow_lock, keep_awake
from .prepared import verify_prepared
VARIANTS = ('w4a16', 'w4a8', 'w8a8', 'w8a16', 'w16a16')
STATUS = {'model': MODEL['repo_id'], 'calibration_profile': MODEL['calibration']['profile'],
          'build_origin': 'local_qairt', 'uploads_performed': False,
          'status': 'RUNNING', 'phone_inference_tested': False, 'speed_tested': False,
          'variants': {}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def update(precision=None, **values):
    target = STATUS if precision is None else STATUS['variants'].setdefault(precision, {})
    target.update(values)
    if precision is not None:
        STATUS['active_precision'] = precision
        if 'stage' in values:
            STATUS['stage'] = values['stage']
    if values.get('status') in ('RUNNING', 'COMPLETE'):
        target.pop('error', None)
    STATUS['updated_utc'] = stamp()
    write_json(BATCH / 'workflow-status.json', STATUS)


def environment():
    return sdk_environment()


def preflight():
    verify_prepared()
    source = read_json(BASE / 'float-manifest.json')
    counts = read_json(BASE / 'calibration-summary.json')
    checks = read_json(BASE / 'xlam-float-validation.json')
    if (source.get('model') != MODEL['repo_id']
            or source.get('calibration_profile') != MODEL['calibration']['profile']
            or len(source['graphs']) != MODEL['expected_graphs'] or set(counts) != set(source['graphs'])
            or set(checks) != set(source['graphs']) or set(counts.values()) != {SAMPLES}):
        raise RuntimeError(f'The validated {SAMPLES}-example preparation is required')
    raw_bytes = 0
    for key, graph in source['graphs'].items():
        if not graph.get('validated') or not (ROOT / graph['directory'] / 'model.onnx').is_file():
            raise RuntimeError(f'Validated float source missing: {key}')
        files = sorted((ROOT / graph['calibration_directory']).glob('*.npz'))
        if len(files) != SAMPLES:
            raise RuntimeError(f'Expected exactly {SAMPLES} calibration samples: {key}')
        raw_bytes += SAMPLES * sum(math.prod(spec['shape']) * np.dtype(spec['dtype']).itemsize
                              for spec in graph['inputs'].values())
    print(f'Local preflight: {len(source["graphs"])} float graphs, {SAMPLES} samples each; raw calibration {raw_bytes / 1e9:.3f} GB', flush=True)
    # Staging and outputs are resumable. Allow substantial space beyond raw data.
    staged_bytes = sum(path.stat().st_size for path in (BATCH / 'calibration').rglob('*.raw'))
    if shutil.disk_usage(ROOT).free < max(0, raw_bytes - staged_bytes) + 30 * 1024**3:
        raise RuntimeError('Insufficient free storage for remaining raw calibration and model outputs')
    return source


def prepare_variant(precision, source):
    out = variant_dir(precision)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(json.dumps(source))
    manifest.update(requested_precision=precision, build_origin='local_qairt')
    for graph in manifest['graphs'].values():
        for key in list(graph):
            if key.startswith('hub_'):
                del graph[key]
    write_json(out / 'float-manifest.json', manifest)
    for name in ('calibration-summary.json', 'calibration-provenance.json',
                 'xlam-float-validation.json', 'float-reference-check.json'):
        shutil.copyfile(BASE / name, out / name)
    state_path = out / 'local-build.json'
    state = read_json(state_path) if state_path.exists() else {
        'model': MODEL['repo_id'], 'precision': precision,
        'build_origin': 'local_qairt', 'uploads_performed': False,
        'compile': {}, 'link': {}, 'phone_inference_tested': False}
    if state.get('precision') != precision or state.get('build_origin') != 'local_qairt':
        raise RuntimeError('Incompatible saved local build')
    write_json(state_path, state)
    return out, manifest, state


def calibration(key, graph):
    target = BATCH / 'calibration' / key
    target.mkdir(parents=True, exist_ok=True)
    source_files = sorted((ROOT / graph['calibration_directory']).glob('*.npz'))
    source_info = [{ 'file': path.name, 'bytes': path.stat().st_size,
                     'sha256': digest(path)} for path in source_files]
    signature_data = {'graph': key, 'inputs': graph['inputs'], 'samples': source_info,
                      'provenance_sha256': digest(BASE / 'calibration-provenance.json')}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    marker = target / 'calibration.json'
    input_list = target / 'input-list.txt'
    if marker.exists():
        saved = read_json(marker)
        valid = (saved.get('signature') == signature and input_list.is_file()
                 and saved.get('raw_directory') == str(target.resolve()))
        if valid:
            valid = digest(input_list) == saved.get('input_list_sha256') and all(
                (target / item['file']).is_file()
                and (target / item['file']).stat().st_size == item['bytes']
                and digest(target / item['file']) == item['sha256']
                for item in saved.get('raw_files', []))
        if valid and len(saved.get('raw_files', [])) == SAMPLES * len(graph['inputs']):
            return input_list, signature
        if saved.get('signature') != signature:
            raise RuntimeError(f'The source calibration changed: {key}')
    print(f'Staging local raw calibration: {key}', flush=True)
    rows, raw_files = [], []
    for index, filename in enumerate(source_files):
        directory = target / f'{index:03d}'
        directory.mkdir(exist_ok=True)
        fields = []
        with np.load(filename, allow_pickle=False) as sample:
            if set(sample.files) != set(graph['inputs']):
                raise RuntimeError(f'Calibration input names differ: {key}/{filename.name}')
            for name, spec in graph['inputs'].items():
                array = sample[name]
                if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
                    raise RuntimeError(f'Calibration tensor contract differs: {key}/{name}')
                if not np.isfinite(array).all():
                    raise RuntimeError(f'Nonfinite calibration values: {key}/{name}')
                path = directory / f'{name}.raw'
                pending = path.with_suffix('.raw.pending')
                np.ascontiguousarray(array).tofile(pending)
                pending.replace(path)
                fields.append(f'{name}:={path}')
                raw_files.append({'file': path.relative_to(target).as_posix(), 'bytes': array.nbytes, 'sha256': digest(path)})
        rows.append(' '.join(fields))
    pending = input_list.with_suffix('.pending')
    pending.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    pending.replace(input_list)
    write_json(marker, {'signature': signature, 'raw_directory': str(target.resolve()),
                       'sample_count': SAMPLES, 'inputs': graph['inputs'],
                       'input_list_sha256': digest(input_list), 'raw_files': raw_files,
                       'source': signature_data, 'completed_utc': stamp()})
    return input_list, signature


def record_file(path, **values):
    return {'status': 'SUCCESS', 'file': path.relative_to(ROOT).as_posix(),
            'sha256': digest(path), 'bytes': path.stat().st_size, **values}


def compile_graph(precision, key, graph, out, state, env):
    from .local_tools import local_convert, local_quantize, recipe_metadata, converter_recipe_metadata
    update(precision, status='RUNNING', stage='calibration_staging', graph=key)
    input_list, signature = calibration(key, graph)
    update(precision, stage='float_conversion', graph=key)
    floating = local_convert(key, graph, BATCH / 'float-dlc' / key, env)
    update(precision, stage='float_validation', graph=key)
    validation_env = export_environment()
    stage('validate_float', ['--graphs', key],
          BATCH / 'float-validation-logs' / f'{key}.log', validation_env)
    update(precision, stage='quantization', graph=key)
    output = local_quantize(precision, key, floating, input_list, out / 'dlc', env,
                            calibration_signature=signature)
    state['compile'][key] = record_file(output, calibration_signature=signature,
        options={**recipe_metadata(precision), 'calibration_samples': SAMPLES},
        converter_options=converter_recipe_metadata(), source_float_sha256=digest(floating))
    write_json(out / 'local-build.json', state)
    print(f'{precision}: quantized {key}', flush=True)


def link_part(precision, part, manifest, out, state, env):
    keys = sorted(key for key, graph in manifest['graphs'].items() if graph['part'] == part)
    if not keys or not all(state['compile'].get(key, {}).get('status') == 'SUCCESS' for key in keys):
        raise RuntimeError(f'Quantized graphs required before linking: {part}')
    options = {name: TARGET[name] for name in ('optimization', 'weight_sharing_enabled',
               'short_depth_conv_on_hmx_off', 'fp16_relaxed_precision', 'vtcm_mb', 'soc_model', 'dsp_arch')}
    options.update(graph_names=keys, sdk_build_id=TARGET['qairt_build_id'], host=HOST.tag)
    sources = {key: state['compile'][key]['sha256'] for key in keys}
    binary = out / 'context' / f'{part}.bin'
    entry = state['link'].get(part)
    if (entry and entry.get('status') == 'SUCCESS' and entry.get('source_dlc_sha256') == sources
            and entry.get('options') == options and binary.is_file()
            and digest(binary) == entry.get('sha256')):
        print(f'{precision}: reused local context {part}', flush=True)
        return
    update(precision, status='RUNNING', stage='context_generation', part=part)
    config_dir = out / 'link-config' / part
    config_dir.mkdir(parents=True, exist_ok=True)
    htp = config_dir / 'htp.json'
    write_json(htp, {'context': {'weight_sharing_enabled': options['weight_sharing_enabled']}, 'graphs': [{
        'graph_names': keys, 'O': options['optimization'], 'vtcm_mb': options['vtcm_mb'],
        'fp16_relaxed_precision': options['fp16_relaxed_precision'],
        'short_depth_conv_on_hmx_off': options['short_depth_conv_on_hmx_off']}],
        'devices': [{'device_id': 0, 'soc_model': TARGET['soc_model'], 'dsp_arch': f"v{TARGET['dsp_arch']}"}]})
    config = config_dir / 'backend.json'
    write_json(config, {'backend_extensions': {
        'shared_library_path': str(HOST.library('QnnHtpNetRunExtensions')),
        'config_file_path': str(htp)}})
    binary.parent.mkdir(exist_ok=True)
    pending_dir = binary.parent / '.work' / part
    pending_dir.mkdir(parents=True, exist_ok=True)
    pending_binary = pending_dir / binary.name
    pending_binary.unlink(missing_ok=True)
    command = [str(HOST.executable('qnn-context-binary-generator')), '--backend', str(HOST.library('QnnHtp')),
               '--model', str(HOST.library('QnnModelDlc')), '--dlc_path',
               ','.join(str(out / 'dlc' / f'{key}.dlc') for key in keys),
               '--config_file', str(config), '--output_dir', str(pending_dir), '--binary_file', part]
    logfile = config_dir / 'link.log'
    print(f'{precision}: generating local context {part}', flush=True)
    with logfile.open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not pending_binary.is_file() or pending_binary.stat().st_size == 0:
        raise RuntimeError(f'Context generation failed ({result.returncode}): {logfile.relative_to(ROOT)}')
    pending_binary.replace(binary)
    state['link'][part] = record_file(binary, options=options, source_dlc_sha256=sources)
    write_json(out / 'local-build.json', state)
    print(f'{precision}: context ready {part}', flush=True)


def stage(script, arguments, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'\nStarting at {stamp()}\n')
        stream.flush()
        result = subprocess.run([EXPORT_PYTHON, '-u', '-m', f'sm8650_qnn.{script}', *arguments],
                                cwd=ROOT, env=export_environment(), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} failed ({result.returncode}); see {log.relative_to(ROOT)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    BATCH.mkdir(parents=True, exist_ok=True)
    with workflow_lock(BATCH / 'workflow.lock'), keep_awake():
        previous = BATCH / 'workflow-status.json'
        if previous.exists():
            STATUS['variants'] = read_json(previous).get('variants', {})
        STATUS['status'] = 'RUNNING'
        update(status='RUNNING', stage='preflight')
        try:
            source, env = preflight(), environment()
            # Validate the model-selected decoder pilot before the broader run.
            out, manifest, state = prepare_variant('w4a16', source)
            for key, graph in manifest['graphs'].items():
                if graph['part'] == MODEL['pilot_part']:
                    compile_graph('w4a16', key, graph, out, state, env)
            link_part('w4a16', MODEL['pilot_part'], manifest, out, state, env)
            update('w4a16', stage='pilot_inspection')
            stage('inspect_native', ['--precision', 'w4a16', '--parts', MODEL['pilot_part']],
                  out / 'workflow-logs/pilot-inspect.log', env)
            write_json(BATCH / 'pilot-status.json', {'status': 'PASSED', 'completed_utc': stamp(),
                       'precision': 'w4a16', 'part': MODEL['pilot_part'], 'uploads_performed': False})
            print('W4A16 decoder pilot passed native inspection', flush=True)
            if args.pilot_only:
                update(status='PILOT_PASSED', stage='pilot_complete')
                return
            for precision in args.variants:
                out, manifest, state = prepare_variant(precision, source)
                parts = sorted({graph['part'] for graph in manifest['graphs'].values()})
                for part in parts:
                    for key, graph in manifest['graphs'].items():
                        if graph['part'] == part:
                            compile_graph(precision, key, graph, out, state, env)
                    link_part(precision, part, manifest, out, state, env)
                update(precision, stage='inspection')
                stage('inspect_native', ['--precision', precision],
                      out / 'workflow-logs/inspect.log', env)
                report = read_json(out / 'native-artifact-report.json')
                if not report.get('complete') or len(report['contexts']) != MODEL['expected_contexts'] or len(report['precision']) != MODEL['expected_graphs']:
                    raise RuntimeError(f'Native inspection incomplete: {precision}')
                update(precision, status='COMPLETE', stage='native_artifacts_complete',
                       contexts=MODEL['expected_contexts'], graphs=MODEL['expected_graphs'], total_context_bytes=report['total_context_bytes'])
            update(status='COMPLETE', stage='selected_variants_complete')
        except BaseException as error:
            update(status='PARTIAL_FAILURE', error=str(error))
            raise


if __name__ == '__main__':
    main()
