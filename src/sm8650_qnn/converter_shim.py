# SPDX-License-Identifier: CC0-1.0
"""Run the installed QAIRT converter while retaining primitive RMS math.

The source ONNX adds epsilon inside the normalization square root. The
installed QNN CPU fused RmsNorm instead adds it outside the square root.
Skip that one converter fusion pass, keeping the validated primitive
operations intact. This process-local wrapper never modifies SDK files.
"""
from pathlib import Path
import runpy
import sys

from sm8650_qnn.settings import HOST, MODEL
from qti.aisw.converters.common.converter_ir import op_graph_optimizations as opt


original = opt.OptimizationTranslations.apply_method_to_graph


def apply(method_name, *args, **kwargs):
    if MODEL.get('primitive_rmsnorm') and method_name == opt.MATCH_RMSNORM:
        return None
    return original(method_name, *args, **kwargs)


opt.OptimizationTranslations.apply_method_to_graph = apply
script = HOST.bin / 'qairt-converter'
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name='__main__')
