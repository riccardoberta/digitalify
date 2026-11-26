"""
End-to-end checks over a matrix of model variants:

* verify   — integer reference model == QONNX execution (no simulator needed)
* simulate — generated VHDL == integer reference model (needs GHDL)
"""

import subprocess
import sys

import pytest

import digitalify
from digitalify.testing import make_mlp

VARIANTS = {
    "default": dict(),
    "mixed_precision": dict(layer_sizes=[16, 8, 6, 4], weight_bits=[4, 3, 6], act_bits=[5, 3], output_bits=6),
    "odd_widths": dict(input_bits=3, weight_bits=5, act_bits=7, output_bits=5),
    "unsigned_input": dict(input_signed=False),
    "raw_output": dict(output_quant=False),
    "float_bias": dict(quantized_bias=False),
    "float_bias_raw_output": dict(quantized_bias=False, output_quant=False),
    "no_bias": dict(bias=False),
    "no_relu": dict(relu=False),
    "gemm_per_channel": dict(use_gemm=True, per_channel=True),
    "non_pow2_scales": dict(pow2_scales=False, per_channel=True),
    "multidim_input": dict(input_shape=(2, 2, 4)),
    "single_features": dict(layer_sizes=[1, 3, 1]),
    **{f"rounding_{m.lower()}": dict(rounding_mode=m)
       for m in ("HALF_UP", "HALF_DOWN", "FLOOR", "CEIL", "UP", "DOWN")},
}

# Non-power-of-two scale ratios are approximated with a 32-bit multiplier;
# QONNX itself computes in float32, so ±1 LSB near rounding ties is accepted.
TOLERANCE = {"non_pow2_scales": 1}


def _model(variant):
    kw = dict(VARIANTS[variant])
    sizes = kw.pop("layer_sizes", [16, 8, 4])
    return make_mlp(sizes, seed=7, **kw)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_reference_matches_qonnx(variant):
    result = digitalify.verify(_model(variant), n_vectors=48,
                               tolerance_lsb=TOLERANCE.get(variant, 0))
    assert result.passed, result.summary()


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_vhdl_matches_reference(variant, ghdl):
    result = digitalify.simulate(_model(variant), n_vectors=12, ghdl_path=ghdl)
    assert result.passed, result.summary() + "\n" + result.log[-2000:]
    assert result.n_passed == 12


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_pipelined_vhdl_matches_reference(variant, ghdl):
    result = digitalify.simulate(_model(variant), n_vectors=12, ghdl_path=ghdl, pipeline=True)
    assert result.passed, result.summary() + "\n" + result.log[-2000:]
    assert result.n_passed == 12


@pytest.mark.parametrize("pipeline", ["acc", "mult", "products", "tree", ("tree:2", "acc")])
@pytest.mark.parametrize("variant", ["default", "raw_output", "single_features"])
def test_single_pipeline_stage(variant, pipeline, ghdl):
    result = digitalify.simulate(_model(variant), n_vectors=12, ghdl_path=ghdl, pipeline=pipeline)
    assert result.passed, result.summary() + "\n" + result.log[-2000:]


def test_standalone_relu(relu_first_model, ghdl):
    ir = digitalify.from_qonnx(relu_first_model)
    assert [n.op_type for n in ir.nodes] == ["relu", "linear"]
    assert digitalify.verify(relu_first_model, ir=ir).passed
    result = digitalify.simulate(ir, n_vectors=12, ghdl_path=ghdl)
    assert result.passed, result.log[-2000:]


def test_package_does_not_import_torch():
    code = ("import sys, digitalify, digitalify.testing, digitalify.verify; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
