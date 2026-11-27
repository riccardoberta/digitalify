"""
Threshold activations: bias + [ReLU] + Quant after a linear / conv layer as
per-channel integer thresholds on the accumulator.
"""

from fractions import Fraction
import random

import numpy as np
import pytest

import digitalify
from digitalify.quant import ROUNDING_MODES, quant_thresholds, round_fraction, shift_round
from digitalify.testing import make_cnn, make_mlp


@pytest.mark.parametrize("mode", ROUNDING_MODES)
def test_round_fraction_matches_shift_round(mode):
    for p in range(-40, 41):
        for n in range(4):
            assert round_fraction(Fraction(p, 1 << n), mode) == shift_round(p, n, mode)


@pytest.mark.parametrize("mode", ROUNDING_MODES)
def test_thresholds_reproduce_the_quantiser(mode):
    """q(S) == out_lo + #{t : S >= T[t]} for every S, ties included."""
    rng = random.Random(mode)
    for _ in range(60):
        # small denominators make exact rounding ties frequent
        c = Fraction(rng.randint(1, 12), rng.choice([1, 2, 3, 4, 8, 7]))
        b = Fraction(rng.randint(-40, 40), rng.choice([1, 2, 4, 5]))
        s_y = Fraction(rng.randint(1, 12), rng.choice([1, 2, 4, 3]))
        out_lo, out_hi = rng.choice([(-8, 7), (0, 15), (-3, 3), (0, 1)])
        thr = quant_thresholds(c, b, s_y, mode, out_lo, out_hi)
        assert thr == sorted(thr)
        for S in range(-120, 121):
            q = min(max(round_fraction((c * S + b) / s_y, mode), out_lo), out_hi)
            assert q == out_lo + sum(S >= t for t in thr), (c, b, s_y, S)


BN_CNN = [
    dict(type="conv", out=4, kernel=3, pad=1, bn=True, bits=4),
    dict(type="maxpool"),
    dict(type="conv", out=6, kernel=3, bn=True, bits=3, relu=False),
    dict(type="flatten"),
    dict(type="linear", out=5, relu=False, bits=8),
]


def _bn_cnn():
    return make_cnn((3, 8, 8), BN_CNN, seed=3)


def _thresholded(ir):
    return [n.name for n in ir.nodes if n.op_type == "linear" and n.attrs.get("thresholds") is not None]


def test_auto_uses_thresholds_only_for_inexact_small_outputs():
    model = _bn_cnn()
    assert _thresholded(digitalify.from_qonnx(model)) == ["linear_0", "linear_1"]
    assert _thresholded(digitalify.from_qonnx(model, thresholds=True)) == [
        "linear_0", "linear_1", "linear_2"]
    assert _thresholded(digitalify.from_qonnx(model, thresholds=False)) == []
    # power-of-two scales: the multiplier form is exact, auto keeps it
    assert _thresholded(digitalify.from_qonnx(make_mlp([8, 6, 4], act_bits=4))) == []
    with pytest.raises(ValueError, match="thresholds"):
        digitalify.from_qonnx(model, thresholds="yes")


def test_thresholds_make_requantisation_exact():
    model = _bn_cnn()
    approx = digitalify.verify(model, ir=digitalify.from_qonnx(model, thresholds=False), n_vectors=4)
    assert not (approx.requant_exact and approx.bias_exact)
    exact = digitalify.verify(model, ir=digitalify.from_qonnx(model, thresholds=True), n_vectors=24)
    assert exact.passed and exact.requant_exact and exact.bias_exact, exact.summary()


def test_threshold_accumulator_has_no_fractional_bits():
    ir = digitalify.from_qonnx(_bn_cnn())
    node = ir.get_node("linear_0")
    assert node.attrs["bias_frac_bits"] == 0 and not node.attrs["relu"]
    assert node.attrs["thresholds"].shape == (4, 15)
    assert node.attrs["acc_dtype"].bits <= node.attrs["sum_dtype"].bits + 1


@pytest.mark.parametrize("pipeline, folding", [
    (None, None),
    (True, None),
    ("mult", None),
    (None, {"linear_0": (2, 9), "linear_1": (3, 4), "linear_2": (5, 2)}),
    (None, {"linear_0": (1, 27), "linear_2": (1, 1)}),
])
def test_threshold_vhdl_matches_reference(pipeline, folding, ghdl):
    ir = digitalify.from_qonnx(_bn_cnn(), thresholds=True)
    result = digitalify.simulate(ir, n_vectors=4, ghdl_path=ghdl, pipeline=pipeline, folding=folding)
    assert result.passed, result.summary() + "\n" + result.log[-3000:]


def test_threshold_vhdl_unsigned_8bit_output(ghdl):
    model = make_cnn((2, 6, 6), [dict(type="conv", out=3, kernel=3, bn=True, bits=8, signed=False)],
                     seed=4, input_signed=False)
    ir = digitalify.from_qonnx(model, thresholds=True)
    node = ir.nodes[-1]
    assert node.attrs["thresholds"].shape[1] <= 255 and str(ir.output_dtype) == "UINT8"
    assert digitalify.verify(model, ir=ir, n_vectors=16).passed
    files = digitalify.convert(ir)
    assert "THRESH" in files[f"{ir.name}_linear_0.vhd"]
    result = digitalify.simulate(ir, n_vectors=4, ghdl_path=ghdl)
    assert result.passed, result.summary() + "\n" + result.log[-3000:]
