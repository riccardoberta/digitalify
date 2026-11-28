"""
A real Brevitas export: the MNIST CNN in examples/mnist (trained with
examples/mnist/train_brevitas.py, 98.4 % test accuracy).

It uses the torch.onnx dynamo exporter (initializers listed as graph inputs),
BatchNorm after bias-free convolutions, non-power-of-two scales and a
Reshape-based flatten.
"""

from pathlib import Path

import numpy as np
import pytest

import digitalify

MODEL = Path(__file__).resolve().parent.parent / "examples" / "mnist" / "mnist_cnn.onnx"


@pytest.fixture(scope="module")
def ir():
    return digitalify.from_qonnx(MODEL)


def test_lowering(ir):
    assert [n.op_type for n in ir.nodes] == [
        "swg", "linear", "swg", "pool", "swg", "linear", "swg", "pool", "flatten", "linear"]
    assert ir.input_stream == (28, 28, 1) and ir.output_stream == (1, 1, 10)
    assert str(ir.input_dtype) == "UINT8" and str(ir.output_dtype) == "INT8"
    # BatchNorm + 4-bit ReLU: threshold activations, no fractional bias bits
    for name in ("linear_0", "linear_1"):
        attrs = ir.get_node(name).attrs
        assert attrs["thresholds"].shape[1] == 15 and attrs["bias_frac_bits"] == 0


def test_reference_matches_qonnx(ir):
    result = digitalify.verify(MODEL, ir=ir, n_vectors=6)
    assert result.passed, result.summary()


def test_quantize_input(ir):
    images = np.random.default_rng(0).random((3, 1, 28, 28), dtype=np.float32)
    q = ir.quantize_input(images)
    assert q.shape == (3, 784) and q.min() >= 0 and q.max() <= 255
    expected = np.clip(np.round(images / np.float32(ir.input_scale)), 0, 255)
    np.testing.assert_array_equal(q, expected.reshape(3, -1))
    result = digitalify.verify(MODEL, ir=ir, inputs=q)
    assert result.passed and result.n_vectors == 3


def test_vhdl_matches_reference(ir, ghdl):
    result = digitalify.simulate(ir, n_vectors=2, ghdl_path=ghdl)
    assert result.passed, result.summary() + "\n" + result.log[-3000:]


def test_balanced_folding_keeps_one_pixel_per_clock(ir, ghdl):
    folding = {"linear_1": (16, 24), "linear_2": (2, 40)}
    runs = [digitalify.simulate(ir, n_vectors=n, ghdl_path=ghdl, backpressure=False,
                                folding=folding) for n in (2, 4)]
    assert all(r.passed for r in runs)
    assert runs[1].cycles - runs[0].cycles == 2 * 28 * 28


def test_wide_product_gets_three_registers(ir):
    """pipeline=True: the final layer's 31-bit x 33-bit requantisation product
    spans several DSP blocks, so it gets 3 registers; the thresholded layers
    keep one comparison register."""
    from digitalify.operators.linear import LinearOperator
    op = LinearOperator(True)
    extra = {n.name: op._extra_product_registers(n) for n in ir.nodes if n.op_type == "linear"}
    assert extra == {"linear_0": 0, "linear_1": 0, "linear_2": 2}
    assert "mult_reg2" in digitalify.convert(ir, pipeline=True)[f"{ir.name}_linear_2.vhd"]
