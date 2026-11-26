import numpy as np
import pytest
from onnx import helper

from conftest import make_graph, quant_node
from digitalify import (
    IntType, QONNXImportError, UnsupportedOperatorError, convert, from_qonnx,
)
from digitalify.backend.vhdl.naming import vhdl_identifier
from digitalify.testing import make_mlp


def test_mixed_precision_types():
    ir = from_qonnx(make_mlp([16, 8, 6, 4], weight_bits=[4, 3, 6], act_bits=[5, 3], output_bits=6))
    assert [n.op_type for n in ir.nodes] == ["linear"] * 3
    assert ir.input_dtype == IntType(8, True)
    assert [n.output_dtype for n in ir.nodes] == [IntType(5, False), IntType(3, False), IntType(6, True)]
    assert ir.nodes[0].attrs["weight_dtype"].bits <= 4
    assert ir.output_dtype == IntType(6, True)


def test_weights_are_integers_in_range():
    ir = from_qonnx(make_mlp([16, 8, 4], weight_bits=4))
    w = ir.nodes[0].attrs["weight"]
    assert w.dtype == np.int64 and w.shape == (8, 16)
    assert w.min() >= -7 and w.max() <= 7          # narrow signed 4-bit


def test_float_bias_gets_fractional_bits():
    ir = from_qonnx(make_mlp([16, 8, 4], quantized_bias=False))
    assert ir.nodes[0].attrs["bias_frac_bits"] > 0
    ir = from_qonnx(make_mlp([16, 8, 4], quantized_bias=True))
    assert ir.nodes[0].attrs["bias_frac_bits"] == 0
    assert ir.nodes[0].attrs["bias_exact"]


def test_multidim_input_is_flattened():
    ir = from_qonnx(make_mlp([16, 8, 4], input_shape=(1, 4, 4)))
    assert ir.input_shape == (1, 4, 4)
    assert ir.in_features == 16


def test_name_override_and_sanitising():
    assert from_qonnx(make_mlp([4, 2]), name="My Net!").name == "my_net"
    assert vhdl_identifier("1-entity") == "n_1_entity"
    assert vhdl_identifier("signal") == "signal_net"
    assert vhdl_identifier("") == "network"


def test_convert_writes_files(tmp_path):
    files = convert(make_mlp([8, 4, 2]), output_dir=str(tmp_path))
    assert list(files) == ["mlp_skid.vhd", "mlp_linear_0.vhd", "mlp_linear_1.vhd", "mlp_top.vhd"]
    assert (tmp_path / "mlp_top.vhd").read_text().startswith("-- Top-level")


# ---- errors -------------------------------------------------------------------

def test_unquantized_input_is_rejected():
    inits = {"W": np.ones((4, 2), dtype=np.float32)}
    nodes = [helper.make_node("MatMul", ["global_in", "W"], ["global_out"], name="mm")]
    with pytest.raises(QONNXImportError, match="input must be quantised"):
        from_qonnx(make_graph(nodes, inits, [1, 4], [1, 2]))


def test_unquantized_float_weights_are_rejected():
    inits = {"W": np.full((4, 2), 0.3, dtype=np.float32)}
    nodes = [
        quant_node("global_in", "xq", "in", inits, 0.5, 8),
        helper.make_node("MatMul", ["xq", "W"], ["global_out"], name="mm"),
    ]
    with pytest.raises(QONNXImportError, match="not quantised"):
        from_qonnx(make_graph(nodes, inits, [1, 4], [1, 2]))


def test_nonzero_zero_point_is_rejected():
    inits = {}
    nodes = [quant_node("global_in", "global_out", "in", inits, 0.5, 8, zeropt=3.0)]
    with pytest.raises(QONNXImportError, match="zero-point"):
        from_qonnx(make_graph(nodes, inits, [1, 4], [1, 4]))


def test_unsupported_operator_is_reported():
    inits = {}
    nodes = [
        quant_node("global_in", "xq", "in", inits, 0.5, 8),
        helper.make_node("Sigmoid", ["xq"], ["global_out"], name="sig"),
    ]
    with pytest.raises(UnsupportedOperatorError, match="Sigmoid"):
        from_qonnx(make_graph(nodes, inits, [1, 4], [1, 4]))


def test_branching_is_rejected():
    inits = {}
    nodes = [
        quant_node("global_in", "xq", "in", inits, 0.5, 8),
        helper.make_node("Relu", ["xq"], ["a"], name="r1"),
        helper.make_node("Relu", ["xq"], ["b"], name="r2"),
        helper.make_node("Add", ["a", "b"], ["global_out"], name="add"),
    ]
    with pytest.raises(UnsupportedOperatorError, match="chains"):
        from_qonnx(make_graph(nodes, inits, [1, 4], [1, 4]))
