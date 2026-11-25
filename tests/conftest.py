import warnings

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from digitalify.testing.ghdl_runner import find_ghdl
from digitalify.testing.qonnx_builder import QONNX_DOMAIN

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def ghdl():
    """Path to GHDL; tests using it are skipped when GHDL is not installed.

    Set DIGITALIFY_GHDL to point at a GHDL binary that is not on PATH.
    """
    try:
        return find_ghdl()
    except FileNotFoundError:
        pytest.skip("GHDL not found (install it or set DIGITALIFY_GHDL)")


def make_graph(nodes, inits, in_shape, out_shape, name="custom"):
    """Wrap nodes/initializers into a QONNX ModelProto."""
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("global_in", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info("global_out", TensorProto.FLOAT, out_shape)],
        initializer=[numpy_helper.from_array(np.asarray(v, dtype=np.float32), k)
                     for k, v in inits.items()],
    )
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13), helper.make_opsetid(QONNX_DOMAIN, 1)])
    model.ir_version = 8
    return model


def quant_node(inp, out, prefix, inits, scale, bits, signed=True, narrow=False,
               zeropt=0.0, rounding="ROUND"):
    inits[f"{prefix}_s"] = scale
    inits[f"{prefix}_z"] = zeropt
    inits[f"{prefix}_b"] = float(bits)
    return helper.make_node(
        "Quant", [inp, f"{prefix}_s", f"{prefix}_z", f"{prefix}_b"], [out],
        domain=QONNX_DOMAIN, name=f"Quant_{prefix}",
        signed=int(signed), narrow=int(narrow), rounding_mode=rounding)


@pytest.fixture
def relu_first_model():
    """Signed input → standalone Relu → MatMul → Quant: exercises the ReLU component."""
    rng = np.random.default_rng(3)
    inits = {"W": (rng.integers(-7, 8, size=(6, 3)) * 0.25).astype(np.float32)}
    nodes = [
        quant_node("global_in", "xq", "in", inits, 0.5, 5, signed=True),
        helper.make_node("Relu", ["xq"], ["xr"], name="Relu_0"),
        quant_node("W", "Wq", "w", inits, 0.25, 4, narrow=True),
        helper.make_node("MatMul", ["xr", "Wq"], ["y"], name="MatMul_0"),
        quant_node("y", "global_out", "out", inits, 0.5, 7, signed=True),
    ]
    return make_graph(nodes, inits, [1, 6], [1, 3], name="relu_first")
