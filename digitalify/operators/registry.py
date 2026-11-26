"""
Operator registry.

Maps IR op types (``IRNode.op_type``) to their ``VHDLOperator`` subclass.

To add a new stage:
1. Lower the QONNX pattern to IRNodes in ``digitalify/frontend/qonnx_importer.py``.
2. Write a ``VHDLOperator`` subclass (implement ``core``) and register it here.
3. Add its bit-exact model to ``digitalify/testing/reference.py``.
"""

from digitalify.operators.activations import LUTOperator, ReLUOperator, RequantOperator
from digitalify.operators.linear import LinearOperator
from digitalify.operators.window import (
    FlattenOperator,
    GlobalPoolOperator,
    PadOperator,
    PoolOperator,
    SWGOperator,
)

# op_type string → VHDLOperator subclass (not an instance)
OPERATOR_REGISTRY: dict[str, type] = {
    "linear": LinearOperator,
    "relu": ReLUOperator,
    "lut": LUTOperator,
    "requant": RequantOperator,
    "pad": PadOperator,
    "swg": SWGOperator,
    "pool": PoolOperator,
    "global_pool": GlobalPoolOperator,
    "flatten": FlattenOperator,
}
