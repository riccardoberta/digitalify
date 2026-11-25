"""
Operator registry.

Maps IR op types (``IRNode.op_type``) to their ``VHDLOperator`` subclass.

To add a new operator:
1. Map the QONNX pattern to an IRNode in ``digitalify/frontend/qonnx_importer.py``.
2. Write a ``VHDLOperator`` subclass and register it here.
3. Add its bit-exact model to ``digitalify/testing/reference.py``.
"""

from digitalify.operators.activations import ReLUOperator
from digitalify.operators.linear import LinearOperator

# op_type string → VHDLOperator subclass (not an instance)
OPERATOR_REGISTRY: dict[str, type] = {
    "linear": LinearOperator,
    "relu": ReLUOperator,
}
