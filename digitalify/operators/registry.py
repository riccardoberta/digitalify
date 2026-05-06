"""
Operator registry.

Maps operator type strings (as stored in ``IRNode.op_type``) to their
``VHDLOperator`` subclass.  The generator instantiates each handler with
the global ``data_width`` / ``acc_width`` settings.

To add a new operator:
1. Write a ``VHDLOperator`` subclass in a new or existing module.
2. Import it here and add an entry to ``OPERATOR_REGISTRY``.
"""

from digitalify.operators.linear import LinearOperator
from digitalify.operators.activations import ReLUOperator

# op_type string → VHDLOperator subclass (not an instance)
OPERATOR_REGISTRY: dict[str, type] = {
    "linear": LinearOperator,
    "quantized_linear": LinearOperator,   # same VHDL template; quant params baked in
    "relu": ReLUOperator,
}
