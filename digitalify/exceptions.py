"""
Custom exceptions for the digitalify library.
"""


class UnsupportedOperatorError(NotImplementedError):
    """
    Raised when the importer or VHDL generator encounters an operator
    (or an operator pattern) that has no hardware mapping yet.

    Attributes
    ----------
    op_type : str
        The unsupported ONNX / QONNX operator type (e.g. ``"Conv"``).
    node_name : str
        Name of the graph node that triggered the error.
    """

    def __init__(self, op_type: str, node_name: str = "", detail: str = ""):
        self.op_type = op_type
        self.node_name = node_name
        location = f" (node '{node_name}')" if node_name else ""
        msg = f"Operator '{op_type}' is not supported{location}."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class QONNXImportError(ValueError):
    """
    Raised when a QONNX model is structurally valid ONNX but cannot be mapped
    to integer hardware (e.g. an unquantised network input, a non-zero
    zero-point, or a float bias that cannot be expressed in accumulator units).
    """


class VHDLGenerationError(RuntimeError):
    """
    Raised when VHDL generation fails for a reason other than an
    unsupported operator (e.g. an empty graph or inconsistent shapes).
    """
