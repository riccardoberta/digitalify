"""
Custom exceptions for the digitalify library.
"""


class UnsupportedOperatorError(NotImplementedError):
    """
    Raised when the importer or VHDL generator encounters an operator
    (layer / function) that has no registered handler.

    Attributes
    ----------
    op_type : str
        The unrecognised operator type string (e.g. ``"nn.BatchNorm1d"``).
    node_name : str
        Name of the graph node that triggered the error.
    """

    def __init__(self, op_type: str, node_name: str = ""):
        self.op_type = op_type
        self.node_name = node_name
        location = f" (node '{node_name}')" if node_name else ""
        super().__init__(
            f"Operator '{op_type}' is not yet supported{location}. "
            "Register a handler in digitalify.operators.registry to add support."
        )


class VHDLGenerationError(RuntimeError):
    """
    Raised when VHDL generation fails for a reason other than an
    unsupported operator (e.g. incompatible shapes, missing attributes).
    """
