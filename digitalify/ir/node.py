"""
Intermediate representation node.

Every layer / operation in the model becomes one IRNode.  The VHDL backend
consumes IRNodes; the torch.fx frontend produces them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IRNode:
    """
    A single operation in the computation graph.

    Parameters
    ----------
    name : str
        Unique name within the graph (e.g. ``"linear_0"``).
    op_type : str
        Operator class string used to look up the handler in the registry.
        Examples: ``"linear"``, ``"relu"``, ``"quantized_linear"``.
    input_names : list[str]
        Names of predecessor nodes whose outputs feed this node.
        Use ``["__input__"]`` for the first node.
    input_shape : tuple[int, ...]
        Shape of the activation tensor arriving at this node
        (batch dimension excluded, e.g. ``(784,)``).
    output_shape : tuple[int, ...]
        Shape of the activation tensor leaving this node.
    attrs : dict[str, Any]
        Operator-specific attributes.  Common keys:

        For ``linear`` / ``quantized_linear``:
          - ``weight``  : numpy array, shape ``(out_features, in_features)``
          - ``bias``    : numpy array, shape ``(out_features,)`` or ``None``
          - ``scale``   : float — requantisation scale (output_scale /
                          (input_scale * weight_scale)), default 1.0
          - ``shift``   : int   — right-shift equivalent of *scale* when
                          scale ≈ 1/2^shift, default 0
          - ``in_zero`` : int   — input zero-point (for unsigned INT8), default 0
          - ``out_zero``: int   — output zero-point, default 0

        For ``relu``: no extra keys needed.
    """

    name: str
    op_type: str
    input_names: list[str] = field(default_factory=list)
    input_shape: tuple[int, ...] = field(default_factory=tuple)
    output_shape: tuple[int, ...] = field(default_factory=tuple)
    attrs: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def in_features(self) -> int:
        """Flat number of input features (last dimension)."""
        return int(self.input_shape[-1])

    @property
    def out_features(self) -> int:
        """Flat number of output features (last dimension)."""
        return int(self.output_shape[-1])

    def __repr__(self) -> str:
        return (
            f"IRNode(name={self.name!r}, op_type={self.op_type!r}, "
            f"input_shape={self.input_shape}, output_shape={self.output_shape})"
        )
