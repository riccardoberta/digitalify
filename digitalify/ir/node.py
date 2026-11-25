"""
Intermediate representation node.

Every hardware stage of the network becomes one IRNode.  The QONNX frontend
produces them; the VHDL backend and the reference model consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

from digitalify.ir.types import IntType


@dataclass
class IRNode:
    """
    A single operation in the computation graph.

    Parameters
    ----------
    name : str
        Unique name within the graph (e.g. ``"linear_0"``).
    op_type : str
        Operator class string used to look up the handler in the registry
        (``"linear"`` or ``"relu"``).
    input_names : list[str]
        Names of the predecessor nodes; ``["__input__"]`` for the network input.
    input_shape, output_shape : tuple[int, ...]
        Activation shapes, batch dimension excluded.
    input_dtype, output_dtype : IntType
        Integer type of one element of the input / output bus.
    attrs : dict[str, Any]
        Operator-specific attributes.

        ``linear``:
          - ``weight``  : int64 array ``(out_features, in_features)``
          - ``bias``    : int64 array ``(out_features,)`` (zeros if absent),
                          in units of ``s_in * s_w / 2**bias_frac_bits``
          - ``bias_frac_bits`` : int F; ``acc = (Σ w·x) · 2**F + bias``
          - ``bias_exact`` : bool — bias represented without rounding
          - ``weight_dtype`` : IntType of the weight constants
          - ``sum_dtype``    : IntType of the MAC chain Σ w·x
          - ``acc_dtype``    : IntType of ``sum·2**F + bias`` (cannot overflow)
          - ``relu``    : bool — apply max(0, ·) to the accumulator
          - ``requant`` : dict or None.  When present:
              ``multiplier`` (int array, per output channel),
              ``shift`` (int array), ``rounding`` (str),
              ``clip_min`` / ``clip_max`` (int), ``exact`` (bool).
            When None the accumulator itself is the output.

        ``relu``: no attributes.
    """

    name: str
    op_type: str
    input_names: list[str] = field(default_factory=list)
    input_shape: tuple[int, ...] = field(default_factory=tuple)
    output_shape: tuple[int, ...] = field(default_factory=tuple)
    input_dtype: IntType | None = None
    output_dtype: IntType | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def in_features(self) -> int:
        """Number of elements on the input bus."""
        return int(prod(self.input_shape))

    @property
    def out_features(self) -> int:
        """Number of elements on the output bus."""
        return int(prod(self.output_shape))

    def __repr__(self) -> str:
        return (
            f"IRNode(name={self.name!r}, op_type={self.op_type!r}, "
            f"{self.input_shape}:{self.input_dtype} -> "
            f"{self.output_shape}:{self.output_dtype})"
        )
