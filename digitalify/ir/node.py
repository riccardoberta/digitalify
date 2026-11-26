"""
Intermediate representation node.

Every hardware stage of the network becomes one IRNode.  The QONNX frontend
produces them; the VHDL backend and the reference model consume them.

Tensors are streams of beats (see ``docs/dataflow_design.md``): a tensor of
stream shape ``(H, W, C)`` travels as ``H·W`` beats in raster order, each beat
carrying the ``C`` elements of one pixel.  A plain vector is ``(1, 1, N)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from digitalify.ir.types import IntType


@dataclass
class IRNode:
    """
    A single stage in the computation graph.

    Parameters
    ----------
    name : str
        Unique name within the graph (e.g. ``"linear_0"``).
    op_type : str
        Stage type used to look up the handler in the registry: ``linear``,
        ``relu``, ``pad``, ``swg``, ``pool``, ``global_pool``, ``flatten``,
        ``lut`` or ``requant``.
    input_names : list[str]
        Names of the predecessor nodes; ``["__input__"]`` for the network input.
    input_shape, output_shape : tuple[int, int, int]
        Stream shapes ``(H, W, C)``.
    input_dtype, output_dtype : IntType
        Integer type of one element of the input / output beat.
    attrs : dict[str, Any]
        Stage-specific attributes.

        ``linear`` (applied to every beat independently):
          - ``weight``  : int64 array ``(out_features, in_features)``
          - ``bias``    : int64 array ``(out_features,)`` (zeros if absent),
                          in units of ``acc_scale / 2**bias_frac_bits``
          - ``bias_frac_bits`` : int F; ``acc = (Σ w·x) · 2**F + bias``
          - ``bias_exact`` : bool — bias represented without rounding
          - ``weight_dtype`` : IntType of the weight constants
          - ``sum_dtype``    : IntType of the sum Σ w·x
          - ``acc_dtype``    : IntType of ``sum·2**F + bias`` (cannot overflow)
          - ``relu``    : bool — apply max(0, ·) to the accumulator
          - ``requant`` : dict or None (see ``requant`` below, per output
            channel).  When None the accumulator itself is the output.

        ``relu``: no attributes.

        ``pad``: ``pads`` (top, left, bottom, right), ``value`` (int).

        ``swg``: ``kernel`` (KH, KW), ``stride`` (SH, SW).  Output beat =
        window in ``(kh, kw, c)`` order.

        ``pool``: ``mode`` ("max" | "sum"), ``window`` (KH·KW): per-channel
        reduction of a window beat.

        ``global_pool``: ``mode`` ("sum"): per-channel sum over all beats.

        ``flatten``: no attributes; collects all beats into one (HWC order).

        ``lut``: ``tables`` (int64 array ``(T, hi - lo + 1)``), ``table_of``
        (int array, table index per channel), ``in_lo`` (int).

        ``requant``: ``multiplier``, ``shift`` (int arrays, per channel),
        ``rounding`` (str), ``clip_min`` / ``clip_max`` (int), ``exact``
        (bool), ``relu`` (bool, applied before scaling).
    """

    name: str
    op_type: str
    input_names: list[str] = field(default_factory=list)
    input_shape: tuple[int, int, int] = (1, 1, 1)
    output_shape: tuple[int, int, int] = (1, 1, 1)
    input_dtype: IntType | None = None
    output_dtype: IntType | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def in_features(self) -> int:
        """Elements per input beat (C of the input stream)."""
        return int(self.input_shape[-1])

    @property
    def out_features(self) -> int:
        """Elements per output beat (C of the output stream)."""
        return int(self.output_shape[-1])

    @property
    def in_pixels(self) -> int:
        """Input beats per tensor (H·W)."""
        return int(self.input_shape[0] * self.input_shape[1])

    @property
    def out_pixels(self) -> int:
        """Output beats per tensor (H·W)."""
        return int(self.output_shape[0] * self.output_shape[1])

    def __repr__(self) -> str:
        return (
            f"IRNode(name={self.name!r}, op_type={self.op_type!r}, "
            f"{self.input_shape}:{self.input_dtype} -> "
            f"{self.output_shape}:{self.output_dtype})"
        )
