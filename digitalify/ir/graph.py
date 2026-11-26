"""
Intermediate representation graph.

An IRGraph is a topologically ordered list of IRNodes describing the data
flow of a network from a single quantised input tensor to a single output
tensor, plus the information needed to move between real values and the
integers carried by the hardware:

    real_input  = input_scale  * q_input
    real_output = output_scale * q_output      (output_scale is per element)

Integer tensors are exchanged in *stream order*: the ``H·W·C`` elements of a
tensor of stream shape ``(H, W, C)``, pixel by pixel in raster order.  The
ONNX model uses NCHW; ``onnx_to_stream`` / ``stream_to_onnx`` convert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

import numpy as np

from digitalify.ir.node import IRNode
from digitalify.ir.types import IntType


@dataclass
class IRGraph:
    """
    Parameters
    ----------
    name : str
        Valid VHDL identifier used as prefix for every entity
        (the top level is ``<name>_top``).
    nodes : list[IRNode]
        Topologically ordered computation nodes.
    input_shape, output_shape : tuple[int, ...]
        ONNX shapes of the network input / output, batch dimension excluded.
    input_stream, output_stream : tuple[int, int, int]
        Stream shapes ``(H, W, C)`` of the input / output.  A rank-3 ONNX
        tensor ``(C, H, W)`` streamed as an image has stream ``(H, W, C)``;
        anything else is one vector beat ``(1, 1, N)``.
    input_dtype, output_dtype : IntType
        Integer type of one element of the input / output beat.
    input_range : tuple[int, int]
        Integer range accepted at the input (narrower than ``input_dtype``
        when the input Quant node uses ``narrow=1``).
    input_scale : float
        Real value of one input LSB.
    output_scale : np.ndarray
        Real value of one output LSB, one entry per output element in stream
        order.
    """

    name: str = "network"
    nodes: list[IRNode] = field(default_factory=list)
    input_shape: tuple[int, ...] = field(default_factory=tuple)
    output_shape: tuple[int, ...] = field(default_factory=tuple)
    input_stream: tuple[int, int, int] = (1, 1, 1)
    output_stream: tuple[int, int, int] = (1, 1, 1)
    input_dtype: IntType | None = None
    output_dtype: IntType | None = None
    input_range: tuple[int, int] | None = None
    input_scale: float = 1.0
    output_scale: np.ndarray = field(default_factory=lambda: np.ones(1))

    def add_node(self, node: IRNode) -> None:
        self.nodes.append(node)

    def get_node(self, name: str) -> IRNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"No node named '{name}' in graph '{self.name}'")

    # ------------------------------------------------------------------
    # Stream geometry
    # ------------------------------------------------------------------

    @property
    def in_features(self) -> int:
        """Elements per input beat."""
        return int(self.input_stream[2])

    @property
    def out_features(self) -> int:
        """Elements per output beat."""
        return int(self.output_stream[2])

    @property
    def in_pixels(self) -> int:
        return int(self.input_stream[0] * self.input_stream[1])

    @property
    def out_pixels(self) -> int:
        return int(self.output_stream[0] * self.output_stream[1])

    @property
    def in_size(self) -> int:
        """Elements per input tensor."""
        return int(prod(self.input_stream))

    @property
    def out_size(self) -> int:
        """Elements per output tensor."""
        return int(prod(self.output_stream))

    @staticmethod
    def _spatial(onnx_shape, stream) -> bool:
        """True if an ONNX (C, H, W) / (C, L) tensor is streamed as (H, W, C) / (1, L, C)."""
        if len(onnx_shape) == 3:
            return tuple(stream) == (onnx_shape[1], onnx_shape[2], onnx_shape[0])
        if len(onnx_shape) == 2:
            return tuple(stream) == (1, onnx_shape[1], onnx_shape[0])
        return False

    def _shapes(self, which: str):
        if which == "input":
            return tuple(self.input_shape), tuple(self.input_stream)
        return tuple(self.output_shape), tuple(self.output_stream)

    def onnx_to_stream(self, x: np.ndarray, which: str = "input") -> np.ndarray:
        """ONNX-layout tensors ``(n, *shape)`` → stream order ``(n, H·W·C)``."""
        shape, stream = self._shapes(which)
        x = np.asarray(x)
        n = x.shape[0]
        if self._spatial(shape, stream):
            x = np.moveaxis(x.reshape(n, *shape), 1, -1)
        return x.reshape(n, -1)

    def stream_to_onnx(self, x: np.ndarray, which: str = "input") -> np.ndarray:
        """Stream order ``(n, H·W·C)`` → ONNX layout ``(n, *shape)``."""
        shape, stream = self._shapes(which)
        x = np.asarray(x)
        n = x.shape[0]
        if self._spatial(shape, stream):
            return np.moveaxis(x.reshape(n, *shape[1:], shape[0]), -1, 1)
        return x.reshape(n, *shape)

    def summary(self) -> str:
        """Human-readable table of the stages."""
        lines = [f"IRGraph '{self.name}': input {self.input_stream} {self.input_dtype} "
                 f"(scale {self.input_scale:g})"]
        for n in self.nodes:
            a = n.attrs
            extra = ""
            if n.op_type == "linear":
                rq = a.get("requant")
                extra = (f"  W:{a['weight_dtype']} acc:{a['acc_dtype']}"
                         f"{' +relu' if a.get('relu') else ''}"
                         + (f" requant({rq['rounding']}{'' if rq['exact'] else ', approx'})"
                            if rq else " (no requant)"))
            elif n.op_type == "swg":
                extra = f"  kernel {a['kernel']} stride {a['stride']}"
            elif n.op_type in ("pool", "global_pool"):
                extra = f"  {a['mode']}"
            elif n.op_type == "pad":
                extra = f"  pads {a['pads']} value {a['value']}"
            lines.append(f"  {n.name:<14} {n.op_type:<11} {str(n.input_shape):>14} {n.input_dtype}"
                         f" -> {str(n.output_shape):>14} {n.output_dtype}{extra}")
        lines.append(f"output {self.output_stream} {self.output_dtype}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __repr__(self) -> str:
        return (f"IRGraph(name={self.name!r}, nodes={[n.name for n in self.nodes]}, "
                f"input={self.input_stream}:{self.input_dtype}, "
                f"output={self.output_stream}:{self.output_dtype})")
