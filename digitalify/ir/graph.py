"""
Intermediate representation graph.

An IRGraph is a topologically ordered list of IRNodes describing the data
flow of a network from a single quantised input tensor to a single output
tensor, plus the information needed to move between real values and the
integers carried by the hardware:

    real_input  = input_scale  * q_input
    real_output = output_scale * q_output      (output_scale is per element)
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
        Network input / output shapes, batch dimension excluded.
    input_dtype, output_dtype : IntType
        Integer type of one element of the input / output bus.
    input_range : tuple[int, int]
        Integer range accepted at the input (narrower than ``input_dtype``
        when the input Quant node uses ``narrow=1``).
    input_scale : float
        Real value of one input LSB.
    output_scale : np.ndarray
        Real value of one output LSB, one entry per output element.
    """

    name: str = "network"
    nodes: list[IRNode] = field(default_factory=list)
    input_shape: tuple[int, ...] = field(default_factory=tuple)
    output_shape: tuple[int, ...] = field(default_factory=tuple)
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

    @property
    def in_features(self) -> int:
        return int(prod(self.input_shape))

    @property
    def out_features(self) -> int:
        return int(prod(self.output_shape))

    def summary(self) -> str:
        """Human-readable table of the layers."""
        lines = [f"IRGraph '{self.name}': input {self.input_shape} {self.input_dtype} "
                 f"(scale {self.input_scale:g})"]
        for n in self.nodes:
            extra = ""
            if n.op_type == "linear":
                rq = n.attrs.get("requant")
                extra = (f"  W:{n.attrs['weight_dtype']} acc:{n.attrs['acc_dtype']}"
                         f"{' +relu' if n.attrs.get('relu') else ''}"
                         + (f" requant({rq['rounding']}{'' if rq['exact'] else ', approx'})"
                            if rq else " (no requant)"))
            lines.append(f"  {n.name:<12} {n.op_type:<7} {n.in_features:>5} x {n.input_dtype}"
                         f" -> {n.out_features:>5} x {n.output_dtype}{extra}")
        lines.append(f"output {self.output_shape} {self.output_dtype}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __repr__(self) -> str:
        return (f"IRGraph(name={self.name!r}, nodes={[n.name for n in self.nodes]}, "
                f"input={self.input_shape}:{self.input_dtype}, "
                f"output={self.output_shape}:{self.output_dtype})")
