"""
Intermediate representation graph.

An IRGraph is a simple ordered list of IRNodes that together describe the
full data-flow of a neural network from a single input tensor to a single
output tensor.  The ordering is topological (each node's inputs are
defined before the node itself).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from digitalify.ir.node import IRNode


@dataclass
class IRGraph:
    """
    Ordered, directed acyclic graph of IRNodes.

    Parameters
    ----------
    name : str
        Human-readable name used as the top-level VHDL entity name
        (e.g. ``"network_top"``).
    nodes : list[IRNode]
        Topologically-ordered list of computation nodes.
    input_shape : tuple[int, ...]
        Shape of the network input activation (batch dim excluded).
    output_shape : tuple[int, ...]
        Shape of the network output activation (batch dim excluded).
    """

    name: str = "network_top"
    nodes: list[IRNode] = field(default_factory=list)
    input_shape: tuple[int, ...] = field(default_factory=tuple)
    output_shape: tuple[int, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Mutation helpers (used by the frontend importer)
    # ------------------------------------------------------------------

    def add_node(self, node: IRNode) -> None:
        """Append *node* to the graph."""
        self.nodes.append(node)

    # ------------------------------------------------------------------
    # Lookup helpers (used by the VHDL backend)
    # ------------------------------------------------------------------

    def get_node(self, name: str) -> IRNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"No node named '{name}' in graph '{self.name}'")

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __repr__(self) -> str:
        node_names = [n.name for n in self.nodes]
        return (
            f"IRGraph(name={self.name!r}, nodes={node_names}, "
            f"input_shape={self.input_shape}, output_shape={self.output_shape})"
        )
