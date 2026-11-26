"""
Abstract base class for VHDL operator handlers.

Each supported IR op type provides a concrete subclass that emits
synthesisable VHDL-2008 for one IRNode.  Every component has the same
streaming interface:

    clk, rst            (only for registered components)
    valid_i, data_i     input handshake + packed input bus
    valid_o, data_o     output handshake + packed output bus

Bus packing: element ``i`` of a bus whose element type has ``B`` bits
occupies ``data((i+1)*B-1 downto i*B)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from digitalify.ir.node import IRNode

#: Optional pipeline registers inside a ``linear`` layer, in data-flow order:
#:   "products" — after the products x·w (before the adder tree)
#:   "tree"     — inside the adder tree, every ``tree_every`` levels
#:                ("tree:2" = every 2 levels; plain "tree" = every level)
#:   "acc"      — after the adder tree + bias (before ReLU / requantisation)
#:   "mult"     — after the requantisation product ``acc · MULT`` (before rounding)
PIPELINE_STAGES = ("products", "tree", "acc", "mult")


@dataclass(frozen=True)
class Pipeline:
    """Normalised ``pipeline`` option (see :func:`normalize_pipeline`)."""

    stages: frozenset = frozenset()
    tree_every: int = 1

    def __contains__(self, stage: str) -> bool:
        return stage in self.stages

    def __bool__(self) -> bool:
        return bool(self.stages)

    def describe(self) -> str:
        names = []
        for st in PIPELINE_STAGES:
            if st in self.stages:
                names.append(f"tree:{self.tree_every}" if st == "tree" else st)
        return ", ".join(names)


def normalize_pipeline(pipeline) -> Pipeline:
    """
    ``None`` / ``False`` → no extra registers; ``True`` → every stage, the
    adder tree registered at every level; a stage name or an iterable of
    names (``"tree:K"`` for a register every K tree levels) → those stages.
    """
    if isinstance(pipeline, Pipeline):
        return pipeline
    if pipeline is None or pipeline is False:
        return Pipeline()
    if pipeline is True:
        return Pipeline(frozenset(PIPELINE_STAGES), 1)
    items = [pipeline] if isinstance(pipeline, str) else list(pipeline)
    stages, tree_every, unknown = set(), 1, []
    for item in items:
        name, _, arg = str(item).partition(":")
        if name not in PIPELINE_STAGES or (arg and name != "tree"):
            unknown.append(item)
            continue
        if arg:
            if not arg.isdigit() or int(arg) < 1:
                raise ValueError(f"'{item}': the tree interval must be a positive integer.")
            tree_every = int(arg)
        stages.add(name)
    if unknown:
        raise ValueError(f"Unknown pipeline stage(s) {unknown}; valid stages are "
                         f"{list(PIPELINE_STAGES)} ('tree:K' for every K tree levels).")
    return Pipeline(frozenset(stages), tree_every)


class VHDLOperator(ABC):
    """Abstract handler for one operator type."""

    #: Number of clock cycles from valid_i to valid_o.
    latency: int = 0

    #: Whether the component has clk / rst ports.
    clocked: bool = False

    def __init__(self, pipeline: bool | str | Iterable[str] | None = None):
        self.pipeline = normalize_pipeline(pipeline)

    def node_latency(self, node: IRNode) -> int:
        """Clock cycles from valid_i to valid_o for this node."""
        return self.latency

    @abstractmethod
    def architecture(self, node: IRNode, entity: str) -> str:
        """Complete VHDL source (library clauses + entity + architecture)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def ports(self, node: IRNode) -> str:
        in_bits = node.in_features * node.input_dtype.bits
        out_bits = node.out_features * node.output_dtype.bits
        lines = []
        if self.clocked:
            lines += ["        clk     : in  std_logic;", "        rst     : in  std_logic;"]
        lines += [
            "        valid_i : in  std_logic;",
            f"        data_i  : in  std_logic_vector({in_bits - 1} downto 0);",
            "        valid_o : out std_logic;",
            f"        data_o  : out std_logic_vector({out_bits - 1} downto 0)",
        ]
        return "    port (\n" + "\n".join(lines) + "\n    );"

    def component_declaration(self, node: IRNode, entity: str) -> str:
        return f"    component {entity} is\n{self.ports(node)}\n    end component {entity};"

    def instantiation(self, node: IRNode, entity: str, data_in: str, valid_in: str,
                      data_out: str, valid_out: str) -> str:
        maps = []
        if self.clocked:
            maps += ["            clk     => clk", "            rst     => rst"]
        maps += [
            f"            valid_i => {valid_in}",
            f"            data_i  => {data_in}",
            f"            valid_o => {valid_out}",
            f"            data_o  => {data_out}",
        ]
        return f"    {entity}_inst : {entity}\n        port map (\n" + ",\n".join(maps) + "\n        );"

    def entity(self, node: IRNode, entity: str) -> str:
        return f"entity {entity} is\n{self.ports(node)}\nend entity {entity};\n\n"

    @staticmethod
    def header(node: IRNode, entity: str) -> str:
        src = node.attrs.get("onnx_node")
        return (
            f"-- {entity}: {node.op_type} "
            f"{node.in_features} x {node.input_dtype} -> {node.out_features} x {node.output_dtype}\n"
            + (f"-- QONNX node: {src}\n" if src else "")
            + "-- Generated by digitalify\n\n"
            "library ieee;\n"
            "use ieee.std_logic_1164.all;\n"
            "use ieee.numeric_std.all;\n\n"
        )

    @staticmethod
    def element(bus: str, idx: str, bits: int) -> str:
        """VHDL slice of element *idx* of *bus*."""
        return f"{bus}(({idx} + 1) * {bits} - 1 downto {idx} * {bits})"

    @staticmethod
    def aggregate(items: list[str], indent: str = "        ") -> str:
        """Positional array aggregate; named for one element (VHDL requires it)."""
        if len(items) == 1:
            return f"(0 => {items[0]})"
        return "(\n" + ",\n".join(indent + it for it in items) + "\n" + indent[:-4] + ")"

    @staticmethod
    def bit_literal(value: int, bits: int) -> str:
        """Two's-complement bit-string literal, valid for any width."""
        value = int(value)
        if not -(1 << (bits - 1)) <= value < (1 << bits):
            raise ValueError(f"{value} does not fit in {bits} bits")
        return '"' + format(value & ((1 << bits) - 1), f"0{bits}b") + '"'
