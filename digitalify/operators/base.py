"""
Abstract base class for VHDL stage handlers.

Each supported IR op type provides a concrete subclass that emits
synthesisable VHDL-2008 for one IRNode.  Every stage has the same streaming
interface (see ``docs/dataflow_design.md``):

    clk, rst
    in_valid,  in_ready,  in_data       upstream handshake + packed beat
    out_valid, out_ready, out_data      downstream handshake + packed beat

A beat moves when valid and ready are both '1' on a rising edge
(AXI4-Stream rules).  Beat packing: element ``i`` of a beat whose element
type has ``B`` bits occupies ``data((i+1)*B-1 downto i*B)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from digitalify.ir.node import IRNode
from digitalify.ir.types import IntType

#: Optional pipeline registers inside a fully parallel ``linear`` stage, in
#: data-flow order:
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


# VHDL condition (on p, q, r, half) under which floor(p / 2^n) is incremented.
ROUND_UP_CONDITION = {
    "FLOOR": None,
    "CEIL": "r /= 0",
    "ROUND": "r > half or (r = half and r /= 0 and q(0) = '1')",
    "HALF_UP": "r > half or (r = half and r /= 0 and p >= 0)",
    "HALF_DOWN": "r > half or (r = half and r /= 0 and p < 0)",
    "UP": "r /= 0 and p > 0",
    "DOWN": "r /= 0 and p < 0",
}


class VHDLOperator(ABC):
    """
    Abstract handler for one stage type.

    A concrete handler writes only the *core* (:meth:`core`): logic that
    advances when ``ce = '1'`` (and freezes otherwise), drives ``in_ready``
    and the output register ``core_valid`` / ``core_data``.  The output
    register is consumed on every edge with ``ce = '1'``.  :meth:`architecture`
    wraps the core with the shared skid buffer, whose registered ``in_ready``
    is ``ce``.
    """

    def __init__(self, pipeline=None, folding=None):
        self.pipeline = normalize_pipeline(pipeline)
        self.folding = folding

    def node_latency(self, node: IRNode) -> int:
        """Clocks from accepting a beat to its result reaching the core output
        register, without stalls (informational)."""
        return 1

    def describe(self, node: IRNode) -> str:
        """One-line description of the hardware choices (for the top level)."""
        return ""

    @abstractmethod
    def core(self, node: IRNode) -> tuple[list[str], str]:
        """``(declarations, statements)`` of the core."""

    # ------------------------------------------------------------------
    # Wrapping
    # ------------------------------------------------------------------

    def architecture(self, node: IRNode, entity: str) -> str:
        """Complete VHDL source (library clauses + entity + architecture)."""
        decl, body = self.core(node)
        out_bits = node.out_features * node.output_dtype.bits
        skid = entity[: -len(node.name)] + "skid"
        return (
            self.header(node, entity) + self.entity(node, entity)
            + f"architecture rtl of {entity} is\n\n"
            + "".join(line + "\n" for line in decl)
            + "\n"
            "    signal ce         : std_logic;\n"
            "    signal core_valid : std_logic := '0';\n"
            f"    signal core_data  : std_logic_vector({out_bits - 1} downto 0) := (others => '0');\n"
            "\nbegin\n\n"
            + body
            + "    -- Output skid buffer: its registered in_ready is the core's clock enable\n"
            f"    skid : entity work.{skid}\n"
            f"        generic map (DATA_W => {out_bits})\n"
            "        port map (\n"
            "            clk       => clk,\n"
            "            rst       => rst,\n"
            "            in_valid  => core_valid,\n"
            "            in_ready  => ce,\n"
            "            in_data   => core_data,\n"
            "            out_valid => out_valid,\n"
            "            out_ready => out_ready,\n"
            "            out_data  => out_data\n"
            "        );\n\n"
            "end architecture rtl;\n"
        )

    # ------------------------------------------------------------------
    # Ports and instantiation
    # ------------------------------------------------------------------

    def ports(self, node: IRNode) -> str:
        in_bits = node.in_features * node.input_dtype.bits
        out_bits = node.out_features * node.output_dtype.bits
        return stream_ports(in_bits, out_bits)

    def component_declaration(self, node: IRNode, entity: str) -> str:
        return f"    component {entity} is\n{self.ports(node)}\n    end component {entity};"

    def instantiation(self, node: IRNode, entity: str, src: str, dst: str) -> str:
        """Instance wired from signals ``<src>_valid/_ready/_data`` to ``<dst>_...``."""
        maps = [
            "            clk       => clk",
            "            rst       => rst",
            f"            in_valid  => {src}_valid",
            f"            in_ready  => {src}_ready",
            f"            in_data   => {src}_data",
            f"            out_valid => {dst}_valid",
            f"            out_ready => {dst}_ready",
            f"            out_data  => {dst}_data",
        ]
        return f"    {entity}_inst : {entity}\n        port map (\n" + ",\n".join(maps) + "\n        );"

    def entity(self, node: IRNode, entity: str) -> str:
        return f"entity {entity} is\n{self.ports(node)}\nend entity {entity};\n\n"

    def header(self, node: IRNode, entity: str) -> str:
        src = node.attrs.get("onnx_node")
        info = self.describe(node)
        return (
            f"-- {entity}: {node.op_type}  {node.input_shape} x {node.input_dtype}"
            f" -> {node.output_shape} x {node.output_dtype}  (stream H, W, C)\n"
            + (f"-- {info}\n" if info else "")
            + (f"-- QONNX node: {src}\n" if src else "")
            + "-- Generated by digitalify\n\n"
            "library ieee;\n"
            "use ieee.std_logic_1164.all;\n"
            "use ieee.numeric_std.all;\n\n"
        )

    # ------------------------------------------------------------------
    # Code helpers
    # ------------------------------------------------------------------

    @staticmethod
    def element(bus: str, idx: str, bits) -> str:
        """VHDL slice of element *idx* of *bus*."""
        if not str(idx).isidentifier() and not str(idx).isdigit():
            idx = f"({idx})"
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

    @staticmethod
    def load_element(dtype: IntType, bus: str, idx: str, var: str) -> str:
        """Statement reading element *idx* of *bus* into the signed variable
        *var*, which is ``bits + 1`` wide so unsigned values keep their sign."""
        sl = VHDLOperator.element(bus, idx, dtype.bits)
        if dtype.signed:
            return f"{var} := resize(signed({sl}), {dtype.bits + 1});"
        return f"{var} := signed('0' & {sl});"

    # ------------------------------------------------------------------
    # Requantisation (shared by linear and requant)
    # ------------------------------------------------------------------

    def requant_decl(self, rq: dict, in_bits: int, count: int) -> list[str]:
        """Constants MULT / SHIFT / masks / clip bounds for *count* channels;
        *in_bits* is the signed width of the value being scaled."""
        mults = [int(m) for m in rq["multiplier"]]
        shifts = [int(n) for n in rq["shift"]]
        mb = max(m.bit_length() for m in mults) + 1           # + sign bit
        pw = max(in_bits + mb, max(shifts) + 2)
        return [
            "",
            "    -- Requantisation: q = clip(round(a * MULT / 2^SHIFT))",
            f"    -- rounding mode {rq['rounding']}, "
            f"{'exact' if rq['exact'] else 'fixed-point approximation of the'} scale ratio",
            f"    constant P_W : natural := {pw};",
            f"    type mult_t is array(0 to {count - 1}) of signed({mb - 1} downto 0);",
            "    constant MULT : mult_t := "
            + self.aggregate([self.bit_literal(m, mb) for m in mults]) + ";",
            f"    type wide_t is array(0 to {count - 1}) of signed(P_W - 1 downto 0);",
            "    constant SHIFT_MASK : wide_t := "
            + self.aggregate([self.bit_literal((1 << n) - 1, pw) for n in shifts]) + ";",
            "    constant SHIFT_HALF : wide_t := "
            + self.aggregate([self.bit_literal((1 << (n - 1)) if n else 0, pw) for n in shifts]) + ";",
            f"    type shift_t is array(0 to {count - 1}) of natural;",
            "    constant SHIFT : shift_t := " + self.aggregate([str(n) for n in shifts]) + ";",
            f"    constant CLIP_MIN : signed(P_W - 1 downto 0) := {self.bit_literal(rq['clip_min'], pw)};",
            f"    constant CLIP_MAX : signed(P_W - 1 downto 0) := {self.bit_literal(rq['clip_max'], pw)};",
        ]

    @staticmethod
    def requant_round(rq: dict, j: str, indent: str) -> list[str]:
        """Statements turning the product in variable ``p`` (channel *j*) into
        the clipped result in variable ``q`` (variables r, half also used)."""
        lines = [
            f"q    := shift_right(p, SHIFT({j}));     -- floor",
            f"r    := p and SHIFT_MASK({j});          -- discarded bits",
            f"half := SHIFT_HALF({j});",
        ]
        cond = ROUND_UP_CONDITION[rq["rounding"]]
        if cond:
            lines.append(f"if {cond} then q := q + 1; end if;")
        lines += [
            "if q > CLIP_MAX then q := CLIP_MAX;",
            "elsif q < CLIP_MIN then q := CLIP_MIN;",
            "end if;",
        ]
        return [indent + line for line in lines]


def stream_ports(in_bits: int, out_bits: int) -> str:
    lines = [
        "        clk       : in  std_logic;",
        "        rst       : in  std_logic;",
        "        in_valid  : in  std_logic;",
        "        in_ready  : out std_logic;",
        f"        in_data   : in  std_logic_vector({in_bits - 1} downto 0);",
        "        out_valid : out std_logic;",
        "        out_ready : in  std_logic;",
        f"        out_data  : out std_logic_vector({out_bits - 1} downto 0)",
    ]
    return "    port (\n" + "\n".join(lines) + "\n    );"


def skid_buffer(net: str) -> str:
    """Shared two-register skid buffer ``<net>_skid`` (registered in_ready)."""
    name = f"{net}_skid"
    return f"""\
-- {name}: two-register skid buffer.
-- in_ready is a register output, so out_ready never reaches in_ready
-- combinationally; full throughput (one beat per clock) without stalls.
-- Generated by digitalify

library ieee;
use ieee.std_logic_1164.all;

entity {name} is
    generic (DATA_W : positive);
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;
        in_valid  : in  std_logic;
        in_ready  : out std_logic;
        in_data   : in  std_logic_vector(DATA_W - 1 downto 0);
        out_valid : out std_logic;
        out_ready : in  std_logic;
        out_data  : out std_logic_vector(DATA_W - 1 downto 0)
    );
end entity {name};

architecture rtl of {name} is
    signal main_v, skid_v : std_logic := '0';
    signal main_d, skid_d : std_logic_vector(DATA_W - 1 downto 0) := (others => '0');
begin

    in_ready  <= not skid_v;
    out_valid <= main_v;
    out_data  <= main_d;

    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                main_v <= '0';
                skid_v <= '0';
            elsif out_ready = '1' or main_v = '0' then
                -- output register free: refill from the skid register or the input
                if skid_v = '1' then
                    main_v <= '1';
                    main_d <= skid_d;
                    skid_v <= '0';
                else
                    main_v <= in_valid;
                    main_d <= in_data;
                end if;
            elsif in_valid = '1' and skid_v = '0' then
                -- output stalled: park the incoming beat
                skid_v <= '1';
                skid_d <= in_data;
            end if;
        end if;
    end process;

end architecture rtl;
"""
