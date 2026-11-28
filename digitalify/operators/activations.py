"""
Elementwise stages: one output element per input element, one beat per clock.

relu     max(0, x); the identity on an unsigned input.
lut      table lookup per channel: out = TABLE[x - IN_LO].  Any activation
         followed by a Quant (Sigmoid, Tanh, Clip, ...) maps to it.
requant  clip(round(relu?(x) · MULT[c] / 2^SHIFT[c])) — a standalone Quant.
"""

from __future__ import annotations

from digitalify.ir.node import IRNode
from digitalify.operators.base import VHDLOperator


def _registered(stmts: list[str], variables: str = "", label: str = "elem") -> str:
    """Output register process applying *stmts* to every channel c."""
    return (
        "    in_ready <= ce;\n\n"
        f"    {label} : process(clk)\n" + variables +
        "    begin\n"
        "        if rising_edge(clk) then\n"
        "            if rst = '1' then\n"
        "                core_valid <= '0';\n"
        "            elsif ce = '1' then\n"
        "                core_valid <= in_valid;\n"
        + "".join(s + "\n" for s in stmts) +
        "            end if;\n"
        "        end if;\n"
        f"    end process {label};\n\n"
    )


class ReLUOperator(VHDLOperator):
    """Handler for ``relu`` IRNodes."""

    def core(self, node: IRNode) -> tuple[list[str], str]:
        bits = node.input_dtype.bits
        n = node.in_features
        out_sl = self.element("core_data", "c", bits)
        in_sl = self.element("in_data", "c", bits)
        if not node.input_dtype.signed:
            stmts = ["                core_data <= in_data;   -- unsigned input: identity"]
        else:
            stmts = [
                f"                for c in 0 to {n - 1} loop",
                f"                    if in_data((c + 1) * {bits} - 1) = '1' then",
                f"                        {out_sl} <= (others => '0');",
                "                    else",
                f"                        {out_sl} <= {in_sl};",
                "                    end if;",
                "                end loop;",
            ]
        return [], _registered(stmts, label="relu")


class LUTOperator(VHDLOperator):
    """Handler for ``lut`` IRNodes (elementwise table lookup)."""

    def describe(self, node: IRNode) -> str:
        t = node.attrs["tables"]
        return f"{len(t)} table(s) of {len(t[0])} entries"

    def core(self, node: IRNode) -> tuple[list[str], str]:
        a = node.attrs
        tables = a["tables"]
        table_of = [int(v) for v in a["table_of"]]
        yb = node.output_dtype.bits
        n = node.in_features
        decl = [
            f"    constant IN_LO : integer := {int(a['in_lo'])};",
            f"    type lut_t is array(0 to {len(tables[0]) - 1}) of std_logic_vector({yb - 1} downto 0);",
        ]
        for t, table in enumerate(tables):
            decl.append(f"    constant LUT{t} : lut_t := "
                        + self.aggregate([self.bit_literal(v, yb) for v in table]) + ";")
        load = self.load_element(node.input_dtype, "in_data", "c", "x")
        out_sl = self.element("core_data", "c", yb)
        if len(tables) == 1:
            stmts = [
                f"                for c in 0 to {n - 1} loop",
                f"                    {load}",
                f"                    {out_sl} <= LUT0(to_integer(x) - IN_LO);",
                "                end loop;",
            ]
        else:
            stmts = []
            for c, t in enumerate(table_of):
                stmts.append(f"                {self.load_element(node.input_dtype, 'in_data', str(c), 'x')}")
                stmts.append(f"                {self.element('core_data', str(c), yb)} "
                             f"<= LUT{t}(to_integer(x) - IN_LO);")
        variables = f"        variable x : signed({node.input_dtype.bits} downto 0);\n"
        return decl, _registered(stmts, variables, label="lut")


class RequantOperator(VHDLOperator):
    """Handler for ``requant`` IRNodes (standalone Quant on an integer tensor)."""

    def core(self, node: IRNode) -> tuple[list[str], str]:
        rq = node.attrs
        n = node.in_features
        xb = node.input_dtype.bits
        yb = node.output_dtype.bits
        decl = self.requant_decl(rq, xb + 1, n)
        stmts = [
            f"                for c in 0 to {n - 1} loop",
            f"                    {self.load_element(node.input_dtype, 'in_data', 'c', 'x')}",
        ]
        if rq.get("relu"):
            stmts.append("                    if x < 0 then x := (others => '0'); end if;")
        stmts.append("                    p := resize(x * MULT(c), P_W);")
        stmts += self.requant_round(rq, "c", " " * 20)
        stmts += [
            f"                    {self.element('core_data', 'c', yb)} <= std_logic_vector(q({yb - 1} downto 0));",
            "                end loop;",
        ]
        variables = (f"        variable x : signed({xb} downto 0);\n"
                     "        variable p, q, grd, stk : signed(P_W - 1 downto 0);\n")
        return decl, _registered(stmts, variables, label="requant")
