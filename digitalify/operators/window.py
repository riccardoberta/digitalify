"""
Spatial stages working on pixel streams (one beat = one pixel, raster order).

pad          emits Hp x Wp pixels: the input pixels surrounded by a border of
             a constant value (border beats need no input).
swg          sliding window generator: KH-1 line buffers + a KH x KW window
             register; for every accepted pixel that completes a window at a
             stride position, emits the window as one beat in (kh, kw, c) order.
pool         per-channel max / sum over a window beat.
global_pool  per-channel sum over all H·W beats of a tensor → one beat.
flatten      collects the H·W beats of a tensor into one beat (HWC order).
"""

from __future__ import annotations

from digitalify.ir.node import IRNode
from digitalify.operators.base import VHDLOperator


def _bits(flags) -> str:
    return '"' + "".join("1" if f else "0" for f in flags) + '"'


def _advance(r: str, c: str, rows: str, cols: str, indent: str) -> list[str]:
    """Raster-order counter update."""
    return [
        f"{indent}if {c} = {cols} - 1 then",
        f"{indent}    {c} <= 0;",
        f"{indent}    if {r} = {rows} - 1 then {r} <= 0; else {r} <= {r} + 1; end if;",
        f"{indent}else",
        f"{indent}    {c} <= {c} + 1;",
        f"{indent}end if;",
    ]


class PadOperator(VHDLOperator):
    """Handler for ``pad`` IRNodes."""

    def node_latency(self, node: IRNode) -> int:
        return 1

    def core(self, node: IRNode) -> tuple[list[str], str]:
        t, l, b, r = node.attrs["pads"]
        h, w, ch = node.input_shape
        hp, wp = h + t + b, w + l + r
        xb = node.input_dtype.bits
        value = int(node.attrs["value"]) & ((1 << xb) - 1)
        word = 0
        for c in range(ch):
            word |= value << (c * xb)
        decl = [
            f"    constant HP : natural := {hp};",
            f"    constant WP : natural := {wp};",
            f"    constant PAD_PIX : std_logic_vector({ch * xb - 1} downto 0) := "
            + '"' + format(word, f"0{ch * xb}b") + '";',
            f"    constant ROW_IN : std_logic_vector(0 to HP - 1) := {_bits(t <= y < t + h for y in range(hp))};",
            f"    constant COL_IN : std_logic_vector(0 to WP - 1) := {_bits(l <= x < l + w for x in range(wp))};",
            "    signal y : natural range 0 to HP - 1 := 0;",
            "    signal x : natural range 0 to WP - 1 := 0;",
            "    signal inside : std_logic;",
        ]
        adv = _advance("y", "x", "HP", "WP", " " * 20)
        body = (
            "    inside   <= ROW_IN(y) and COL_IN(x);\n"
            "    in_ready <= ce and inside;\n\n"
            "    pad : process(clk)\n"
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                core_valid <= '0';\n"
            "                y <= 0;\n"
            "                x <= 0;\n"
            "            elsif ce = '1' then\n"
            "                core_valid <= '0';\n"
            "                if inside = '0' or in_valid = '1' then\n"
            "                    core_valid <= '1';\n"
            "                    if inside = '1' then core_data <= in_data; else core_data <= PAD_PIX; end if;\n"
            + "\n".join(adv) + "\n"
            "                end if;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process pad;\n\n"
        )
        return decl, body


class SWGOperator(VHDLOperator):
    """Handler for ``swg`` IRNodes (sliding window generator, no padding)."""

    def describe(self, node: IRNode) -> str:
        kh, kw = node.attrs["kernel"]
        return f"kernel {kh}x{kw}, stride {node.attrs['stride']}, {kh - 1} line buffer(s)"

    def core(self, node: IRNode) -> tuple[list[str], str]:
        kh, kw = node.attrs["kernel"]
        sh, sw = node.attrs["stride"]
        h, w, ch = node.input_shape
        xb = node.input_dtype.bits
        row_ok = [y >= kh - 1 and (y - kh + 1) % sh == 0 for y in range(h)]
        col_ok = [x >= kw - 1 and (x - kw + 1) % sw == 0 for x in range(w)]
        decl = [
            f"    constant H     : natural := {h};",
            f"    constant W     : natural := {w};",
            f"    constant KH    : natural := {kh};",
            f"    constant KW    : natural := {kw};",
            f"    constant PIX_W : natural := {ch * xb};",
            f"    constant ROW_OK : std_logic_vector(0 to H - 1) := {_bits(row_ok)};",
            f"    constant COL_OK : std_logic_vector(0 to W - 1) := {_bits(col_ok)};",
            "    subtype pix_t is std_logic_vector(PIX_W - 1 downto 0);",
            "    type win_t  is array(0 to KH * KW - 1) of pix_t;      -- index kh * KW + kw",
            "    type col_t  is array(0 to KH - 1) of pix_t;           -- col(i): row r - i",
            "    type line_t is array(0 to W - 1) of pix_t;",
            "    signal win : win_t := (others => (others => '0'));",
            "    signal r   : natural range 0 to H - 1 := 0;",
            "    signal c   : natural range 0 to W - 1 := 0;",
        ]
        decl += [f"    signal lb{i} : line_t;   -- row r - {i + 1}" for i in range(kh - 1)]

        stmts = ["                    col(0) := in_data;"]
        stmts += [f"                    col({i}) := lb{i - 1}(c);" for i in range(1, kh)]
        stmts += ["                    lb0(c) <= in_data;"] if kh > 1 else []
        stmts += [f"                    lb{i}(c) <= col({i});" for i in range(1, kh - 1)]
        for r_ in range(kh):
            for c_ in range(kw - 1):
                stmts.append(f"                    win({r_ * kw + c_}) <= win({r_ * kw + c_ + 1});")
            stmts.append(f"                    win({r_ * kw + kw - 1}) <= col({kh - 1 - r_});")
        stmts.append("                    core_valid <= ROW_OK(r) and COL_OK(c);")
        stmts += _advance("r", "c", "H", "W", " " * 20)

        body = (
            "    in_ready <= ce;\n\n"
            "    pack : for k in 0 to KH * KW - 1 generate\n"
            "        core_data((k + 1) * PIX_W - 1 downto k * PIX_W) <= win(k);\n"
            "    end generate pack;\n\n"
            "    swg : process(clk)\n"
            "        variable col : col_t;\n"
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                core_valid <= '0';\n"
            "                r <= 0;\n"
            "                c <= 0;\n"
            "            elsif ce = '1' then\n"
            "                core_valid <= '0';\n"
            "                if in_valid = '1' then\n"
            + "\n".join(stmts) + "\n"
            "                end if;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process swg;\n\n"
        )
        return decl, body


class PoolOperator(VHDLOperator):
    """Handler for ``pool`` IRNodes (per-channel max / sum over a window beat)."""

    def describe(self, node: IRNode) -> str:
        return f"{node.attrs['mode']} over {node.attrs['window']} pixels"

    def core(self, node: IRNode) -> tuple[list[str], str]:
        k = int(node.attrs["window"])
        ch = node.out_features
        xb = node.input_dtype.bits
        yb = node.output_dtype.bits
        vw = max(xb, yb) + 1
        load = lambda idx: self.load_element(node.input_dtype, "in_data", idx, "v")  # noqa: E731
        if node.attrs["mode"] == "max":
            red = "if v > m then m := v; end if;"
        else:
            red = "m := m + v;"
        stmts = [
            f"                for c in 0 to {ch - 1} loop",
            f"                    {load('c')}",
            f"                    m := resize(v, {vw});",
            f"                    for k in 1 to {k - 1} loop",
            f"                        {load(f'k * {ch} + c')}",
            f"                        {red}",
            "                    end loop;",
            f"                    {self.element('core_data', 'c', yb)} <= std_logic_vector(m({yb - 1} downto 0));",
            "                end loop;",
        ] if k > 1 else [
            f"                for c in 0 to {ch - 1} loop",
            f"                    {load('c')}",
            f"                    m := resize(v, {vw});",
            f"                    {self.element('core_data', 'c', yb)} <= std_logic_vector(m({yb - 1} downto 0));",
            "                end loop;",
        ]
        variables = (f"        variable v : signed({xb} downto 0);\n"
                     f"        variable m : signed({vw - 1} downto 0);\n")
        body = (
            "    in_ready <= ce;\n\n"
            "    pool : process(clk)\n" + variables +
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                core_valid <= '0';\n"
            "            elsif ce = '1' then\n"
            "                core_valid <= in_valid;\n"
            + "\n".join(stmts) + "\n"
            "            end if;\n"
            "        end if;\n"
            "    end process pool;\n\n"
        )
        return [], body


class GlobalPoolOperator(VHDLOperator):
    """Handler for ``global_pool`` IRNodes (per-channel sum over all beats)."""

    def node_latency(self, node: IRNode) -> int:
        return node.in_pixels

    def core(self, node: IRNode) -> tuple[list[str], str]:
        n = node.in_pixels
        ch = node.out_features
        xb = node.input_dtype.bits
        yb = node.output_dtype.bits
        sw = max(xb, yb) + 1
        decl = [
            f"    constant N : natural := {n};    -- beats per tensor",
            f"    type acc_t is array(0 to {ch - 1}) of signed({sw - 1} downto 0);",
            "    signal acc : acc_t;",
            "    signal cnt : natural range 0 to N - 1 := 0;",
        ]
        body = (
            "    in_ready <= ce;\n\n"
            "    gpool : process(clk)\n"
            f"        variable v : signed({xb} downto 0);\n"
            f"        variable s : signed({sw - 1} downto 0);\n"
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                core_valid <= '0';\n"
            "                cnt <= 0;\n"
            "            elsif ce = '1' then\n"
            "                core_valid <= '0';\n"
            "                if in_valid = '1' then\n"
            f"                    for c in 0 to {ch - 1} loop\n"
            f"                        {self.load_element(node.input_dtype, 'in_data', 'c', 'v')}\n"
            f"                        if cnt = 0 then s := resize(v, {sw}); else s := acc(c) + v; end if;\n"
            "                        acc(c) <= s;\n"
            f"                        {self.element('core_data', 'c', yb)} <= std_logic_vector(s({yb - 1} downto 0));\n"
            "                    end loop;\n"
            "                    if cnt = N - 1 then\n"
            "                        core_valid <= '1';\n"
            "                        cnt <= 0;\n"
            "                    else\n"
            "                        cnt <= cnt + 1;\n"
            "                    end if;\n"
            "                end if;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process gpool;\n\n"
        )
        return decl, body


class FlattenOperator(VHDLOperator):
    """Handler for ``flatten`` IRNodes (H·W beats → one beat, HWC order)."""

    def node_latency(self, node: IRNode) -> int:
        return node.in_pixels

    def core(self, node: IRNode) -> tuple[list[str], str]:
        n = node.in_pixels
        pix = node.in_features * node.input_dtype.bits
        decl = [
            f"    constant N     : natural := {n};    -- beats per tensor",
            f"    constant PIX_W : natural := {pix};",
            "    signal buf : std_logic_vector((N - 1) * PIX_W - 1 downto 0);   -- beats 0 .. N-2",
            "    signal cnt : natural range 0 to N - 1 := 0;",
        ]
        shift = ("buf <= in_data;" if n == 2 else
                 "buf <= in_data & buf((N - 1) * PIX_W - 1 downto PIX_W);")
        body = (
            "    in_ready <= ce;\n\n"
            "    flat : process(clk)\n"
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                core_valid <= '0';\n"
            "                cnt <= 0;\n"
            "            elsif ce = '1' then\n"
            "                core_valid <= '0';\n"
            "                if in_valid = '1' then\n"
            "                    if cnt = N - 1 then\n"
            "                        core_data  <= in_data & buf;\n"
            "                        core_valid <= '1';\n"
            "                        cnt <= 0;\n"
            "                    else\n"
            f"                        {shift}\n"
            "                        cnt <= cnt + 1;\n"
            "                    end if;\n"
            "                end if;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process flat;\n\n"
        )
        return decl, body
