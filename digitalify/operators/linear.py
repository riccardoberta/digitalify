"""
VHDL handler for the ``linear`` operator (MatMul / Gemm [+ bias] [+ ReLU]
[+ requantisation]).

Architecture
------------
* Weights, biases and requantisation constants are VHDL constants, sized to
  the actual integer values (e.g. 4-bit weights stay 4-bit).
* MAC: fully combinational, ``sum_j = Σ_i W[j][i] · x[i]`` summed by a
  balanced adder tree at ``SUM_W`` bits, then ``acc_j = sum_j · 2^F + BIAS[j]`` at ``ACC_W`` bits.  Both widths come
  from interval analysis of the real weights and input range, so the
  accumulator can never overflow.
* Output register: optional ReLU, then either the
  accumulator itself or the requantisation

      q = clip( round( acc · MULT[j] / 2^SHIFT[j] ) )

  with the rounding mode of the QONNX ``Quant`` node, bit-exact with
  ``digitalify.testing.reference``.

Optional pipeline registers (``pipeline`` option, one clock each):

    products -[products]- tree level 1 -[tree]- level 2 -[tree]- ... last level
        -> +bias -[acc]- ReLU -> x MULT -[mult]- round/clip -[out]-

``tree`` registers every ``tree_every`` levels (``"tree:K"``), never after
the last level; ``mult`` exists only when the layer requantises.  Latency is
1 + (number of enabled registers); throughput stays one vector per clock.
"""

from __future__ import annotations

from digitalify.ir.node import IRNode
from digitalify.operators.base import VHDLOperator

# VHDL condition (on p, q, r, half) under which floor(p / 2^n) is incremented.
_ROUND_UP_CONDITION = {
    "FLOOR": None,
    "CEIL": "r /= 0",
    "ROUND": "r > half or (r = half and r /= 0 and q(0) = '1')",
    "HALF_UP": "r > half or (r = half and r /= 0 and p >= 0)",
    "HALF_DOWN": "r > half or (r = half and r /= 0 and p < 0)",
    "UP": "r /= 0 and p > 0",
    "DOWN": "r /= 0 and p < 0",
}


class LinearOperator(VHDLOperator):
    """Handler for ``linear`` IRNodes."""

    latency = 1
    clocked = True

    def node_latency(self, node: IRNode) -> int:
        has_requant = node.attrs.get("requant") is not None
        return (1 + len(self._mac_cuts(node.attrs["weight"].shape[1]))
                + ("acc" in self.pipeline)
                + ("mult" in self.pipeline and has_requant))

    def _mac_cuts(self, in_f: int) -> list[int]:
        """Register positions inside the MAC: 0 = after the products,
        L >= 1 = after adder-tree level L (the last level is never cut: the
        bias adder follows it, and "acc" registers that)."""
        depth = len(self._tree_sizes(in_f)) - 1
        cuts = [0] if "products" in self.pipeline else []
        if "tree" in self.pipeline:
            cuts += list(range(self.pipeline.tree_every, depth, self.pipeline.tree_every))
        return cuts

    @staticmethod
    def _tree_sizes(n: int) -> list[int]:
        """Number of partial sums before and after each adder-tree level."""
        sizes = [n]
        while sizes[-1] > 1:
            sizes.append((sizes[-1] + 1) // 2)
        return sizes

    def architecture(self, node: IRNode, entity: str) -> str:
        a = node.attrs
        weight = a["weight"]
        out_f, in_f = weight.shape
        xb = node.input_dtype.bits
        x_signed = node.input_dtype.signed
        wb = a["weight_dtype"].bits
        sb = a["sum_dtype"].bits
        ab = a["acc_dtype"].bits
        yb = node.output_dtype.bits
        frac = int(a.get("bias_frac_bits", 0))
        rq = a.get("requant")

        def w_lit(v: int) -> str:
            return f"to_signed({int(v)}, {wb})" if wb <= 31 else self.bit_literal(v, wb)

        rows = [self.aggregate([w_lit(v) for v in weight[j]], " " * 12) for j in range(out_f)]
        decl = [
            f"    constant IN_F  : natural := {in_f};",
            f"    constant OUT_F : natural := {out_f};",
            f"    constant X_W   : natural := {xb};    -- input element ({node.input_dtype})",
            f"    constant W_W   : natural := {wb};    -- weight ({a['weight_dtype']})",
            f"    constant SUM_W : natural := {sb};    -- adder tree",
            f"    constant ACC_W : natural := {ab};    -- sum * 2^F + bias",
            f"    constant Y_W   : natural := {yb};    -- output element ({node.output_dtype})",
            f"    constant F     : natural := {frac};    -- fractional bias bits",
            "",
            "    type weight_row_t is array(0 to IN_F - 1) of signed(W_W - 1 downto 0);",
            "    type weight_t     is array(0 to OUT_F - 1) of weight_row_t;",
            f"    constant WEIGHTS : weight_t := {self.aggregate(rows)};",
            "",
            "    type sum_vec_t is array(0 to IN_F - 1) of signed(SUM_W - 1 downto 0);",
            "    type acc_t is array(0 to OUT_F - 1) of signed(ACC_W - 1 downto 0);",
            "    constant BIASES : acc_t := "
            + self.aggregate([self.bit_literal(v, ab) for v in a["bias"]]) + ";",
            "    signal acc : acc_t;",
        ]

        if rq is not None:
            mults = [int(m) for m in rq["multiplier"]]
            shifts = [int(n) for n in rq["shift"]]
            mb = max(m.bit_length() for m in mults) + 1           # + sign bit
            pw = max(ab + mb, max(shifts) + 2)
            decl += [
                "",
                "    -- Requantisation: q = clip(round(acc * MULT / 2^SHIFT))",
                f"    -- rounding mode {rq['rounding']}, "
                f"{'exact' if rq['exact'] else 'fixed-point approximation of the'} scale ratio",
                f"    constant P_W : natural := {pw};",
                f"    type mult_t is array(0 to OUT_F - 1) of signed({mb - 1} downto 0);",
                "    constant MULT : mult_t := "
                + self.aggregate([self.bit_literal(m, mb) for m in mults]) + ";",
                "    type wide_t is array(0 to OUT_F - 1) of signed(P_W - 1 downto 0);",
                "    constant SHIFT_MASK : wide_t := "
                + self.aggregate([self.bit_literal((1 << n) - 1, pw) for n in shifts]) + ";",
                "    constant SHIFT_HALF : wide_t := "
                + self.aggregate([self.bit_literal((1 << (n - 1)) if n else 0, pw) for n in shifts]) + ";",
                "    type shift_t is array(0 to OUT_F - 1) of natural;",
                "    constant SHIFT : shift_t := " + self.aggregate([str(n) for n in shifts]) + ";",
                f"    constant CLIP_MIN : signed(P_W - 1 downto 0) := {self.bit_literal(rq['clip_min'], pw)};",
                f"    constant CLIP_MAX : signed(P_W - 1 downto 0) := {self.bit_literal(rq['clip_max'], pw)};",
            ]

        if x_signed:
            load_x = f"x := resize(signed({self.element('data_i', 'i', 'X_W')}), X_W + 1);"
        else:
            load_x = (f"xs := '0' & {self.element('data_i', 'i', 'X_W')};\n"
                      "                x  := signed(xs);")

        reg_acc = "acc" in self.pipeline
        reg_mult = "mult" in self.pipeline and rq is not None
        if reg_acc:
            decl += ["    signal acc_r   : acc_t;", "    signal valid_a : std_logic;"]
        if reg_mult:
            decl += ["    signal prod_r  : wide_t;", "    signal valid_p : std_logic;"]

        body = self.header(node, entity) + self.entity(node, entity)
        body += f"architecture rtl of {entity} is\n\n" + "\n".join(decl) + "\n\nbegin\n\n"
        body += self._mac(in_f, load_x)
        if self._pending_decl:
            marker = "\n\nbegin\n\n"
            head, tail = body.split(marker, 1)
            body = head + "\n" + "\n".join(self._pending_decl) + marker + tail
        acc_src, valid_src = "acc", self._mac_valid
        if reg_acc:
            body += self._register("acc_reg", "Pipeline register: accumulator",
                                   "valid_a", valid_src, "acc_r <= (others => (others => '0'));",
                                   "acc_r <= acc;")
            acc_src, valid_src = "acc_r", "valid_a"

        relu = ["                    if a < 0 then a := (others => '0'); end if;"] if a.get("relu") else []
        out_slice = self.element("data_o", "j", "Y_W")

        if reg_mult:
            body += self._register(
                "mult_reg", "Pipeline register: " + ("ReLU + " if relu else "") + "requantisation product",
                "valid_p", valid_src, "prod_r <= (others => (others => '0'));",
                "for j in 0 to OUT_F - 1 loop\n"
                f"                    a := {acc_src}(j);\n"
                + "".join(line + "\n" for line in relu) +
                "                    prod_r(j) <= resize(a * MULT(j), P_W);\n"
                "                end loop;",
                variables="        variable a : signed(ACC_W - 1 downto 0);\n",
            )
            valid_src = "valid_p"
            stage = ["                    p    := prod_r(j);"]
            variables = "        variable p, q, r, half : signed(P_W - 1 downto 0);\n"
            relu_here = False
        else:
            stage = [f"                    a := {acc_src}(j);"] + relu
            relu_here = bool(relu)
            variables = "        variable a : signed(ACC_W - 1 downto 0);\n"
            if rq is not None:
                variables += "        variable p, q, r, half : signed(P_W - 1 downto 0);\n"
                stage.append("                    p    := resize(a * MULT(j), P_W);")

        if rq is None:
            stage.append(f"                    {out_slice} <= std_logic_vector(a(Y_W - 1 downto 0));")
        else:
            stage += [
                "                    q    := shift_right(p, SHIFT(j));     -- floor",
                "                    r    := p and SHIFT_MASK(j);          -- discarded bits",
                "                    half := SHIFT_HALF(j);",
            ]
            cond = _ROUND_UP_CONDITION[rq["rounding"]]
            if cond:
                stage.append(f"                    if {cond} then q := q + 1; end if;")
            stage += [
                "                    if q > CLIP_MAX then q := CLIP_MAX;",
                "                    elsif q < CLIP_MIN then q := CLIP_MIN;",
                "                    end if;",
                f"                    {out_slice} <= std_logic_vector(q(Y_W - 1 downto 0));",
            ]

        what = (" + ReLU" if relu_here else "") + (
            " + rounding/clipping" if reg_mult else " + requantisation" if rq is not None else "")
        body += (
            "    -- Output register" + what + "\n"
            "    outp : process(clk)\n" + variables +
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                valid_o <= '0';\n"
            "                data_o  <= (others => '0');\n"
            "            else\n"
            f"                valid_o <= {valid_src};\n"
            "                for j in 0 to OUT_F - 1 loop\n"
            + "\n".join(stage) + "\n"
            "                end loop;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process outp;\n\n"
            "end architecture rtl;\n"
        )
        return body

    def _mac(self, in_f: int, load_x: str) -> str:
        """
        Products + balanced adder tree + bias, split by the "products" /
        "tree" pipeline registers into segments.  Segment s reads the
        previous register (or data_i), computes with a variable ``t`` and
        drives either the next register's input or ``acc``.
        """
        sizes = self._tree_sizes(in_f)
        depth = len(sizes) - 1
        cuts = self._mac_cuts(in_f)
        bounds = [None] + cuts + [depth]          # segment s spans (bounds[s], bounds[s+1]]
        self._pending_decl = []
        out = ""
        valid = "valid_i"
        for s_idx in range(len(bounds) - 1):
            start, end = bounds[s_idx], bounds[s_idx + 1]
            last = s_idx == len(bounds) - 2
            src = "data_i" if start is None else f"mac_r{s_idx - 1}"
            lines = []
            if start is None:
                lines += [
                    "            for i in 0 to IN_F - 1 loop",
                    f"                {load_x}",
                    "                t(i) := resize(x * WEIGHTS(j)(i), SUM_W);",
                    "            end loop;",
                ]
                first_level = 1
            else:
                lines.append(f"            for i in 0 to {sizes[start] - 1} loop "
                             f"t(i) := {src}(j)(i); end loop;")
                first_level = start + 1
            for level in range(first_level, end + 1):
                lines.append(self._tree_level(sizes[level - 1]).rstrip("\n"))
            if last:
                lines.append("            acc(j) <= shift_left(resize(t(0), ACC_W), F) + BIASES(j);")
            else:
                n = sizes[end]
                self._pending_decl += [
                    f"    type mac{s_idx}_vec_t is array(0 to {n - 1}) of signed(SUM_W - 1 downto 0);",
                    f"    type mac{s_idx}_t is array(0 to OUT_F - 1) of mac{s_idx}_vec_t;",
                    f"    signal mac_c{s_idx}, mac_r{s_idx} : mac{s_idx}_t;",
                    f"    signal mac_v{s_idx} : std_logic;",
                ]
                lines.append(f"            for i in 0 to {n - 1} loop mac_c{s_idx}(j)(i) <= t(i); end loop;")

            parts = ["products"] if start is None else []
            lo = 1 if start is None else start + 1
            if end >= lo:
                parts.append(f"adder tree level{'s' if end > lo else ''} "
                             + (f"{lo}..{end}" if end > lo else f"{lo}"))
            if last:
                parts.append("bias")
            what = " + ".join(parts) or "pass-through"
            header = (f"    -- MAC segment {s_idx}: {what}" if len(bounds) > 2 else
                      "    -- Combinational multiply-accumulate: products, then a balanced\n"
                      "    -- adder tree (depth ceil(log2(IN_F)) instead of an IN_F-long chain)")
            variables = "        variable t  : sum_vec_t;\n"
            if start is None:
                variables = ("        variable xs : std_logic_vector(X_W downto 0);\n"
                             "        variable x  : signed(X_W downto 0);\n") + variables
            label = "mac" if len(bounds) == 2 else f"mac_s{s_idx}"
            out += (
                f"{header}\n"
                f"    {label} : process({src})\n" + variables +
                "    begin\n"
                "        for j in 0 to OUT_F - 1 loop\n"
                + "\n".join(lines) + "\n"
                "        end loop;\n"
                f"    end process {label};\n\n"
            )
            if not last:
                out += self._register(
                    f"mac_reg{s_idx}", f"Pipeline register {s_idx} inside the MAC",
                    f"mac_v{s_idx}", valid,
                    f"mac_r{s_idx} <= (others => (others => (others => '0')));",
                    f"mac_r{s_idx} <= mac_c{s_idx};")
                valid = f"mac_v{s_idx}"
        self._mac_valid = valid
        return out

    @staticmethod
    def _tree_level(n: int) -> str:
        """One adder-tree level: t(0 .. ceil(n/2)-1) from t(0 .. n-1), in place.

        t(k) is written from t(2k), t(2k+1); both indices are >= k, so no
        value is overwritten before it is read.
        """
        half = n // 2
        line = (f"            for k in 0 to {half - 1} loop "
                "t(k) := t(2 * k) + t(2 * k + 1); end loop;\n")
        if n % 2:
            line += f"            t({half}) := t({n - 1});\n"
        return line

    @staticmethod
    def _adder_tree(n: int) -> str:
        """In-place pairwise reduction of t(0 .. n-1) into t(0), one line per level.

        Level k writes t(k) from t(2k), t(2k+1); both indices are >= k, so
        no value is overwritten before it is read.  Two's-complement sums
        wrap consistently and the final sum fits SUM_W, so the order of the
        additions does not change the result.
        """
        sizes = LinearOperator._tree_sizes(n)
        return "".join(LinearOperator._tree_level(m) for m in sizes[:-1])

    @staticmethod
    def _register(label: str, comment: str, valid_out: str, valid_in: str,
                  reset: str, load: str, variables: str = "") -> str:
        return (
            f"    -- {comment}\n"
            f"    {label} : process(clk)\n" + variables +
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            f"                {valid_out} <= '0';\n"
            f"                {reset}\n"
            "            else\n"
            f"                {valid_out} <= {valid_in};\n"
            f"                {load}\n"
            "            end if;\n"
            "        end if;\n"
            f"    end process {label};\n\n"
        )
