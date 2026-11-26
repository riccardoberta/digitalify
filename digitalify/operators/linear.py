"""
VHDL handler for the ``linear`` stage (MatMul / Gemm / Conv after the window
generator [+ bias, BatchNorm] [+ ReLU] [+ requantisation]).

Each input beat ``x`` (``IN_F`` elements) gives one output beat:

    acc_j = (Σ_i W[j][i] · x[i]) · 2^F + BIAS[j]
    y_j   = clip(round(relu(acc_j) · MULT[j] / 2^SHIFT[j]))   or relu(acc_j)

Two architectures, chosen by the ``folding`` option ``(PE, SIMD)``:

Fully parallel (default, ``PE = OUT_F`` and ``SIMD = IN_F``)
    One beat per clock.  Products, balanced adder tree, bias, output register,
    with optional pipeline registers (``pipeline`` option, one clock each):

        products -[products]- tree level 1 -[tree]- ... last level
            -> +bias -[acc]- ReLU -> x MULT -[mult]- round/clip -[out]-

    ``tree`` registers every ``tree_every`` levels (``"tree:K"``), never after
    the last level; ``mult`` exists only when the layer requantises.

Folded (matrix-vector unit)
    ``PE`` neurons x ``SIMD`` inputs per clock: ``PE·SIMD`` multipliers,
    ``NF·SF`` clocks per beat (``NF = OUT_F/PE``, ``SF = IN_F/SIMD``).  The
    input beat is latched and rotated by SIMD elements per clock; weights
    come from a constant ROM indexed by the fold counter; a finalisation
    register adds the bias, applies ReLU and requantises PE neurons per clock.
    The ``pipeline`` option does not apply.

Widths come from interval analysis of the real weights and input range, so
the accumulator can never overflow; results are bit-exact with
``digitalify.testing.reference``.
"""

from __future__ import annotations

from digitalify.ir.node import IRNode
from digitalify.operators.base import VHDLOperator


class LinearOperator(VHDLOperator):
    """Handler for ``linear`` IRNodes."""

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def fold(self, node: IRNode) -> tuple[int, int]:
        """``(PE, SIMD)`` for *node*, validated."""
        out_f, in_f = node.attrs["weight"].shape
        if self.folding is None:
            return out_f, in_f
        pe, simd = self.folding
        if pe < 1 or out_f % pe or simd < 1 or in_f % simd:
            raise ValueError(
                f"Node '{node.name}': folding (PE={pe}, SIMD={simd}) must divide "
                f"(out_features={out_f}, in_features={in_f})."
            )
        return pe, simd

    def folded(self, node: IRNode) -> bool:
        return self.fold(node) != tuple(node.attrs["weight"].shape)

    def node_latency(self, node: IRNode) -> int:
        if self.folded(node):
            out_f, in_f = node.attrs["weight"].shape
            pe, simd = self.fold(node)
            return 1 + (out_f // pe) * (in_f // simd) + 1
        has_requant = node.attrs.get("requant") is not None
        return (1 + len(self._mac_cuts(node.attrs["weight"].shape[1]))
                + ("acc" in self.pipeline)
                + ("mult" in self.pipeline and has_requant))

    def describe(self, node: IRNode) -> str:
        out_f, in_f = node.attrs["weight"].shape
        if self.folded(node):
            pe, simd = self.fold(node)
            return (f"folded: PE={pe}, SIMD={simd}, {(out_f // pe) * (in_f // simd)} clocks "
                    f"per beat, {pe * simd} multipliers")
        return "fully parallel" + (f", pipeline: {self.pipeline.describe()}" if self.pipeline else "")

    def _mac_cuts(self, in_f: int) -> list[int]:
        """Register positions inside the MAC: 0 = after the products,
        L >= 1 = after adder-tree level L (the last level is never cut: the
        bias adder follows it, and "acc" registers that)."""
        depth = len(self._tree_sizes(in_f)) - 1
        cuts = [0] if "products" in self.pipeline else []
        if "tree" in self.pipeline:
            cuts += list(range(self.pipeline.tree_every, depth, self.pipeline.tree_every))
        return cuts

    # ------------------------------------------------------------------
    # Shared declarations
    # ------------------------------------------------------------------

    def _common_decl(self, node: IRNode) -> list[str]:
        a = node.attrs
        out_f, in_f = a["weight"].shape
        ab = a["acc_dtype"].bits
        decl = [
            f"    constant IN_F  : natural := {in_f};",
            f"    constant OUT_F : natural := {out_f};",
            f"    constant X_W   : natural := {node.input_dtype.bits};    -- input element ({node.input_dtype})",
            f"    constant W_W   : natural := {a['weight_dtype'].bits};    -- weight ({a['weight_dtype']})",
            f"    constant SUM_W : natural := {a['sum_dtype'].bits};    -- adder tree",
            f"    constant ACC_W : natural := {ab};    -- sum * 2^F + bias",
            f"    constant Y_W   : natural := {node.output_dtype.bits};    -- output element ({node.output_dtype})",
            f"    constant F     : natural := {int(a.get('bias_frac_bits', 0))};    -- fractional bias bits",
            "",
            "    type acc_t is array(0 to OUT_F - 1) of signed(ACC_W - 1 downto 0);",
            "    constant BIASES : acc_t := "
            + self.aggregate([self.bit_literal(v, ab) for v in a["bias"]]) + ";",
        ]
        if a.get("requant") is not None:
            decl += self.requant_decl(a["requant"], ab, out_f)
        return decl

    def _w_lit(self, node: IRNode, v: int) -> str:
        wb = node.attrs["weight_dtype"].bits
        return f"to_signed({int(v)}, {wb})" if wb <= 31 else self.bit_literal(v, wb)

    def core(self, node: IRNode) -> tuple[list[str], str]:
        return self._folded_core(node) if self.folded(node) else self._parallel_core(node)

    # ------------------------------------------------------------------
    # Fully parallel
    # ------------------------------------------------------------------

    def _parallel_core(self, node: IRNode) -> tuple[list[str], str]:
        a = node.attrs
        weight = a["weight"]
        out_f, in_f = weight.shape
        rq = a.get("requant")

        rows = [self.aggregate([self._w_lit(node, v) for v in weight[j]], " " * 12) for j in range(out_f)]
        decl = self._common_decl(node) + [
            "",
            "    type weight_row_t is array(0 to IN_F - 1) of signed(W_W - 1 downto 0);",
            "    type weight_t     is array(0 to OUT_F - 1) of weight_row_t;",
            f"    constant WEIGHTS : weight_t := {self.aggregate(rows)};",
            "    type sum_vec_t is array(0 to IN_F - 1) of signed(SUM_W - 1 downto 0);",
            "    signal acc : acc_t;",
        ]
        load_x = self.load_element(node.input_dtype, "in_data", "i", "x")

        reg_acc = "acc" in self.pipeline
        reg_mult = "mult" in self.pipeline and rq is not None
        if reg_acc:
            decl += ["    signal acc_r   : acc_t;", "    signal valid_a : std_logic := '0';"]
        if reg_mult:
            decl += ["    signal prod_r  : wide_t;", "    signal valid_p : std_logic := '0';"]

        body = "    in_ready <= ce;\n\n" + self._mac(in_f, load_x)
        decl += self._pending_decl
        acc_src, valid_src = "acc", self._mac_valid
        if reg_acc:
            body += self._register("acc_reg", "Pipeline register: accumulator",
                                   "valid_a", valid_src, "acc_r <= acc;")
            acc_src, valid_src = "acc_r", "valid_a"

        relu = ["                    if a < 0 then a := (others => '0'); end if;"] if a.get("relu") else []
        out_slice = self.element("core_data", "j", "Y_W")

        if reg_mult:
            body += self._register(
                "mult_reg", "Pipeline register: " + ("ReLU + " if relu else "") + "requantisation product",
                "valid_p", valid_src,
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
            stage += self.requant_round(rq, "j", " " * 20)
            stage.append(f"                    {out_slice} <= std_logic_vector(q(Y_W - 1 downto 0));")

        what = (" + ReLU" if relu_here else "") + (
            " + rounding/clipping" if reg_mult else " + requantisation" if rq is not None else "")
        body += (
            "    -- Output register" + what + "\n"
            "    outp : process(clk)\n" + variables +
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                core_valid <= '0';\n"
            "            elsif ce = '1' then\n"
            f"                core_valid <= {valid_src};\n"
            "                for j in 0 to OUT_F - 1 loop\n"
            + "\n".join(stage) + "\n"
            "                end loop;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process outp;\n\n"
        )
        return decl, body

    def _mac(self, in_f: int, load_x: str) -> str:
        """
        Products + balanced adder tree + bias, split by the "products" /
        "tree" pipeline registers into segments.  Segment s reads the
        previous register (or in_data), computes with a variable ``t`` and
        drives either the next register's input or ``acc``.
        """
        sizes = self._tree_sizes(in_f)
        depth = len(sizes) - 1
        cuts = self._mac_cuts(in_f)
        bounds = [None] + cuts + [depth]          # segment s spans (bounds[s], bounds[s+1]]
        self._pending_decl = []
        out = ""
        valid = "in_valid"
        for s_idx in range(len(bounds) - 1):
            start, end = bounds[s_idx], bounds[s_idx + 1]
            last = s_idx == len(bounds) - 2
            src = "in_data" if start is None else f"mac_r{s_idx - 1}"
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
                    f"    signal mac_v{s_idx} : std_logic := '0';",
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
                variables = "        variable x  : signed(X_W downto 0);\n" + variables
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
                    f"mac_v{s_idx}", valid, f"mac_r{s_idx} <= mac_c{s_idx};")
                valid = f"mac_v{s_idx}"
        self._mac_valid = valid
        return out

    @staticmethod
    def _tree_sizes(n: int) -> list[int]:
        """Number of partial sums before and after each adder-tree level."""
        sizes = [n]
        while sizes[-1] > 1:
            sizes.append((sizes[-1] + 1) // 2)
        return sizes

    @staticmethod
    def _tree_level(n: int, var: str = "t", indent: str = " " * 12) -> str:
        """One adder-tree level: t(0 .. ceil(n/2)-1) from t(0 .. n-1), in place.

        t(k) is written from t(2k), t(2k+1); both indices are >= k, so no
        value is overwritten before it is read.  Two's-complement sums wrap
        consistently and the final sum fits SUM_W, so the order of the
        additions does not change the result.
        """
        half = n // 2
        line = (f"{indent}for k in 0 to {half - 1} loop "
                f"{var}(k) := {var}(2 * k) + {var}(2 * k + 1); end loop;\n")
        if n % 2:
            line += f"{indent}{var}({half}) := {var}({n - 1});\n"
        return line

    @staticmethod
    def _adder_tree(n: int, var: str = "t", indent: str = " " * 12) -> str:
        """Every level of the in-place reduction of var(0 .. n-1) into var(0)."""
        sizes = LinearOperator._tree_sizes(n)
        return "".join(LinearOperator._tree_level(m, var, indent) for m in sizes[:-1])

    @staticmethod
    def _register(label: str, comment: str, valid_out: str, valid_in: str,
                  load: str, variables: str = "") -> str:
        return (
            f"    -- {comment}\n"
            f"    {label} : process(clk)\n" + variables +
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            f"                {valid_out} <= '0';\n"
            "            elsif ce = '1' then\n"
            f"                {valid_out} <= {valid_in};\n"
            f"                {load}\n"
            "            end if;\n"
            "        end if;\n"
            f"    end process {label};\n\n"
        )

    # ------------------------------------------------------------------
    # Folded matrix-vector unit
    # ------------------------------------------------------------------

    def _folded_core(self, node: IRNode) -> tuple[list[str], str]:
        a = node.attrs
        weight = a["weight"]
        out_f, in_f = weight.shape
        pe, simd = self.fold(node)
        nf, sf = out_f // pe, in_f // simd
        rq = a.get("requant")

        rom = []
        for g in range(nf):
            for s in range(sf):
                lanes = [self.aggregate([self._w_lit(node, weight[g * pe + p, s * simd + k])
                                         for k in range(simd)], " " * 16) for p in range(pe)]
                rom.append(self.aggregate(lanes, " " * 12))
        decl = self._common_decl(node) + [
            "",
            f"    constant PE   : natural := {pe};",
            f"    constant SIMD : natural := {simd};",
            f"    constant NF   : natural := {nf};    -- neuron folds",
            f"    constant SF   : natural := {sf};    -- synapse folds",
            "",
            "    -- Weight ROM: WROM(nf * SF + sf)(pe)(simd) = W[nf*PE + pe][sf*SIMD + simd]",
            "    type wlane_t is array(0 to SIMD - 1) of signed(W_W - 1 downto 0);",
            "    type wword_t is array(0 to PE - 1) of wlane_t;",
            "    type wrom_t  is array(0 to NF * SF - 1) of wword_t;",
            f"    constant WROM : wrom_t := {self.aggregate(rom)};",
            "",
            "    type simd_vec_t is array(0 to SIMD - 1) of signed(SUM_W - 1 downto 0);",
            "    type pe_sum_t   is array(0 to PE - 1) of signed(SUM_W - 1 downto 0);",
            "    type pe_out_t   is array(0 to PE - 1) of std_logic_vector(Y_W - 1 downto 0);",
            "    signal busy      : std_logic := '0';",
            "    signal x_reg     : std_logic_vector(IN_F * X_W - 1 downto 0);",
            "    signal widx      : natural range 0 to NF * SF - 1 := 0;",
            "    signal sf_cnt    : natural range 0 to SF - 1 := 0;",
            "    signal nf_cnt    : natural range 0 to NF - 1 := 0;",
            "    signal accs      : pe_sum_t;",
            "    signal fin_valid : std_logic := '0';",
            "    signal fin_nf    : natural range 0 to NF - 1 := 0;",
            "    signal fin_sum   : pe_sum_t;",
            "    signal out_buf   : std_logic_vector(OUT_F * Y_W - 1 downto 0) := (others => '0');",
        ]

        load_x = self.load_element(node.input_dtype, "x_reg", "s", "x")
        tree = self._adder_tree(simd, "t", " " * 20)
        rotate = ("                x_reg <= x_reg(SIMD * X_W - 1 downto 0) & "
                  "x_reg(IN_F * X_W - 1 downto SIMD * X_W);\n") if sf > 1 else ""
        fin = [
            "                    j := fin_nf * PE + k;",
            "                    a := shift_left(resize(fin_sum(k), ACC_W), F) + BIASES(j);",
        ]
        if a.get("relu"):
            fin.append("                    if a < 0 then a := (others => '0'); end if;")
        if rq is None:
            fin.append("                    res(k) := std_logic_vector(a(Y_W - 1 downto 0));")
        else:
            fin.append("                    p := resize(a * MULT(j), P_W);")
            fin += self.requant_round(rq, "j", " " * 20)
            fin.append("                    res(k) := std_logic_vector(q(Y_W - 1 downto 0));")
        variables = (
            "        variable x    : signed(X_W downto 0);\n"
            "        variable t    : simd_vec_t;\n"
            "        variable psum : signed(SUM_W - 1 downto 0);\n"
            "        variable a    : signed(ACC_W - 1 downto 0);\n"
            "        variable j    : natural range 0 to OUT_F - 1;\n"
            "        variable res  : pe_out_t;\n"
            "        variable ob   : std_logic_vector(OUT_F * Y_W - 1 downto 0);\n"
        )
        if rq is not None:
            variables += "        variable p, q, r, half : signed(P_W - 1 downto 0);\n"

        body = (
            "    in_ready <= ce and not busy;\n\n"
            "    -- Folded matrix-vector unit: PE neurons x SIMD inputs per clock\n"
            "    mvau : process(clk)\n" + variables +
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                busy       <= '0';\n"
            "                fin_valid  <= '0';\n"
            "                core_valid <= '0';\n"
            "            elsif ce = '1' then\n"
            "                core_valid <= '0';\n"
            "                fin_valid  <= '0';\n"
            "\n"
            "                -- Finalise the neuron group accumulated in the previous clock\n"
            "                if fin_valid = '1' then\n"
            "                    for k in 0 to PE - 1 loop\n"
            + "\n".join(fin) + "\n"
            "                    end loop;\n"
            "                    ob := out_buf;\n"
            "                    for g in 0 to NF - 1 loop\n"
            "                        if g = fin_nf then\n"
            "                            for k in 0 to PE - 1 loop\n"
            "                                ob(((g * PE + k) + 1) * Y_W - 1 downto (g * PE + k) * Y_W) := res(k);\n"
            "                            end loop;\n"
            "                        end if;\n"
            "                    end loop;\n"
            "                    out_buf <= ob;\n"
            "                    if fin_nf = NF - 1 then\n"
            "                        core_valid <= '1';\n"
            "                        core_data  <= ob;\n"
            "                    end if;\n"
            "                end if;\n"
            "\n"
            "                if busy = '0' then\n"
            "                    if in_valid = '1' then\n"
            "                        x_reg  <= in_data;\n"
            "                        busy   <= '1';\n"
            "                        widx   <= 0;\n"
            "                        sf_cnt <= 0;\n"
            "                        nf_cnt <= 0;\n"
            "                    end if;\n"
            "                else\n"
            "                    for k in 0 to PE - 1 loop\n"
            "                        for s in 0 to SIMD - 1 loop\n"
            f"                            {load_x}\n"
            "                            t(s) := resize(x * WROM(widx)(k)(s), SUM_W);\n"
            "                        end loop;\n"
            + tree +
            "                        if sf_cnt = 0 then psum := t(0); else psum := accs(k) + t(0); end if;\n"
            "                        if sf_cnt = SF - 1 then fin_sum(k) <= psum; else accs(k) <= psum; end if;\n"
            "                    end loop;\n"
            + rotate.replace("                x_reg", "                    x_reg") +
            "                    if sf_cnt = SF - 1 then\n"
            "                        sf_cnt    <= 0;\n"
            "                        fin_valid <= '1';\n"
            "                        fin_nf    <= nf_cnt;\n"
            "                        if nf_cnt = NF - 1 then\n"
            "                            busy <= '0';\n"
            "                        else\n"
            "                            nf_cnt <= nf_cnt + 1;\n"
            "                        end if;\n"
            "                    else\n"
            "                        sf_cnt <= sf_cnt + 1;\n"
            "                    end if;\n"
            "                    if widx /= NF * SF - 1 then widx <= widx + 1; end if;\n"
            "                end if;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process mvau;\n\n"
        )
        return decl, body
