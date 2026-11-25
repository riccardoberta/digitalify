"""
VHDL handler for the ``linear`` operator (MatMul / Gemm [+ bias] [+ ReLU]
[+ requantisation]).

Architecture
------------
* Weights, biases and requantisation constants are VHDL constants, sized to
  the actual integer values (e.g. 4-bit weights stay 4-bit).
* MAC: fully combinational, ``sum_j = Σ_i W[j][i] · x[i]`` at ``SUM_W`` bits,
  then ``acc_j = sum_j · 2^F + BIAS[j]`` at ``ACC_W`` bits.  Both widths come
  from interval analysis of the real weights and input range, so the
  accumulator can never overflow.
* Output register (latency 1 clock): optional ReLU, then either the
  accumulator itself or the requantisation

      q = clip( round( acc · MULT[j] / 2^SHIFT[j] ) )

  with the rounding mode of the QONNX ``Quant`` node, bit-exact with
  ``digitalify.testing.reference``.
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
            f"    constant SUM_W : natural := {sb};    -- MAC chain",
            f"    constant ACC_W : natural := {ab};    -- sum * 2^F + bias",
            f"    constant Y_W   : natural := {yb};    -- output element ({node.output_dtype})",
            f"    constant F     : natural := {frac};    -- fractional bias bits",
            "",
            "    type weight_row_t is array(0 to IN_F - 1) of signed(W_W - 1 downto 0);",
            "    type weight_t     is array(0 to OUT_F - 1) of weight_row_t;",
            f"    constant WEIGHTS : weight_t := {self.aggregate(rows)};",
            "",
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

        body = self.header(node, entity) + self.entity(node, entity)
        body += f"architecture rtl of {entity} is\n\n" + "\n".join(decl) + "\n\nbegin\n\n"
        body += (
            "    -- Combinational multiply-accumulate\n"
            "    mac : process(data_i)\n"
            "        variable xs  : std_logic_vector(X_W downto 0);\n"
            "        variable x   : signed(X_W downto 0);\n"
            "        variable s   : signed(SUM_W - 1 downto 0);\n"
            "    begin\n"
            "        for j in 0 to OUT_F - 1 loop\n"
            "            s := (others => '0');\n"
            "            for i in 0 to IN_F - 1 loop\n"
            f"                {load_x}\n"
            "                s := s + resize(x * WEIGHTS(j)(i), SUM_W);\n"
            "            end loop;\n"
            "            acc(j) <= shift_left(resize(s, ACC_W), F) + BIASES(j);\n"
            "        end loop;\n"
            "    end process mac;\n\n"
        )

        stage = ["                    a := acc(j);"]
        if a.get("relu"):
            stage.append("                    if a < 0 then a := (others => '0'); end if;")
        out_slice = self.element("data_o", "j", "Y_W")
        if rq is None:
            stage.append(f"                    {out_slice} <= std_logic_vector(a(Y_W - 1 downto 0));")
            variables = "        variable a : signed(ACC_W - 1 downto 0);\n"
        else:
            variables = (
                "        variable a : signed(ACC_W - 1 downto 0);\n"
                "        variable p, q, r, half : signed(P_W - 1 downto 0);\n"
            )
            stage += [
                "                    p    := resize(a * MULT(j), P_W);",
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

        body += (
            "    -- Output register"
            + (" + ReLU" if a.get("relu") else "")
            + (" + requantisation" if rq is not None else "") + "\n"
            "    outp : process(clk)\n" + variables +
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                valid_o <= '0';\n"
            "                data_o  <= (others => '0');\n"
            "            else\n"
            "                valid_o <= valid_i;\n"
            "                for j in 0 to OUT_F - 1 loop\n"
            + "\n".join(stage) + "\n"
            "                end loop;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process outp;\n\n"
            "end architecture rtl;\n"
        )
        return body
