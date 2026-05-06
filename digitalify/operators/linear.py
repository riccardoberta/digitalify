"""
VHDL handler for the ``linear`` and ``quantized_linear`` operators.

The generated component implements a fully-combinational multiply-accumulate
(MAC) array followed by a single registered output stage.

Architecture overview
---------------------
* Weights and biases are embedded as VHDL constants — synthesisers will
  infer them as look-up tables or block-RAM depending on size.
* The accumulator is ``ACC_WIDTH`` bits wide (default 32) to avoid overflow
  for typical layer sizes (up to ~16 M input elements with INT8 inputs/weights).
* Requantisation: the accumulated sum is right-shifted by ``SHIFT`` bits and
  saturated-clipped to ``DATA_WIDTH`` bits before being stored in the output
  register.  The shift amount is derived from the combined scale factor
  ``output_scale / (input_scale × weight_scale)``.
* Latency: 1 clock cycle (accumulation is combinational; output is registered).
"""

from __future__ import annotations

import numpy as np

from digitalify.operators.base import VHDLOperator
from digitalify.ir.node import IRNode


def _int8_literal(val: float | int) -> str:
    """Clamp *val* to [-128, 127] and emit as a VHDL integer literal."""
    clamped = int(max(-128, min(127, round(float(val)))))
    return str(clamped)


def _int32_literal(val: float | int) -> str:
    clamped = int(max(-(2**31), min(2**31 - 1, round(float(val)))))
    return str(clamped)


class LinearOperator(VHDLOperator):
    """Handler for ``linear`` / ``quantized_linear`` IRNodes."""

    # ------------------------------------------------------------------
    # Helpers: weight / bias constant arrays
    # ------------------------------------------------------------------

    def _weight_constant(self, node: IRNode) -> str:
        """Emit the VHDL weight constant (2-D array of signed integers)."""
        dw = self.data_width
        weight: np.ndarray = node.attrs["weight"]  # shape (out, in)
        out_f, in_f = weight.shape

        lines = [
            f"    type weight_t is array(0 to {out_f - 1}, 0 to {in_f - 1})"
            f" of signed({dw - 1} downto 0);",
            "    constant WEIGHTS : weight_t := (",
        ]
        for j in range(out_f):
            row_vals = ", ".join(
                f"to_signed({_int8_literal(weight[j, i])}, {dw})"
                for i in range(in_f)
            )
            comma = "," if j < out_f - 1 else ""
            lines.append(f"        ({row_vals}){comma}")
        lines.append("    );")
        return "\n".join(lines)

    def _bias_constant(self, node: IRNode) -> str:
        """Emit the VHDL bias constant (1-D array of signed 32-bit integers)."""
        aw = self.acc_width
        bias = node.attrs.get("bias")
        out_f = node.out_features

        if bias is None:
            # Zero bias represented as a constant zero array
            lines = [
                f"    type bias_t is array(0 to {out_f - 1}) of signed({aw - 1} downto 0);",
                "    constant BIASES : bias_t := (others => to_signed(0, "
                f"{aw}));",
            ]
            return "\n".join(lines)

        lines = [
            f"    type bias_t is array(0 to {out_f - 1}) of signed({aw - 1} downto 0);",
            "    constant BIASES : bias_t := (",
        ]
        for j in range(out_f):
            val = _int32_literal(bias[j])
            comma = "," if j < out_f - 1 else ""
            lines.append(f"        to_signed({val}, {aw}){comma}")
        lines.append("    );")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Generic declarations (shift, acc array type)
    # ------------------------------------------------------------------

    def _generic_declarations(self, node: IRNode) -> str:
        aw = self.acc_width
        dw = self.data_width
        shift: int = node.attrs.get("shift", 0)
        out_f = node.out_features

        return (
            f"    constant SHIFT : integer := {shift};\n"
            f"    type acc_t is array(0 to {out_f - 1}) of signed({aw - 1} downto 0);\n"
            f"    signal acc : acc_t;"
        )

    # ------------------------------------------------------------------
    # Port definitions (reused in entity and component)
    # ------------------------------------------------------------------

    def _port_map(self, node: IRNode) -> str:
        dw = self.data_width
        in_f = node.in_features
        out_f = node.out_features
        return (
            "    port (\n"
            "        clk       : in  std_logic;\n"
            "        rst       : in  std_logic;\n"
            "        valid_i   : in  std_logic;\n"
            f"        data_i    : in  std_logic_vector({in_f * dw - 1} downto 0);\n"
            "        valid_o   : out std_logic;\n"
            f"        data_o    : out std_logic_vector({out_f * dw - 1} downto 0)\n"
            "    );"
        )

    # ------------------------------------------------------------------
    # component_declaration
    # ------------------------------------------------------------------

    def component_declaration(self, node: IRNode) -> str:
        lines = [
            f"    component {node.name} is",
            self._port_map(node),
            f"    end component {node.name};",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # component_architecture
    # ------------------------------------------------------------------

    def component_architecture(self, node: IRNode) -> str:
        dw = self.data_width
        aw = self.acc_width
        in_f = node.in_features
        out_f = node.out_features

        body = self._vhdl_header()
        body += f"entity {node.name} is\n"
        body += self._port_map(node) + "\n"
        body += f"end entity {node.name};\n\n"
        body += f"architecture rtl of {node.name} is\n\n"
        body += self._weight_constant(node) + "\n\n"
        body += self._bias_constant(node) + "\n\n"
        body += self._generic_declarations(node) + "\n\n"
        body += "begin\n\n"

        # ---- Combinational MAC ----
        body += (
            "    -- Combinational multiply-accumulate\n"
            "    process(data_i)\n"
            f"        variable x        : signed({dw - 1} downto 0);\n"
            f"        variable temp_acc : signed({aw - 1} downto 0);\n"
            "    begin\n"
            f"        for j in 0 to {out_f - 1} loop\n"
            "            temp_acc := BIASES(j);\n"
            f"            for i in 0 to {in_f - 1} loop\n"
            f"                x := signed(data_i((i + 1) * {dw} - 1 downto i * {dw}));\n"
            f"                temp_acc := temp_acc + resize(x * WEIGHTS(j, i), {aw});\n"
            "            end loop;\n"
            "            acc(j) <= temp_acc;\n"
            "        end loop;\n"
            "    end process;\n\n"
        )

        # ---- Registered output with requantisation ----
        body += (
            "    -- Output register + requantisation (right-shift + saturate)\n"
            "    process(clk)\n"
            f"        variable shifted : signed({aw - 1} downto 0);\n"
            f"        variable clamped : signed({dw - 1} downto 0);\n"
            f"        constant MAX_VAL : signed({dw - 1} downto 0) := to_signed({2**(dw-1) - 1}, {dw});\n"
            f"        constant MIN_VAL : signed({dw - 1} downto 0) := to_signed({-(2**(dw-1))}, {dw});\n"
            "    begin\n"
            "        if rising_edge(clk) then\n"
            "            if rst = '1' then\n"
            "                valid_o <= '0';\n"
            "                data_o  <= (others => '0');\n"
            "            else\n"
            "                valid_o <= valid_i;\n"
            f"                for j in 0 to {out_f - 1} loop\n"
            "                    shifted := shift_right(acc(j), SHIFT);\n"
            f"                    -- Saturate to {dw}-bit range\n"
            f"                    if shifted > resize(MAX_VAL, {aw}) then\n"
            f"                        clamped := MAX_VAL;\n"
            f"                    elsif shifted < resize(MIN_VAL, {aw}) then\n"
            f"                        clamped := MIN_VAL;\n"
            "                    else\n"
            f"                        clamped := resize(shifted, {dw});\n"
            "                    end if;\n"
            f"                    data_o((j + 1) * {dw} - 1 downto j * {dw}) <= std_logic_vector(clamped);\n"
            "                end loop;\n"
            "            end if;\n"
            "        end if;\n"
            "    end process;\n\n"
        )

        body += f"end architecture rtl;\n"
        return body

    # ------------------------------------------------------------------
    # instantiation
    # ------------------------------------------------------------------

    def instantiation(
        self,
        node: IRNode,
        signal_in: str,
        signal_out: str,
        signal_valid_in: str,
        signal_valid_out: str,
    ) -> str:
        return (
            f"    {node.name}_inst : {node.name}\n"
            "        port map (\n"
            "            clk     => clk,\n"
            "            rst     => rst,\n"
            f"            valid_i => {signal_valid_in},\n"
            f"            data_i  => {signal_in},\n"
            f"            valid_o => {signal_valid_out},\n"
            f"            data_o  => {signal_out}\n"
            "        );"
        )
