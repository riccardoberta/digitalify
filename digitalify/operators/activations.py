"""
VHDL handler for activation functions.

Currently supported
-------------------
relu   — element-wise max(0, x), fully combinational (zero latency).
         Each element is treated as a signed integer; if the MSB (sign bit)
         is 1 the output is forced to zero, otherwise it passes through.
"""

from __future__ import annotations

from digitalify.operators.base import VHDLOperator
from digitalify.ir.node import IRNode


class ReLUOperator(VHDLOperator):
    """Handler for ``relu`` IRNodes."""

    # ------------------------------------------------------------------
    # Port definitions
    # ------------------------------------------------------------------

    def _port_map(self, node: IRNode) -> str:
        dw = self.data_width
        features = node.in_features
        return (
            "    port (\n"
            "        valid_i : in  std_logic;\n"
            f"        data_i  : in  std_logic_vector({features * dw - 1} downto 0);\n"
            "        valid_o : out std_logic;\n"
            f"        data_o  : out std_logic_vector({features * dw - 1} downto 0)\n"
            "    );"
        )

    # ------------------------------------------------------------------
    # component_declaration
    # ------------------------------------------------------------------

    def component_declaration(self, node: IRNode) -> str:
        return "\n".join([
            f"    component {node.name} is",
            self._port_map(node),
            f"    end component {node.name};",
        ])

    # ------------------------------------------------------------------
    # component_architecture
    # ------------------------------------------------------------------

    def component_architecture(self, node: IRNode) -> str:
        dw = self.data_width
        features = node.in_features

        body = self._vhdl_header()
        body += f"entity {node.name} is\n"
        body += self._port_map(node) + "\n"
        body += f"end entity {node.name};\n\n"
        body += f"architecture rtl of {node.name} is\n\n"
        body += "begin\n\n"

        body += (
            "    -- ReLU: pass through if positive, clamp to zero otherwise\n"
            "    process(data_i, valid_i)\n"
            f"        variable x : signed({dw - 1} downto 0);\n"
            "    begin\n"
            "        valid_o <= valid_i;\n"
            f"        for i in 0 to {features - 1} loop\n"
            f"            x := signed(data_i((i + 1) * {dw} - 1 downto i * {dw}));\n"
            "            if x > to_signed(0, x'length) then\n"
            f"                data_o((i + 1) * {dw} - 1 downto i * {dw}) <= std_logic_vector(x);\n"
            "            else\n"
            f"                data_o((i + 1) * {dw} - 1 downto i * {dw}) <= (others => '0');\n"
            "            end if;\n"
            "        end loop;\n"
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
            f"            valid_i => {signal_valid_in},\n"
            f"            data_i  => {signal_in},\n"
            f"            valid_o => {signal_valid_out},\n"
            f"            data_o  => {signal_out}\n"
            "        );"
        )
