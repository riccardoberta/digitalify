"""
VHDL handler for activation functions.

relu — element-wise max(0, x), fully combinational (zero latency).
       For a signed bus an element is zeroed when its sign bit is set;
       on an unsigned bus ReLU is the identity.
"""

from __future__ import annotations

from digitalify.ir.node import IRNode
from digitalify.operators.base import VHDLOperator


class ReLUOperator(VHDLOperator):
    """Handler for ``relu`` IRNodes."""

    latency = 0
    clocked = False

    def architecture(self, node: IRNode, entity: str) -> str:
        bits = node.input_dtype.bits
        n = node.in_features
        body = self.header(node, entity) + self.entity(node, entity)
        body += f"architecture rtl of {entity} is\nbegin\n\n"
        if not node.input_dtype.signed:
            body += (
                "    -- Unsigned input: ReLU is the identity\n"
                "    valid_o <= valid_i;\n"
                "    data_o  <= data_i;\n\n"
            )
        else:
            slice_ = self.element("data_i", "i", str(bits))
            out_slice = self.element("data_o", "i", str(bits))
            body += (
                "    -- ReLU: zero every element whose sign bit is set\n"
                "    relu : process(data_i, valid_i)\n"
                "    begin\n"
                "        valid_o <= valid_i;\n"
                f"        for i in 0 to {n - 1} loop\n"
                f"            if data_i((i + 1) * {bits} - 1) = '1' then\n"
                f"                {out_slice} <= (others => '0');\n"
                "            else\n"
                f"                {out_slice} <= {slice_};\n"
                "            end if;\n"
                "        end loop;\n"
                "    end process relu;\n\n"
            )
        return body + "end architecture rtl;\n"
