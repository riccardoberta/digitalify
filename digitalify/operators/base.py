"""
Abstract base class for VHDL operator handlers.

Each supported operator type (linear, relu, …) provides a concrete
subclass that knows how to emit synthesisable VHDL for that operation.

A handler must implement three methods:
  - ``component_declaration``  – the reusable ``component`` block used
                                 inside the top-level architecture.
  - ``component_architecture`` – the self-contained entity + architecture
                                 written to its own .vhd file.
  - ``instantiation``          – the ``port map`` statement that wires
                                 one instance into the top-level.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from digitalify.ir.node import IRNode


class VHDLOperator(ABC):
    """
    Abstract handler for one operator type.

    Parameters
    ----------
    data_width : int
        Bit-width of every activation and weight sample (default 8).
    acc_width : int
        Bit-width of the internal accumulator register (default 32).
    """

    def __init__(self, data_width: int = 8, acc_width: int = 32) -> None:
        self.data_width = data_width
        self.acc_width = acc_width

    # ------------------------------------------------------------------
    # Interface that every handler must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def component_declaration(self, node: IRNode) -> str:
        """
        Return the VHDL ``component`` block for use inside another
        architecture's declarative region.
        """

    @abstractmethod
    def component_architecture(self, node: IRNode) -> str:
        """
        Return the complete VHDL source (library clauses + entity +
        architecture) for a standalone ``.vhd`` file.
        """

    @abstractmethod
    def instantiation(
        self,
        node: IRNode,
        signal_in: str,
        signal_out: str,
        signal_valid_in: str,
        signal_valid_out: str,
    ) -> str:
        """
        Return the component instantiation statement (``port map``) that
        connects ``signal_in`` / ``signal_out`` to this component instance
        inside the top-level architecture.
        """

    # ------------------------------------------------------------------
    # Shared VHDL header (library + use clauses)
    # ------------------------------------------------------------------

    @staticmethod
    def _vhdl_header() -> str:
        return (
            "library ieee;\n"
            "use ieee.std_logic_1164.all;\n"
            "use ieee.numeric_std.all;\n\n"
        )
