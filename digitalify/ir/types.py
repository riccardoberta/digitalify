"""
Integer data types carried by IR tensors.

Every activation bus in the generated hardware is a packed vector of
fixed-width integers.  ``IntType`` records the width and signedness of one
element so that each layer can use its own precision (e.g. INT8 input,
UINT4 hidden activations, INT20 accumulator output).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntType:
    """Fixed-width two's-complement (signed) or plain binary (unsigned) integer."""

    bits: int
    signed: bool

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError(f"IntType needs at least 1 bit, got {self.bits}")

    @property
    def min(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    @property
    def name(self) -> str:
        return f"{'INT' if self.signed else 'UINT'}{self.bits}"

    @staticmethod
    def for_range(lo: int, hi: int, signed: bool | None = None) -> "IntType":
        """Smallest type that can hold every integer in ``[lo, hi]``.

        If *signed* is None the type is signed only when ``lo < 0``.
        """
        lo, hi = int(lo), int(hi)
        if lo > hi:
            raise ValueError(f"empty range [{lo}, {hi}]")
        if signed is None:
            signed = lo < 0
        if not signed and lo < 0:
            raise ValueError(f"range [{lo}, {hi}] needs a signed type")
        bits = 1
        while True:
            t = IntType(bits, signed)
            if t.min <= lo and hi <= t.max:
                return t
            bits += 1

    def __str__(self) -> str:
        return self.name
