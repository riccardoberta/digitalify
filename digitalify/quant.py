"""
Integer quantisation arithmetic shared by the frontend, the reference model
and the VHDL backend.

Requantisation
--------------
A layer whose output is re-quantised by a QONNX ``Quant`` node computes

    q_out = clip( R( acc * M ) )        with  M = s_in * s_w / s_out

where ``R`` is the Quant node's rounding mode.  In hardware the real factor
``M`` is replaced by an integer multiplier and a right shift,
``M ≈ m / 2**n``, so that

    q_out = clip( shift_round(acc * m, n, R) )

Every float32 scale is a dyadic rational, so for power-of-two scales (the
common case for Brevitas / QKeras / hls4ml models) ``m / 2**n`` is *exactly*
``M`` and the hardware is bit-exact with the QONNX model.  For arbitrary
scales the multiplier is rounded to ``max_bits`` bits of precision.

Note that QONNX clamps before rounding, while the hardware rounds before
clamping; the two are equivalent because the clip bounds are integers and
every rounding mode is monotonic.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

# QONNX rounding modes (see qonnx.custom_op.general.intquant.resolve_rounding_mode).
# "HALF_EVEN" is an alias of "ROUND".
ROUNDING_MODES = ("ROUND", "CEIL", "FLOOR", "UP", "DOWN", "HALF_UP", "HALF_DOWN")


def normalize_rounding(mode: str) -> str:
    m = mode.upper()
    if m == "HALF_EVEN":
        m = "ROUND"
    if m not in ROUNDING_MODES:
        raise ValueError(f"Unknown rounding mode '{mode}'")
    return m


def shift_round(p: int, n: int, mode: str) -> int:
    """Return ``R(p / 2**n)`` for integer *p* using exact integer arithmetic.

    This is the bit-exact model of the rounding logic emitted in VHDL.
    """
    p, n = int(p), int(n)
    if n == 0:
        return p
    q = p >> n                      # floor division (Python >> is arithmetic)
    r = p - (q << n)                # remainder in [0, 2**n)
    half = 1 << (n - 1)
    if mode == "FLOOR":
        return q
    if mode == "CEIL":
        return q + (r != 0)
    if mode == "ROUND":             # half to even
        if r > half or (r == half and (q & 1)):
            return q + 1
        return q
    if mode == "HALF_UP":           # half away from zero
        if r > half or (r == half and p >= 0):
            return q + 1
        return q
    if mode == "HALF_DOWN":         # half towards zero
        if r > half or (r == half and p < 0):
            return q + 1
        return q
    if mode == "UP":                # away from zero
        return q + (r != 0 and p > 0)
    if mode == "DOWN":              # towards zero (truncate)
        return q + (r != 0 and p < 0)
    raise ValueError(f"Unknown rounding mode '{mode}'")


def requant_multiplier(M: Fraction, max_bits: int = 32) -> tuple[int, int, bool]:
    """Represent the positive rational *M* as ``m / 2**n``.

    Returns ``(m, n, exact)`` with ``0 < m < 2**max_bits`` and ``n >= 0``.
    ``exact`` is True when ``m / 2**n == M`` exactly.
    """
    if M <= 0:
        raise ValueError(f"requantisation factor must be positive, got {M}")
    num, den = M.numerator, M.denominator
    if den & (den - 1) == 0 and num < (1 << max_bits):
        return num, den.bit_length() - 1, True

    # Approximate: put the most significant bit of m at position max_bits-1.
    e = math.floor(math.log2(M))
    n = max(0, max_bits - 1 - e)
    m = round(M * (1 << n))
    if m >= (1 << max_bits):        # rounding overflowed into one more bit
        n = max(0, n - 1)
        m = round(M * (1 << n))
    if m == 0:
        m = 1
    return int(m), int(n), Fraction(m, 1 << n) == M


def to_fraction(x: float) -> Fraction:
    """Exact rational value of a float32 scale."""
    return Fraction(float(np.float32(x)))
