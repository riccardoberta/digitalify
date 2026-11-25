from fractions import Fraction
import math

import numpy as np
import pytest

from digitalify.ir.types import IntType
from digitalify.quant import ROUNDING_MODES, requant_multiplier, shift_round

# Real-valued definitions of the QONNX rounding modes.
_REAL = {
    "FLOOR": math.floor,
    "CEIL": math.ceil,
    "ROUND": round,                                   # Python: half to even
    "HALF_UP": lambda x: int(math.copysign(math.floor(abs(x) + Fraction(1, 2)), x)),
    "HALF_DOWN": lambda x: int(math.copysign(math.ceil(abs(x) - Fraction(1, 2)), x)),
    "UP": lambda x: int(math.copysign(math.ceil(abs(x)), x)),
    "DOWN": lambda x: int(math.copysign(math.floor(abs(x)), x)),
}


@pytest.mark.parametrize("mode", ROUNDING_MODES)
def test_shift_round_matches_real_rounding(mode):
    for n in range(0, 6):
        for p in range(-200, 201):
            expected = _REAL[mode](Fraction(p, 1 << n))
            assert shift_round(p, n, mode) == expected, (p, n)


def test_requant_multiplier_exact_for_dyadic():
    m, n, exact = requant_multiplier(Fraction(3, 64))
    assert (m, n, exact) == (3, 6, True)


def test_requant_multiplier_approximation_precision():
    M = Fraction(1, 3)
    m, n, exact = requant_multiplier(M, max_bits=16)
    assert not exact
    assert m < 1 << 16
    assert abs(Fraction(m, 1 << n) - M) / M < Fraction(1, 1 << 15)


@pytest.mark.parametrize("lo,hi,expected", [
    (0, 255, IntType(8, False)),
    (-128, 127, IntType(8, True)),
    (-1, 0, IntType(1, True)),
    (0, 256, IntType(9, False)),
    (-129, 5, IntType(9, True)),
])
def test_inttype_for_range(lo, hi, expected):
    assert IntType.for_range(lo, hi) == expected


def test_inttype_bounds():
    assert (IntType(4, True).min, IntType(4, True).max) == (-8, 7)
    assert (IntType(4, False).min, IntType(4, False).max) == (0, 15)
