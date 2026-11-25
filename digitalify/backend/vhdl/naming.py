"""VHDL identifier helpers."""

from __future__ import annotations

import re

# VHDL-2008 reserved words (IEEE 1076-2008, section 15.10).
VHDL_RESERVED = frozenset("""
abs access after alias all and architecture array assert assume assume_guarantee
attribute begin block body buffer bus case component configuration constant context
cover default disconnect downto else elsif end entity exit fairness file for force
function generate generic group guarded if impure in inertial inout is label library
linkage literal loop map mod nand new next nor not null of on open or others out
package parameter port postponed procedure process property protected pure range
record register reject release rem report restrict restrict_guarantee return rol ror
select sequence severity shared signal sla sll sra srl strong subtype then to
transport type unaffected units until use variable vmode vprop vunit wait when while
with xnor xor
""".split())


def vhdl_identifier(name: str, fallback: str = "network") -> str:
    """Turn an arbitrary string into a valid, lower-case VHDL basic identifier.

    Rules: letters, digits and single underscores only; must start with a
    letter; must not end with an underscore; must not be a reserved word.
    """
    s = re.sub(r"[^A-Za-z0-9]+", "_", name or "").strip("_").lower()
    s = re.sub(r"_+", "_", s)
    if not s:
        s = fallback
    if not s[0].isalpha():
        s = f"n_{s}"
    if s in VHDL_RESERVED:
        s = f"{s}_net"
    return s
