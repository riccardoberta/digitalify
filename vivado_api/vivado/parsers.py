"""
Parsers for Vivado text reports.

Each function accepts the raw text of a report file and returns a dict
with the fields defined in models.py.  All values are None if the
relevant line could not be found (e.g. report was incomplete due to an error).

The reports are Vivado's default plain-text format (not XML).
"""

from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# Utilization report  (report_utilization)
# ---------------------------------------------------------------------------
# Vivado table format example:
#
#  +-------------------------+------+-------+------------+-----------+-------+
#  |        Site Type        | Used | Fixed | Prohibited | Available | Util% |
#  +-------------------------+------+-------+------------+-----------+-------+
#  | Slice LUTs              |  123 |     0 |          0 |     20800 |  0.59 |
#  | LUT as Logic            |  123 |     0 |          0 |     20800 |  0.59 |
#  | LUT as Memory           |    0 |     0 |          0 |      9600 |  0.00 |
#  | Slice Registers         |   64 |     0 |          0 |     41600 |  0.15 |
#  |   Register as Flip Flop |   64 |     0 |          0 |     41600 |  0.15 |
#  | F7 Muxes                |    0 |     0 |          0 |     16300 |  0.00 |
#  | F8 Muxes                |    0 |     0 |          0 |      8150 |  0.00 |
#  +-------------------------+------+-------+------------+-----------+-------+
#
#  +----------------+------+-------+------------+-----------+-------+
#  |    Site Type   | Used | Fixed | Prohibited | Available | Util% |
#  +----------------+------+-------+------------+-----------+-------+
#  | DSPs           |    0 |     0 |          0 |        90 |  0.00 |
#  +----------------+------+-------+------------+-----------+-------+


def _table_value(text: str, label: str, col: int = 1) -> str | None:
    """
    Find a row in a Vivado ASCII table by *label* and return column *col*
    (0-indexed from after the first ``|``).

    ``\S*`` after the label allows an optional suffix such as the ``*``
    that Vivado appends to "Slice LUTs*" in synthesis-only reports to
    indicate the count is an estimate (pre-implementation).
    """
    pattern = rf"\|\s*{re.escape(label)}\S*\s*\|([^|]+)\|" + r"([^|]+)\|" * col
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return m.group(col).strip()


def _int_val(text: str, label: str, col: int = 1) -> int | None:
    v = _table_value(text, label, col)
    if v is None:
        return None
    try:
        return int(v.replace(",", "").strip())
    except ValueError:
        return None


def _float_pct(text: str, label: str) -> float | None:
    """Extract the Util% column (last numeric column) for *label*."""
    # Util% is the 5th column after the label
    v = _table_value(text, label, col=5)
    if v is None:
        return None
    try:
        return float(v.strip())
    except ValueError:
        return None


def parse_utilization(report: str) -> dict:
    return {
        "luts":       _int_val(report,  "Slice LUTs"),
        "lut_ram":    _int_val(report,  "LUT as Memory"),
        "flip_flops": _int_val(report,  "Slice Registers"),
        "dsps":       _int_val(report,  "DSPs"),
        "brams":      _int_val(report,  "Block RAM Tile"),
        "ios":        _int_val(report,  "Bonded IOB"),
        "lut_pct":    _float_pct(report,"Slice LUTs"),
        "ff_pct":     _float_pct(report,"Slice Registers"),
        "dsp_pct":    _float_pct(report,"DSPs"),
        "bram_pct":   _float_pct(report,"Block RAM Tile"),
    }


# ---------------------------------------------------------------------------
# Timing summary report  (report_timing_summary)
# ---------------------------------------------------------------------------
# Key lines look like:
#   WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints
#      1.234       0.000                      0                  123
#
#   WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints
#      0.567       0.000                      0                  123

def _timing_row(text: str, header: str) -> list[str] | None:
    """Return the data row immediately following the header row."""
    idx = text.find(header)
    if idx == -1:
        return None
    # Move to the start of the line *after* the header line
    eol = text.find("\n", idx)
    if eol == -1:
        return None
    rest = text[eol + 1:]
    # Skip blank lines and pure-dash separator lines (e.g. "------")
    # Note: data lines with negative values also start with '-', so we
    # distinguish separators by checking that the line contains only '-' and spaces.
    for line in rest.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if all(c in "-+ " for c in stripped):
            continue   # separator line
        return stripped.split()
    return None


def parse_timing(report: str) -> dict:
    setup_row = _timing_row(report, "WNS(ns)")
    hold_row  = _timing_row(report, "WHS(ns)")

    def _f(row, idx):
        if row is None or idx >= len(row):
            return None
        try:
            return float(row[idx])
        except ValueError:
            return None

    wns = _f(setup_row, 0)
    tns = _f(setup_row, 1)
    whs = _f(hold_row,  0)
    ths = _f(hold_row,  1)

    timing_met: bool | None = None
    if wns is not None and tns is not None:
        timing_met = (wns >= 0.0 and tns == 0.0)

    return {
        "wns_ns":     wns,
        "tns_ns":     tns,
        "whs_ns":     whs,
        "ths_ns":     ths,
        "timing_met": timing_met,
    }


# ---------------------------------------------------------------------------
# Power report  (report_power)
# ---------------------------------------------------------------------------
# Key table section:
#
#  +--------------------------+----------------------------------+
#  | Total On-Chip Power (W)  | 0.172                            |
#  | Dynamic (W)              | 0.026                            |
#  | Device Static (W)        | 0.146                            |
#  | Junction Temperature (C) | 30.6                             |
#  | Confidence Level         | Low                              |
#  +--------------------------+----------------------------------+

def _power_field(text: str, label: str) -> str | None:
    pattern = rf"\|\s*{re.escape(label)}\s*\|\s*([^\|]+)\s*\|"
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _power_float(text: str, label: str) -> float | None:
    v = _power_field(text, label)
    if v is None:
        return None
    # Sometimes the value has units like "0.172 W" — strip non-numeric suffix
    numeric = re.match(r"[\d.]+", v)
    if not numeric:
        return None
    try:
        return float(numeric.group())
    except ValueError:
        return None


def parse_power(report: str) -> dict:
    return {
        "total_power_w":   _power_float(report, "Total On-Chip Power (W)"),
        "dynamic_power_w": _power_float(report, "Dynamic (W)"),
        "static_power_w":  _power_float(report, "Device Static (W)"),
        "junction_temp_c": _power_float(report, "Junction Temperature (C)"),
        "confidence":      _power_field(report, "Confidence Level"),
    }
