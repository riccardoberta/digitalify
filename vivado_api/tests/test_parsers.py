"""Vivado report parsers."""

from vivado.parsers import parse_timing

# Layout of report_timing_summary (Vivado 2025.1): setup, hold and pulse-width
# columns share one header line.
TIMING_SUMMARY = """\
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints
    -------      -------  ---------------------  -------------------      -------      -------  ---------------------  -------------------     --------     --------  ----------------------  --------------------
    -14.525     -792.920                     80                  165       -0.276       -9.738                     76                  165        1.500        0.000                       0                    82


Clock             WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)
-----             -------      -------  ---------------------  -------------------      -------      -------
clk                 9.000        0.000                      0                  165        0.100        0.000
"""


def test_setup_and_hold_come_from_their_own_columns():
    t = parse_timing(TIMING_SUMMARY)
    assert t == {
        "wns_ns": -14.525, "tns_ns": -792.92,
        "whs_ns": -0.276, "ths_ns": -9.738,
        "timing_met": False,
    }


def test_met_timing():
    report = TIMING_SUMMARY.replace("    -14.525     -792.920", "      1.250        0.000")
    t = parse_timing(report)
    assert t["wns_ns"] == 1.25 and t["timing_met"] is True


def test_missing_table_gives_none():
    t = parse_timing("No timing information: design has no clocks\n")
    assert t == {"wns_ns": None, "tns_ns": None, "whs_ns": None, "ths_ns": None,
                 "timing_met": None}
