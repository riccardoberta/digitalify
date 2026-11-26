"""Clock-period / out-of-context synthesis options."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import SynthesisRequest


def test_defaults_are_unconstrained():
    req = SynthesisRequest()
    assert req.clock_period_ns is None
    assert req.clock_port == "clk"
    assert req.out_of_context is False


@pytest.mark.parametrize("period", [0, -5, 1001, "fast"])
def test_invalid_period(period):
    with pytest.raises(ValidationError):
        SynthesisRequest(clock_period_ns=period)


@pytest.mark.parametrize("port", ["clk]", "clk; exec calc", "[exec calc]", "", "1clk"])
def test_invalid_clock_port(port):
    with pytest.raises(ValidationError):
        SynthesisRequest(clock_period_ns=10, clock_port=port)


def test_clock_constraints_text():
    from vivado.runner import clock_constraints

    xdc = clock_constraints(4, "sys_clk")
    assert "create_clock -name sys_clk -period 4.000 [get_ports sys_clk]" in xdc
    assert 'NAME != "sys_clk"' in xdc
    assert "set_output_delay -clock sys_clk 0.000" in xdc
    with pytest.raises(ValueError):
        clock_constraints(4, "clk]")
    with pytest.raises(ValueError):
        clock_constraints(0)


def test_synthesis_tcl_reads_clock_xdc_and_ooc(client, uploaded, tmp_path):
    import job_store
    from vivado import runner

    fids = [uploaded["top.vhd"]]
    jid = job_store.create_job("basys3", "top", fids)["job_id"]
    xdc = tmp_path / "clock.xdc"

    tcl = runner._synthesis_tcl(jid, "xc7a35tcpg236-1", "top", fids, "default", "rebuilt",
                                clock_xdc=xdc, out_of_context=True)
    assert "read_xdc {" in tcl and "clock.xdc}" in tcl
    assert "-mode out_of_context" in tcl

    plain = runner._synthesis_tcl(jid, "xc7a35tcpg236-1", "top", fids, "default", "rebuilt")
    assert "read_xdc" not in plain and "out_of_context" not in plain


@pytest.mark.parametrize("period,wns,expected", [
    (10.0, 2.0, 125.0),       # met: fmax above target
    (4.0, -1.0, 200.0),       # violated: fmax below target
    (10.0, None, None),
    (None, 1.0, None),
])
def test_fmax(period, wns, expected):
    from vivado.runner import _fmax

    out = _fmax(period, wns)
    assert out["clock_period_ns"] == period
    assert out["fmax_mhz"] == expected


def test_synthesize_endpoint_rejects_bad_clock(client, uploaded):
    r = client.post("/jobs", json={"board_id": "basys3", "top_entity": "top",
                                   "file_ids": [uploaded["top.vhd"]]})
    job_id = r.json()["job_id"]
    r = client.post(f"/jobs/{job_id}/synthesize", json={"clock_period_ns": -1})
    assert r.status_code == 422
    assert client.get(f"/jobs/{job_id}").json()["status"] == "pending"
