"""Input validation: everything that reaches a generated TCL script."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from models import CreateJobRequest, SimulationRequest, SynthesisRequest

FID = str(uuid.uuid4())

INJECTIONS = [
    "x[exec calc]",
    "top; exec calc",
    "top\nexec calc",
    "top\n",
    "$env(PATH)",
    "{top}",
    'top"',
    "top entity",
    "",
    "1top",
    "_top",
    "tóp",
]


# --- top_entity / testbench_top ----------------------------------------------

@pytest.mark.parametrize("name", ["top", "Top_1", "tb_top", "a", "A1_b2_C3"])
def test_valid_identifiers(name):
    assert CreateJobRequest(board_id="basys3", top_entity=name, file_ids=[FID]).top_entity == name
    assert SimulationRequest(testbench_file_id=FID, testbench_top=name).testbench_top == name


@pytest.mark.parametrize("name", INJECTIONS)
def test_invalid_identifiers(name):
    with pytest.raises(ValidationError):
        CreateJobRequest(board_id="basys3", top_entity=name, file_ids=[FID])
    with pytest.raises(ValidationError):
        SimulationRequest(testbench_file_id=FID, testbench_top=name)


# --- file IDs ------------------------------------------------------------------

@pytest.mark.parametrize("fid", ["..", "../x", "a\\b", "not-a-uuid", FID + "\n", FID.upper(), ""])
def test_invalid_file_ids(fid):
    with pytest.raises(ValidationError):
        CreateJobRequest(board_id="basys3", top_entity="top", file_ids=[fid])
    with pytest.raises(ValidationError):
        SimulationRequest(testbench_file_id=fid, testbench_top="tb")


def test_file_ids_must_not_be_empty():
    with pytest.raises(ValidationError):
        CreateJobRequest(board_id="basys3", top_entity="top", file_ids=[])


# --- synthesis options ---------------------------------------------------------

@pytest.mark.parametrize("strategy", [
    "default", "RuntimeOptimized", "AreaOptimized_high", "AreaOptimized_medium",
    "AlternateRoutability", "AreaMapLargeShiftRegToBRAM", "FewerCarryChains",
])
def test_valid_strategies(strategy):
    assert SynthesisRequest(strategy=strategy).strategy == strategy


@pytest.mark.parametrize("strategy", ["Default ", "x[exec calc]", "fast", ""])
def test_invalid_strategy(strategy):
    with pytest.raises(ValidationError):
        SynthesisRequest(strategy=strategy)


@pytest.mark.parametrize("flatten", ["rebuilt", "full", "none"])
def test_valid_flatten(flatten):
    assert SynthesisRequest(flatten_hierarchy=flatten).flatten_hierarchy == flatten


@pytest.mark.parametrize("flatten", ["FULL", "rebuilt [exec calc]", ""])
def test_invalid_flatten(flatten):
    with pytest.raises(ValidationError):
        SynthesisRequest(flatten_hierarchy=flatten)


# --- runtime -----------------------------------------------------------------------

@pytest.mark.parametrize("rt", ["1000ns", "1us", "1.5us", "20ms", "1s", "300fs", "7ps"])
def test_valid_runtime(rt):
    assert SimulationRequest(testbench_file_id=FID, testbench_top="tb", runtime=rt).runtime == rt


@pytest.mark.parametrize("rt", [
    "1000", "ns", "1 us", "1.us", "-1ns", "1ns;exec calc", "1ns[exec calc]",
    "1ns\n", "1hr", "all", "",
])
def test_invalid_runtime(rt):
    with pytest.raises(ValidationError):
        SimulationRequest(testbench_file_id=FID, testbench_top="tb", runtime=rt)


# --- through the HTTP layer --------------------------------------------------------

def test_create_job_rejects_tcl_injection(client, uploaded):
    r = client.post("/jobs", json={
        "board_id": "basys3",
        "top_entity": "x[exec calc]",
        "file_ids": [uploaded["top.vhd"]],
    })
    assert r.status_code == 422


def test_synthesize_rejects_bad_strategy(client, uploaded):
    r = client.post("/jobs", json={
        "board_id": "basys3", "top_entity": "top", "file_ids": [uploaded["top.vhd"]],
    })
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]
    r = client.post(f"/jobs/{job_id}/synthesize", json={"strategy": "default [exec calc]"})
    assert r.status_code == 422
    # The job must not have moved out of pending
    assert client.get(f"/jobs/{job_id}").json()["status"] == "pending"


# --- defence in depth in the TCL generators ------------------------------------------

def test_runner_rejects_unsafe_values_even_without_pydantic(client, uploaded):
    import job_store
    from vivado import runner

    state = job_store.create_job("basys3", "top", [uploaded["top.vhd"]])
    jid = state["job_id"]
    fids = [uploaded["top.vhd"]]

    tcl = runner._synthesis_tcl(jid, "xc7a35tcpg236-1", "top", fids, "default", "rebuilt")
    assert "set top       top" in tcl
    assert "read_vhdl -vhdl2008 {" in tcl

    with pytest.raises(ValueError):
        runner._synthesis_tcl(jid, "xc7a35tcpg236-1", "x[exec calc]", fids, "default", "rebuilt")
    with pytest.raises(ValueError):
        runner._synthesis_tcl(jid, "xc7a35tcpg236-1", "top", fids, "default;exec calc", "rebuilt")
    with pytest.raises(ValueError):
        runner._synthesis_tcl(jid, "xc7a35tcpg236-1", "top", fids, "default", "none]")
    with pytest.raises(ValueError):
        runner._simulation_tcl(jid, "xc7a35tcpg236-1", fids, uploaded["tb_top.vhd"], "tb[x]", "1us")
    with pytest.raises(ValueError):
        runner._simulation_tcl(jid, "xc7a35tcpg236-1", fids, uploaded["tb_top.vhd"], "tb_top", "1us;x")


def test_xdc_files_are_read_as_constraints(client):
    import job_store
    from vivado import runner

    vhd = job_store.register_upload("top.vhd", b"--")["file_id"]
    xdc = job_store.register_upload("top.xdc", b"#")["file_id"]
    state = job_store.create_job("basys3", "top", [vhd, xdc])

    syn = runner._synthesis_tcl(state["job_id"], "xc7a35tcpg236-1", "top", [vhd, xdc], "default", "rebuilt")
    assert "read_xdc {" in syn and syn.count("read_vhdl") == 1

    sim = runner._simulation_tcl(state["job_id"], "xc7a35tcpg236-1", [vhd, xdc], vhd, "tb_top", "1us")
    assert "read_xdc" not in sim
