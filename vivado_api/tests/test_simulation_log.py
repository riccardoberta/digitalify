"""Detection of testbench failures in the simulation log."""

from __future__ import annotations

import pytest

import job_store
from vivado import runner
from vivado.runner import check_simulation_log

# Excerpt of a real XSim run (see example/vivado_api_test.ipynb)
PASSING_LOG = """\
INFO: [USF-XSim-98] *** Running xsim
## run 1000ns
Note: Simulation completed successfully.
Time: 236 ns  Iteration: 0  Process: /tb_top/stim  File: C:/vivado_api/uploads/x/tb_top.vhd
INFO: [USF-XSim-96] XSim completed. Design snapshot 'tb_top_behav' loaded.
INFO: Simulation complete.
"""

ERROR_LOG = """\
## run 1000ns
Error: Unexpected LED value at cycle 3
Time: 76 ns  Iteration: 0  Process: /tb_top/stim  File: tb_top.vhd
Error: Unexpected LED value at cycle 4
Time: 86 ns  Iteration: 0  Process: /tb_top/stim  File: tb_top.vhd
Note: Simulation completed successfully.
"""

FAILURE_LOG = """\
## run 1000ns
Failure: TB: TESTS FAILED
Time: 500 ns  Iteration: 0  Process: /tb/stim  File: tb.vhd
$finish called at time : 500 ns
"""


def test_passing_log():
    assert check_simulation_log(PASSING_LOG) is None


def test_warning_and_vivado_info_are_not_failures():
    log = PASSING_LOG + "Warning: something odd\nINFO: [Common 17-206] Exiting Vivado\n"
    assert check_simulation_log(log) is None


def test_severity_error_fails():
    err = check_simulation_log(ERROR_LOG)
    assert err is not None
    assert "2 assertion" in err
    assert "Unexpected LED value at cycle 3" in err


def test_severity_failure_fails():
    err = check_simulation_log(FAILURE_LOG)
    assert err is not None and "TB: TESTS FAILED" in err


def test_tb_summary_with_failures():
    log = "Note: TB SUMMARY: 7 passed, 3 failed out of 10 tests\n"
    err = check_simulation_log(log)
    assert err is not None and "7 passed, 3 failed" in err


def test_tb_summary_all_passed():
    log = "Note: TB SUMMARY: 10 passed, 0 failed out of 10 tests\nNote: TB: ALL TESTS PASSED\n"
    assert check_simulation_log(log) is None


def test_mid_line_error_word_is_not_an_assertion():
    # e.g. a TCL echo or a signal name mentioning "Error:" not at line start
    log = '# puts "no Error: here"\nINFO: checked Error: count = 0\n'
    assert check_simulation_log(log) is None


# --- end to end through run_simulation and the API -------------------------------

@pytest.fixture
def synthesized_job(client, uploaded):
    r = client.post("/jobs", json={
        "board_id": "basys3", "top_entity": "top", "file_ids": [uploaded["top.vhd"]],
    })
    jid = r.json()["job_id"]
    job_store.update_job(jid, status="synthesized")
    return jid


@pytest.mark.parametrize("log, expected_status", [
    (PASSING_LOG, "done"),
    (ERROR_LOG, "failed"),
    (FAILURE_LOG, "failed"),
])
def test_simulation_status_reflects_log(client, uploaded, synthesized_job, monkeypatch,
                                        log, expected_status):
    captured = {}

    def fake_run_vivado(tcl, log_file, timeout):
        captured["tcl"] = tcl
        return 0, log   # Vivado itself exits 0 in all three cases

    monkeypatch.setattr(runner, "_run_vivado", fake_run_vivado)

    r = client.post(f"/jobs/{synthesized_job}/simulate", json={
        "testbench_file_id": uploaded["tb_top.vhd"],
        "testbench_top": "tb_top",
        "runtime": "1us",
    })
    assert r.status_code == 200, r.text
    # TestClient runs background tasks before returning
    assert client.get(f"/jobs/{synthesized_job}").json()["status"] == expected_status

    sim = client.get(f"/jobs/{synthesized_job}/simulation").json()
    if expected_status == "failed":
        assert sim["error"].startswith("Testbench failed")
    else:
        assert sim["error"] is None
    assert "set tb_top     tb_top" in captured["tcl"]
    assert "set sim_time   1us" in captured["tcl"]


def test_simulate_requires_synthesized_state(client, uploaded):
    r = client.post("/jobs", json={
        "board_id": "basys3", "top_entity": "top", "file_ids": [uploaded["top.vhd"]],
    })
    jid = r.json()["job_id"]
    r = client.post(f"/jobs/{jid}/simulate", json={
        "testbench_file_id": uploaded["tb_top.vhd"], "testbench_top": "tb_top",
    })
    assert r.status_code == 409
