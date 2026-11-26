"""
Vivado subprocess wrapper and TCL script generation.

All Vivado operations run in batch mode:
    vivado -mode batch -source <script.tcl> -log <run.log> -journal <run.jou>

Both ``run_synthesis`` and ``run_simulation`` are blocking functions intended
to be called inside a thread-pool executor so that FastAPI remains responsive:

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_synthesis, job_id, opts)

Every value interpolated into the generated TCL is validated twice: by the
pydantic models at the API boundary and again here (``_require``), so a
caller that bypasses the HTTP layer cannot inject TCL either.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import get_args

from config import settings, BOARD_CATALOG
from job_store import job_dir, resolve_upload_path, get_job
from models import (
    VHDL_IDENTIFIER_PATTERN, RUNTIME_PATTERN, SynthStrategy, FlattenHierarchy,
)
from vivado.parsers import parse_utilization, parse_timing, parse_power

_LOG_TAIL_CHARS = 4000

_VALID_STRATEGIES = frozenset(get_args(SynthStrategy))
_VALID_FLATTEN = frozenset(get_args(FlattenHierarchy))
# TCL-safe characters for a Xilinx part name (e.g. xcku040-ffva1156-2-e).
_PART_PATTERN = r"^[A-Za-z0-9_\-]+$"


def _require(value: str, pattern: str, what: str) -> str:
    """Raise ValueError unless *value* fully matches *pattern*."""
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid {what}: {value!r}")
    return value


def _require_in(value: str, allowed: frozenset[str], what: str) -> str:
    if value not in allowed:
        raise ValueError(f"Invalid {what}: {value!r}")
    return value


def _tcl_brace(s: str) -> str:
    """
    Quote *s* as a TCL braced word (no command/variable substitution).
    Raises ValueError if *s* contains characters that could break out of it.
    """
    if any(c in s for c in "{}\\\n\r"):
        raise ValueError(f"Unsafe characters for TCL: {s!r}")
    return "{" + s + "}"


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def _tcl_path(p: Path) -> str:
    """
    Convert *p* to a forward-slash string suitable for Vivado's TCL interpreter.
    Vivado requires forward slashes on all platforms, including Windows.
    """
    return p.as_posix()


def _vivado_cmd(script_path: Path, log_file: Path, journal_path: Path) -> list[str]:
    """
    Build the Vivado command list for the current platform.

    On Windows, vivado.bat must be invoked through the shell so that the
    .bat extension is resolved by cmd.exe.  On Linux/macOS the binary is
    invoked directly.
    """
    return [
        settings.vivado_path,
        "-mode",    "batch",
        "-source",  _tcl_path(script_path),
        "-log",     _tcl_path(log_file),
        "-journal", _tcl_path(journal_path),
    ]


# ---------------------------------------------------------------------------
# TCL script builders
# ---------------------------------------------------------------------------

def _read_source_lines(file_ids: list[str], include_xdc: bool = True) -> str:
    """
    Return TCL commands that read every uploaded file: ``read_vhdl`` for
    VHDL sources, ``read_xdc`` for constraints (skipped if *include_xdc* is
    False, e.g. for behavioral simulation).
    """
    lines = []
    for fid in file_ids:
        path = resolve_upload_path(fid)
        quoted = _tcl_brace(_tcl_path(path))
        if path.suffix.lower() == ".xdc":
            if include_xdc:
                lines.append(f"read_xdc {quoted}")
        else:
            lines.append(f"read_vhdl -vhdl2008 {quoted}")
    return "\n".join(lines)


def clock_constraints(clock_period_ns: float, clock_port: str = "clk") -> str:
    """
    XDC text that clocks *clock_port* at *clock_period_ns* and gives every
    other port a 0 ns I/O delay, so input→register and register→output paths
    are timed against the full period as well.
    """
    port = _require(clock_port, VHDL_IDENTIFIER_PATTERN, "clock port")
    period = float(clock_period_ns)
    if not 0 < period <= 1000:
        raise ValueError(f"Invalid clock period: {clock_period_ns!r}")
    return (
        f"create_clock -name {port} -period {period:.3f} [get_ports {port}]\n"
        f"set_input_delay  -clock {port} 0.000 "
        f'[get_ports -filter {{DIRECTION == IN && NAME != "{port}"}}]\n'
        f"set_output_delay -clock {port} 0.000 [get_ports -filter {{DIRECTION == OUT}}]\n"
    )


def _synthesis_tcl(
    job_id: str,
    part: str,
    top_entity: str,
    file_ids: list[str],
    strategy: str,
    flatten_hierarchy: str,
    clock_xdc: Path | None = None,
    out_of_context: bool = False,
) -> str:
    part = _require(part, _PART_PATTERN, "part")
    top_entity = _require(top_entity, VHDL_IDENTIFIER_PATTERN, "top entity")
    strategy = _require_in(strategy, _VALID_STRATEGIES, "strategy")
    flatten_hierarchy = _require_in(
        flatten_hierarchy, _VALID_FLATTEN, "flatten_hierarchy"
    )
    work = job_dir(job_id)
    work_tcl = _tcl_brace(_tcl_path(work))
    read_cmds = _read_source_lines(file_ids)
    if clock_xdc is not None:
        read_cmds += f"\nread_xdc {_tcl_brace(_tcl_path(clock_xdc))}"
    mode = " \\\n    -mode out_of_context" if out_of_context else ""

    return f"""\
# Vivado batch synthesis — generated by digitalify vivado_api
# job_id: {job_id}

set work_dir  {work_tcl}
set part      {part}
set top       {top_entity}

# ── Project ──────────────────────────────────────────────────────────────────
create_project -in_memory -part $part

# ── Sources ──────────────────────────────────────────────────────────────────
{read_cmds}
set_property top $top [current_fileset]
update_compile_order -fileset sources_1

# ── Synthesis ─────────────────────────────────────────────────────────────────
puts "INFO: Starting synthesis (strategy={strategy}) ..."
synth_design \\
    -top $top \\
    -part $part \\
    -flatten_hierarchy {flatten_hierarchy} \\
    -directive {strategy}{mode}
puts "INFO: Synthesis finished."

# ── Reports ──────────────────────────────────────────────────────────────────
puts "INFO: Writing reports ..."
report_utilization    -file "$work_dir/utilization.rpt"
report_timing_summary -file "$work_dir/timing.rpt"     -quiet
report_power          -file "$work_dir/power.rpt"       -quiet

# ── Checkpoint ───────────────────────────────────────────────────────────────
write_checkpoint -force "$work_dir/synth.dcp"
puts "INFO: All done."
"""


def _simulation_tcl(
    job_id: str,
    part: str,
    design_file_ids: list[str],
    testbench_file_id: str,
    testbench_top: str,
    runtime: str,
) -> str:
    part = _require(part, _PART_PATTERN, "part")
    testbench_top = _require(testbench_top, VHDL_IDENTIFIER_PATTERN, "testbench top")
    runtime = _require(runtime, RUNTIME_PATTERN, "runtime")
    work = job_dir(job_id)
    work_tcl = _tcl_brace(_tcl_path(work))
    read_design = _read_source_lines(design_file_ids, include_xdc=False)
    tb_path = _tcl_brace(_tcl_path(resolve_upload_path(testbench_file_id)))

    return f"""\
# Vivado batch simulation — generated by digitalify vivado_api
# job_id: {job_id}

set work_dir   {work_tcl}
set part       {part}
set tb_top     {testbench_top}
set sim_time   {runtime}

# ── Project ──────────────────────────────────────────────────────────────────
# launch_simulation requires a real on-disk project (not -in_memory).
# We create it inside the job work directory; -force overwrites on re-runs.
set proj_dir "$work_dir/sim_project"
create_project sim_project $proj_dir -part $part -force

# ── Design sources ────────────────────────────────────────────────────────────
{read_design}

# ── Testbench ────────────────────────────────────────────────────────────────
read_vhdl -vhdl2008 {tb_path}

# ── Simulation fileset ────────────────────────────────────────────────────────
set_property top      $tb_top      [get_filesets sim_1]
set_property top_lib  xil_defaultlib [get_filesets sim_1]
update_compile_order  -fileset sim_1

# ── Run ───────────────────────────────────────────────────────────────────────
puts "INFO: Launching behavioral simulation ..."
launch_simulation -simset sim_1 -mode behavioral

puts "INFO: Running for $sim_time ..."
run $sim_time

close_sim
puts "INFO: Simulation complete."
"""


# ---------------------------------------------------------------------------
# Vivado subprocess runner
# ---------------------------------------------------------------------------

def _run_vivado(tcl_script: str, log_file: Path, timeout: int) -> tuple[int, str]:
    """
    Write *tcl_script* to disk and invoke Vivado in batch mode.

    On Windows, vivado.bat must be run through the shell (shell=True) so
    that cmd.exe resolves the .bat extension.  On Linux/macOS the binary
    is called directly (shell=False).

    Returns (returncode, full log text).
    """
    work_dir   = log_file.parent
    script_path  = work_dir / "run.tcl"
    journal_path = work_dir / "run.jou"

    # TCL script must be UTF-8; use LF line endings (Vivado's TCL parser is
    # fine with LF on Windows too, and avoids CRLF surprises).
    script_path.write_text(tcl_script, encoding="utf-8", newline="\n")

    cmd = _vivado_cmd(script_path, log_file, journal_path)

    # On Windows, .bat files require shell=True to be resolved by cmd.exe.
    use_shell = sys.platform == "win32"

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(work_dir),
        shell=use_shell,
    )

    # Vivado writes its own log file; prefer that over stdout/stderr.
    if log_file.exists():
        log_text = log_file.read_text(encoding="utf-8", errors="replace")
    else:
        log_text = (proc.stdout or "") + (proc.stderr or "")

    return proc.returncode, log_text


def _tail(text: str, n: int = _LOG_TAIL_CHARS) -> str:
    return text[-n:] if len(text) > n else text


# ---------------------------------------------------------------------------
# Simulation log analysis
# ---------------------------------------------------------------------------
# XSim prints VHDL assertion/report messages as "<Severity>: <message>",
# e.g. "Note: Simulation completed successfully." or
# "Error: Unexpected LED value at cycle 3".  Vivado's own tool messages use
# upper-case ("ERROR: [Common 17-...]"), so matching is case-sensitive.
_ASSERTION_FAIL_RE = re.compile(r"^\s*(Error|Failure|Fatal):\s*(.*)$", re.MULTILINE)
# Summary line printed by digitalify-generated testbenches.
_TB_SUMMARY_RE = re.compile(r"TB SUMMARY:\s*(\d+)\s+passed,\s*(\d+)\s+failed")


def check_simulation_log(log_text: str) -> str | None:
    """
    Inspect an XSim/Vivado simulation log for testbench failures.

    Returns ``None`` if the testbench passed, otherwise a short error message.
    A run fails if any VHDL assertion with severity ``error`` or ``failure``
    fired, or if a digitalify ``TB SUMMARY`` line reports failed tests.
    """
    problems: list[str] = []

    hits = _ASSERTION_FAIL_RE.findall(log_text)
    if hits:
        severity, msg = hits[0]
        problems.append(
            f"{len(hits)} assertion(s) with severity error/failure "
            f"(first: {severity}: {msg.strip()})"
        )

    summaries = _TB_SUMMARY_RE.findall(log_text)
    if summaries:
        passed, failed = (int(x) for x in summaries[-1])
        if failed > 0:
            problems.append(f"TB SUMMARY: {passed} passed, {failed} failed")

    if not problems:
        return None
    return "Testbench failed: " + "; ".join(problems)


# ---------------------------------------------------------------------------
# Public: synthesis
# ---------------------------------------------------------------------------

def run_synthesis(
    job_id: str,
    strategy: str = "default",
    flatten_hierarchy: str = "rebuilt",
    clock_period_ns: float | None = None,
    clock_port: str = "clk",
    out_of_context: bool = False,
) -> dict:
    """
    Blocking function — call from a thread-pool executor.

    Runs Vivado synthesis, writes reports, returns a result dict with keys:
    elapsed_sec, utilization, timing, power, log_tail, error.
    """
    state = get_job(job_id)
    board  = BOARD_CATALOG[state["board_id"]]
    part   = board["part"]
    top    = state["top_entity"]
    fids   = state["file_ids"]
    work   = job_dir(job_id)

    clock_xdc = None
    if clock_period_ns is not None:
        clock_xdc = work / "clock.xdc"
        clock_xdc.write_text(clock_constraints(clock_period_ns, clock_port),
                             encoding="utf-8", newline="\n")

    tcl = _synthesis_tcl(job_id, part, top, fids, strategy, flatten_hierarchy,
                         clock_xdc=clock_xdc, out_of_context=out_of_context)
    log_file = work / "synthesis.log"

    t0 = time.monotonic()
    try:
        rc, log_text = _run_vivado(tcl, log_file, settings.synthesis_timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return {
            "elapsed_sec": elapsed,
            "error": f"Vivado timed out after {settings.synthesis_timeout}s",
            "log_tail": "",
        }
    elapsed = time.monotonic() - t0
    tail = _tail(log_text)

    if rc != 0:
        return {
            "elapsed_sec": elapsed,
            "error": f"Vivado exited with code {rc}",
            "log_tail": tail,
        }

    # Parse reports
    util  = _load_and_parse(work / "utilization.rpt", parse_utilization)
    timing = _load_and_parse(work / "timing.rpt",     parse_timing)
    power  = _load_and_parse(work / "power.rpt",      parse_power)
    if timing is not None:
        timing.update(_fmax(clock_period_ns, timing.get("wns_ns")))

    return {
        "elapsed_sec":  elapsed,
        "utilization":  util,
        "timing":       timing,
        "power":        power,
        "log_tail":     tail,
        "error":        None,
    }


# ---------------------------------------------------------------------------
# Public: simulation
# ---------------------------------------------------------------------------

def run_simulation(
    job_id: str,
    testbench_file_id: str,
    testbench_top: str,
    runtime: str,
) -> dict:
    """
    Blocking function — call from a thread-pool executor.

    Runs Vivado behavioral simulation; returns a result dict with keys:
    elapsed_sec, log_tail, error.
    """
    state = get_job(job_id)
    board  = BOARD_CATALOG[state["board_id"]]
    part   = board["part"]
    fids   = state["file_ids"]
    work   = job_dir(job_id)

    tcl = _simulation_tcl(
        job_id, part, fids, testbench_file_id, testbench_top, runtime
    )
    log_file = work / "simulation.log"

    t0 = time.monotonic()
    try:
        rc, log_text = _run_vivado(tcl, log_file, settings.simulation_timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return {
            "elapsed_sec": elapsed,
            "error": f"Simulation timed out after {settings.simulation_timeout}s",
            "log_tail": "",
        }
    elapsed = time.monotonic() - t0
    tail = _tail(log_text)

    if rc != 0:
        return {
            "elapsed_sec": elapsed,
            "error": f"Vivado exited with code {rc}",
            "log_tail": tail,
        }

    # Vivado exits 0 even when the testbench reports errors, so the log
    # itself has to be checked.
    tb_error = check_simulation_log(log_text)
    return {"elapsed_sec": elapsed, "log_tail": tail, "error": tb_error}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fmax(clock_period_ns: float | None, wns_ns: float | None) -> dict:
    """Constraint used and the implied maximum frequency (period − WNS)."""
    if clock_period_ns is None:
        return {"clock_period_ns": None, "fmax_mhz": None}
    fmax = None
    if wns_ns is not None and clock_period_ns - wns_ns > 0:
        fmax = round(1000.0 / (clock_period_ns - wns_ns), 2)
    return {"clock_period_ns": clock_period_ns, "fmax_mhz": fmax}


def _load_and_parse(path: Path, parser) -> dict | None:
    if not path.exists():
        return None
    try:
        return parser(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
