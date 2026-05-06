"""
GHDL subprocess runner.

Drives the three-step GHDL workflow:
  1. ``ghdl -a`` — analyze (compile) each VHDL file
  2. ``ghdl -e`` — elaborate (link) the testbench entity
  3. ``ghdl -r`` — run the simulation

All steps target VHDL-2008 (``--std=08``), which is required for the
``to_hstring`` function used in the testbench.

Typical usage
-------------
>>> result = run_ghdl(vhdl_files, tb_filename, tb_source, tb_entity="tb_mlp")
>>> print(result.summary())
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GHDLResult:
    """
    Structured outcome of a GHDL simulation run.

    Attributes
    ----------
    passed : bool
        True when all test vectors passed and GHDL exited successfully.
    n_passed : int
        Number of test vectors that produced the expected output.
    n_failed : int
        Number of test vectors that produced wrong output.
    log : str
        Full concatenated output from all GHDL sub-commands.
    vcd_path : Path | None
        Path to the generated VCD waveform file, or None if not produced.
    work_dir : Path
        Directory containing all intermediate files.  Deleted automatically
        unless ``keep_work_dir=True`` was passed to ``run_ghdl``.
    """
    passed:    bool
    n_passed:  int
    n_failed:  int
    log:       str
    vcd_path:  Path | None
    work_dir:  Path

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"[{status}]  {self.n_passed} passed, {self.n_failed} failed"
            + (f"  — VCD: {self.vcd_path}" if self.vcd_path else "")
        )


# ---------------------------------------------------------------------------
# GHDL sub-command wrappers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    """Run *cmd* and return (returncode, combined stdout+stderr)."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _analyze(ghdl: str, vhd_path: Path, work_dir: Path) -> tuple[int, str]:
    return _run(
        [ghdl, "-a", "--std=08", f"--workdir={work_dir}", str(vhd_path)],
        cwd=work_dir,
    )


def _elaborate(ghdl: str, entity: str, work_dir: Path) -> tuple[int, str]:
    return _run(
        [ghdl, "-e", "--std=08", f"--workdir={work_dir}", entity],
        cwd=work_dir,
    )


def _simulate(
    ghdl: str,
    entity: str,
    work_dir: Path,
    vcd: bool,
    stop_time: str,
) -> tuple[int, str]:
    cmd = [ghdl, "-r", "--std=08", f"--workdir={work_dir}", entity]
    if vcd:
        cmd.append(f"--vcd={work_dir / 'waveform.vcd'}")
    cmd.append(f"--stop-time={stop_time}")
    return _run(cmd, cwd=work_dir)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _parse_counts(log: str) -> tuple[int, int]:
    """
    Extract (n_passed, n_failed) from the testbench summary line.

    The testbench emits:
        TB SUMMARY: X passed, Y failed out of N tests
    """
    m = re.search(r"TB SUMMARY:\s*(\d+) passed,\s*(\d+) failed", log)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Fallback: count individual PASSED / FAILED lines
    n_passed = len(re.findall(r"Test \d+: PASSED", log))
    n_failed = len(re.findall(r"Test \d+: FAILED", log))
    return n_passed, n_failed


# ---------------------------------------------------------------------------
# Analysis file ordering
# ---------------------------------------------------------------------------

def _analysis_order(vhdl_files: dict[str, str], tb_filename: str) -> list[str]:
    """
    Return filenames in a safe analysis order for GHDL.

    GHDL requires that a component's entity + architecture be analyzed before
    any architecture that instantiates it.  The VHDLGenerator already outputs
    files in topological order (leaf components first, top-level last), so we
    just put the top-level before the testbench.

    Order:
      1. Component files (everything that is not the top-level or testbench)
      2. Top-level file  (*_top.vhd)
      3. Testbench file
    """
    top_files  = [f for f in vhdl_files if f.endswith("_top.vhd")]
    comp_files = [f for f in vhdl_files if f not in top_files]
    return comp_files + top_files + [tb_filename]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ghdl(
    vhdl_files: dict[str, str],
    tb_filename: str,
    tb_source: str,
    tb_entity: str,
    ghdl_path: str = "ghdl",
    work_dir: Path | None = None,
    keep_work_dir: bool = False,
    vcd: bool = True,
    stop_time: str = "10us",
) -> GHDLResult:
    """
    Run a full GHDL analyze → elaborate → simulate cycle.

    Parameters
    ----------
    vhdl_files : dict[str, str]
        Design VHDL files as ``{filename: vhdl_source}``, e.g. from
        ``VHDLGenerator.generate()``.
    tb_filename : str
        Filename for the testbench (e.g. ``"tb_mlp.vhd"``).
    tb_source : str
        VHDL source text of the testbench.
    tb_entity : str
        Name of the testbench entity to elaborate and run
        (e.g. ``"tb_mlp"``).
    ghdl_path : str
        Name or full path of the GHDL executable.
    work_dir : Path | None
        Directory for intermediate files.  A unique temporary directory is
        created when ``None``; it is deleted after the run unless
        ``keep_work_dir=True``.
    keep_work_dir : bool
        Preserve the working directory after the run (useful for inspecting
        generated files or loading the VCD into GTKWave).
    vcd : bool
        Dump a VCD waveform to ``{work_dir}/waveform.vcd``.
    stop_time : str
        Simulation stop-time safety limit (e.g. ``"10us"``).  The testbench
        terminates naturally before this; the limit is a fallback only.

    Returns
    -------
    GHDLResult
        Structured result with pass/fail counts, log, and VCD path.
    """
    # ── Working directory setup ───────────────────────────────────────────────
    tmp_dir: str | None = None
    if work_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="digitalify_ghdl_")
        work_dir = Path(tmp_dir)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def _fail(msg: str) -> GHDLResult:
        return GHDLResult(
            passed=False, n_passed=0, n_failed=0,
            log="\n".join(log_lines) + f"\n{msg}",
            vcd_path=None,
            work_dir=work_dir,
        )

    try:
        # ── Write all VHDL to disk ────────────────────────────────────────────
        all_files = {**vhdl_files, tb_filename: tb_source}
        for fname, src in all_files.items():
            (work_dir / fname).write_text(src, encoding="utf-8")

        # ── Step 1: Analyze ───────────────────────────────────────────────────
        log_lines.append("=== GHDL Analyze ===")
        order = _analysis_order(vhdl_files, tb_filename)

        for fname in order:
            rc, out = _analyze(ghdl_path, work_dir / fname, work_dir)
            status = "OK" if rc == 0 else "FAILED"
            log_lines.append(f"  {fname}: {status}")
            if out:
                log_lines.append(out)
            if rc != 0:
                return _fail(f"Analysis failed on '{fname}'")

        # ── Step 2: Elaborate ─────────────────────────────────────────────────
        log_lines.append("\n=== GHDL Elaborate ===")
        rc, out = _elaborate(ghdl_path, tb_entity, work_dir)
        log_lines.append(f"  {tb_entity}: {'OK' if rc == 0 else 'FAILED'}")
        if out:
            log_lines.append(out)
        if rc != 0:
            return _fail(f"Elaboration failed for entity '{tb_entity}'")

        # ── Step 3: Simulate ──────────────────────────────────────────────────
        log_lines.append("\n=== GHDL Simulate ===")
        rc, out = _simulate(ghdl_path, tb_entity, work_dir, vcd, stop_time)
        if out:
            log_lines.append(out)

        full_log = "\n".join(log_lines)
        n_passed, n_failed = _parse_counts(full_log)
        passed = (rc == 0) and (n_failed == 0) and (n_passed > 0)

        vcd_path = work_dir / "waveform.vcd"
        vcd_path = vcd_path if (vcd and vcd_path.exists()) else None

        return GHDLResult(
            passed=passed,
            n_passed=n_passed,
            n_failed=n_failed,
            log=full_log,
            vcd_path=vcd_path,
            work_dir=work_dir,
        )

    finally:
        if tmp_dir and not keep_work_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
