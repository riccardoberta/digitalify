"""
Pydantic request / response schemas for the Vivado API.
"""

from __future__ import annotations
from enum import Enum
from typing import Annotated, Literal
from pydantic import BaseModel, Field, StringConstraints
from datetime import datetime


# ---------------------------------------------------------------------------
# Validated string types
# ---------------------------------------------------------------------------
# Every value below ends up interpolated into a generated TCL script, so it
# must be restricted to a safe character set: anything else (brackets, $,
# quotes, braces, whitespace, newlines) would allow TCL command injection.

#: Canonical hyphenated lowercase UUID, as produced by str(uuid.uuid4()).
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

#: Basic VHDL identifier (extended/escaped identifiers are not allowed).
VHDL_IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"

#: Simulation time: a number immediately followed by a VHDL time unit.
RUNTIME_PATTERN = r"^\d+(\.\d+)?(fs|ps|ns|us|ms|s)$"

UUIDStr = Annotated[str, StringConstraints(pattern=UUID_PATTERN)]
VHDLIdentifier = Annotated[
    str, StringConstraints(pattern=VHDL_IDENTIFIER_PATTERN, max_length=128)
]
RuntimeStr = Annotated[str, StringConstraints(pattern=RUNTIME_PATTERN, max_length=32)]

SynthStrategy = Literal[
    "default",
    "RuntimeOptimized",
    "AreaOptimized_high",
    "AreaOptimized_medium",
    "AlternateRoutability",
    "AreaMapLargeShiftRegToBRAM",
    "FewerCarryChains",
]

FlattenHierarchy = Literal["rebuilt", "full", "none"]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING      = "pending"
    SYNTHESIZING = "synthesizing"
    SYNTHESIZED  = "synthesized"
    SIMULATING   = "simulating"
    DONE         = "done"
    FAILED       = "failed"


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------

class BoardInfo(BaseModel):
    board_id:    str
    name:        str
    part:        str
    family:      str
    description: str


class BoardListResponse(BaseModel):
    boards: list[BoardInfo]


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

class UploadedFile(BaseModel):
    file_id:   str
    filename:  str
    size_bytes: int


class UploadResponse(BaseModel):
    files: list[UploadedFile]


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

class CreateJobRequest(BaseModel):
    board_id:    str   = Field(..., description="Board ID from GET /boards")
    top_entity:  VHDLIdentifier = Field(..., description="Name of the VHDL top-level entity")
    file_ids:    list[UUIDStr] = Field(
        ..., min_length=1, description="File IDs returned by POST /upload"
    )


class JobResponse(BaseModel):
    job_id:      str
    status:      JobStatus
    board_id:    str
    top_entity:  str
    created_at:  datetime
    updated_at:  datetime
    error:       str | None = None


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

class SynthesisRequest(BaseModel):
    """Optionally override synthesis options at trigger time."""
    strategy: SynthStrategy = Field(
        default="default",
        description=(
            "Vivado synth_design -directive value. "
            "Valid options: default, RuntimeOptimized, AreaOptimized_high, "
            "AreaOptimized_medium, AlternateRoutability, "
            "AreaMapLargeShiftRegToBRAM, FewerCarryChains."
        ),
    )
    flatten_hierarchy: FlattenHierarchy = Field(
        default="rebuilt",
        description="flatten_hierarchy setting: rebuilt, full, or none.",
    )


class UtilizationReport(BaseModel):
    luts:         int | None = None
    lut_ram:      int | None = None
    flip_flops:   int | None = None
    dsps:         int | None = None
    brams:        int | None = None
    ios:          int | None = None
    lut_pct:      float | None = None
    ff_pct:       float | None = None
    dsp_pct:      float | None = None
    bram_pct:     float | None = None


class TimingReport(BaseModel):
    wns_ns:       float | None = None   # Worst Negative Slack
    tns_ns:       float | None = None   # Total Negative Slack
    whs_ns:       float | None = None   # Worst Hold Slack
    ths_ns:       float | None = None   # Total Hold Slack
    timing_met:   bool  | None = None


class PowerReport(BaseModel):
    total_power_w:   float | None = None
    dynamic_power_w: float | None = None
    static_power_w:  float | None = None
    junction_temp_c: float | None = None
    confidence:      str   | None = None


class SynthesisResult(BaseModel):
    status:       JobStatus
    elapsed_sec:  float | None = None
    utilization:  UtilizationReport | None = None
    timing:       TimingReport      | None = None
    power:        PowerReport       | None = None
    log_tail:     str | None = None    # last N lines of the Vivado log
    error:        str | None = None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    testbench_file_id: UUIDStr = Field(..., description="File ID of the VHDL testbench")
    testbench_top:     VHDLIdentifier = Field(..., description="Name of the testbench entity")
    runtime:           RuntimeStr = Field(
        default="1000ns",
        description="Simulation run time, no spaces (e.g. '1000ns', '1.5us')",
    )


class SimulationResult(BaseModel):
    status:       JobStatus
    elapsed_sec:  float | None = None
    log_tail:     str   | None = None
    error:        str   | None = None


# ---------------------------------------------------------------------------
# Reports (combined endpoint)
# ---------------------------------------------------------------------------

class ReportsResponse(BaseModel):
    utilization:  UtilizationReport | None = None
    timing:       TimingReport      | None = None
    power:        PowerReport       | None = None
    raw:          dict[str, str]    = Field(
        default_factory=dict,
        description="Raw report text keyed by report name (utilization/timing/power)"
    )
