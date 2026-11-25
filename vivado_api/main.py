"""
Vivado API — FastAPI application.

Start with (an API key is required, see below):
    uvicorn main:app --host 0.0.0.0 --port 8000

Or for local development with auto-reload, bound to localhost only:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload

Environment variables (see config.py for defaults):
    VIVADO_API_API_KEY       bearer token (required)
    VIVADO_API_ALLOW_NO_AUTH set to "true" to start without an API key
                             (development only, never with --host 0.0.0.0)
    VIVADO_API_VIVADO_PATH   path to vivado binary
    VIVADO_API_WORKSPACE_DIR root directory for jobs and uploads
    VIVADO_API_SYNTHESIS_TIMEOUT  seconds (default 3600)
    VIVADO_API_SIMULATION_TIMEOUT seconds (default 600)
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI, BackgroundTasks, Depends, File, HTTPException,
    Path as PathParam, Security, UploadFile, status,
)
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import job_store
from config import settings, BOARD_CATALOG
from models import (
    BoardInfo, BoardListResponse,
    CreateJobRequest, JobResponse, JobStatus, UUID_PATTERN,
    ReportsResponse, SimulationRequest, SimulationResult,
    SynthesisRequest, SynthesisResult,
    UploadResponse, UploadedFile,
    UtilizationReport, TimingReport, PowerReport,
)
from vivado.runner import run_synthesis, run_simulation


logger = logging.getLogger("vivado_api")

# Placeholder value shipped in .env.example — never accept it as a real key.
_PLACEHOLDER_KEYS = {"change-me-to-a-strong-secret"}


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def check_auth_config() -> None:
    """
    Refuse to run with an unsafe authentication configuration.

    Without an API key anyone who can reach the port can run Vivado (and
    write files) on this machine, so an empty key is only accepted when
    VIVADO_API_ALLOW_NO_AUTH=true is set explicitly.
    """
    if settings.api_key in _PLACEHOLDER_KEYS:
        raise RuntimeError(
            "VIVADO_API_API_KEY is still the placeholder from .env.example; "
            "set a strong random secret."
        )
    if not settings.api_key:
        if not settings.allow_no_auth:
            raise RuntimeError(
                "VIVADO_API_API_KEY is not set. Set an API key, or set "
                "VIVADO_API_ALLOW_NO_AUTH=true to run WITHOUT authentication "
                "(local development only; bind to 127.0.0.1)."
            )
        logger.warning(
            "Authentication is DISABLED (VIVADO_API_ALLOW_NO_AUTH=true). "
            "Do not expose this server beyond localhost."
        )


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    check_auth_config()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    lifespan=_lifespan,
    title="Vivado API",
    description=(
        "REST API for Xilinx Vivado synthesis and simulation. "
        "Upload VHDL sources, select a target board, trigger synthesis, "
        "run simulations, and retrieve performance reports."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)]
) -> None:
    """Dependency: check the Bearer token against VIVADO_API_API_KEY."""
    if not settings.api_key:
        if settings.allow_no_auth:
            return   # auth explicitly disabled
        # Misconfigured server (startup check bypassed): fail closed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server has no API key configured",
        )
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode("utf-8"),
        settings.api_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


Auth = Annotated[None, Depends(verify_token)]

# Job IDs are UUIDs; validating them here keeps "..", "a\\b" etc. away from
# the filesystem (job_store re-checks as defence in depth).
JobId = Annotated[str, PathParam(pattern=UUID_PATTERN, description="Job ID (UUID)")]


# ---------------------------------------------------------------------------
# Helper: state dict → JobResponse
# ---------------------------------------------------------------------------

def _job_response(state: dict) -> JobResponse:
    return JobResponse(
        job_id=state["job_id"],
        status=JobStatus(state["status"]),
        board_id=state["board_id"],
        top_entity=state["top_entity"],
        created_at=state["created_at"],
        updated_at=state["updated_at"],
        error=state.get("error"),
    )


def _require_job(job_id: str) -> dict:
    try:
        return job_store.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _background_synthesize(job_id: str, opts: SynthesisRequest) -> None:
    # Status was already set to SYNTHESIZING atomically by trigger_synthesis.
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_synthesis(
                job_id,
                strategy=opts.strategy,
                flatten_hierarchy=opts.flatten_hierarchy,
            ),
        )
    except Exception as exc:
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=str(exc),
            synthesis={"error": str(exc)},
        )
        return

    if result.get("error"):
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=result["error"],
            synthesis=result,
        )
    else:
        job_store.update_job(
            job_id,
            status=JobStatus.SYNTHESIZED.value,
            synthesis=result,
        )


async def _background_simulate(
    job_id: str, req: SimulationRequest
) -> None:
    # Status was already set to SIMULATING atomically by trigger_simulation.
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_simulation(
                job_id,
                testbench_file_id=req.testbench_file_id,
                testbench_top=req.testbench_top,
                runtime=req.runtime,
            ),
        )
    except Exception as exc:
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=str(exc),
            simulation={"error": str(exc)},
        )
        return

    if result.get("error"):
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=result["error"],
            simulation=result,
        )
    else:
        job_store.update_job(
            job_id,
            status=JobStatus.DONE.value,
            simulation=result,
        )


# ---------------------------------------------------------------------------
# Routes — health & boards
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health() -> dict:
    """Quick liveness check."""
    vivado_available = Path(settings.vivado_path).exists()
    return {
        "status": "ok",
        "vivado_path": settings.vivado_path,
        "vivado_available": vivado_available,
        "workspace_dir": settings.workspace_dir,
        "auth_enabled": bool(settings.api_key),
    }


@app.get("/boards", response_model=BoardListResponse, tags=["boards"])
def list_boards(_: Auth) -> BoardListResponse:
    """Return all boards available for synthesis."""
    boards = [
        BoardInfo(board_id=bid, **info)
        for bid, info in BOARD_CATALOG.items()
    ]
    return BoardListResponse(boards=boards)


# ---------------------------------------------------------------------------
# Routes — file upload
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse, tags=["files"])
async def upload_files(
    _: Auth,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    """
    Upload one or more VHDL source files.
    Returns a list of ``file_id`` tokens to use when creating a job.
    """
    # Validate every name before writing anything, so a bad file in the
    # batch does not leave the others half-uploaded.
    for f in files:
        try:
            job_store.sanitize_filename(f.filename)
        except job_store.InvalidUploadError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    uploaded = []
    for f in files:
        content = await f.read()
        meta = job_store.register_upload(f.filename, content)
        uploaded.append(UploadedFile(**meta))
    return UploadResponse(files=uploaded)


# ---------------------------------------------------------------------------
# Routes — jobs
# ---------------------------------------------------------------------------

@app.post(
    "/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["jobs"],
)
def create_job(_: Auth, req: CreateJobRequest) -> JobResponse:
    """
    Create a new synthesis job.

    You must upload sources first with ``POST /upload`` and supply the
    returned ``file_id`` values here.  The job starts in ``pending`` state;
    call ``POST /jobs/{job_id}/synthesize`` to trigger Vivado.
    """
    if req.board_id not in BOARD_CATALOG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown board '{req.board_id}'. Call GET /boards for the list.",
        )
    # Validate all file IDs exist
    for fid in req.file_ids:
        try:
            job_store.get_upload_meta(fid)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"File ID '{fid}' not found")

    state = job_store.create_job(
        board_id=req.board_id,
        top_entity=req.top_entity,
        file_ids=req.file_ids,
    )
    return _job_response(state)


@app.get("/jobs", response_model=list[JobResponse], tags=["jobs"])
def list_jobs(_: Auth) -> list[JobResponse]:
    """List all jobs on this server."""
    return [_job_response(s) for s in job_store.list_jobs()]


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"])
def get_job(_: Auth, job_id: JobId) -> JobResponse:
    """Return the current status and metadata of a job."""
    return _job_response(_require_job(job_id))


# ---------------------------------------------------------------------------
# Routes — synthesis
# ---------------------------------------------------------------------------

@app.post(
    "/jobs/{job_id}/synthesize",
    response_model=JobResponse,
    tags=["synthesis"],
)
async def trigger_synthesis(
    _: Auth,
    job_id: JobId,
    opts: SynthesisRequest = SynthesisRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JobResponse:
    """
    Start Vivado synthesis for this job in the background.

    The job must be in ``pending`` or ``failed`` state.
    Poll ``GET /jobs/{job_id}`` until status is ``synthesized`` or ``failed``,
    then fetch results from ``GET /jobs/{job_id}/synthesis``.
    """
    allowed = {JobStatus.PENDING.value, JobStatus.FAILED.value}
    try:
        # Check + transition under the job lock, so two concurrent requests
        # cannot both start Vivado on the same job.
        ok, state = job_store.transition_job(
            job_id, allowed, JobStatus.SYNTHESIZING.value,
            error=None, synthesis=None, simulation=None,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot synthesize: job is in state '{state['status']}'",
        )

    background_tasks.add_task(_background_synthesize, job_id, opts)
    return _job_response(state)


@app.get(
    "/jobs/{job_id}/synthesis",
    response_model=SynthesisResult,
    tags=["synthesis"],
)
def get_synthesis_result(_: Auth, job_id: JobId) -> SynthesisResult:
    """
    Return synthesis results (utilization, timing, power, log tail).

    Available once the job status is ``synthesized`` or ``failed``.
    """
    state = _require_job(job_id)
    syn = state.get("synthesis") or {}

    def _util(d):
        return UtilizationReport(**d) if d else None
    def _timing(d):
        return TimingReport(**d) if d else None
    def _power(d):
        return PowerReport(**d) if d else None

    return SynthesisResult(
        status=JobStatus(state["status"]),
        elapsed_sec=syn.get("elapsed_sec"),
        utilization=_util(syn.get("utilization")),
        timing=_timing(syn.get("timing")),
        power=_power(syn.get("power")),
        log_tail=syn.get("log_tail"),
        error=syn.get("error"),
    )


# ---------------------------------------------------------------------------
# Routes — log
# ---------------------------------------------------------------------------

@app.get(
    "/jobs/{job_id}/log",
    response_class=PlainTextResponse,
    tags=["synthesis"],
)
def get_log(_: Auth, job_id: JobId, type: str = "synthesis") -> PlainTextResponse:
    """
    Return the full Vivado log file for this job as plain text.

    The ``type`` query parameter selects which log to retrieve:
    - ``synthesis``  (default) — written during ``POST /jobs/{job_id}/synthesize``
    - ``simulation`` — written during ``POST /jobs/{job_id}/simulate``

    Unlike the ``log_tail`` field in the synthesis/simulation result endpoints,
    this returns the complete file with no truncation.
    """
    _require_job(job_id)
    work = job_store.job_dir(job_id)

    log_files = {
        "synthesis":  work / "synthesis.log",
        "simulation": work / "simulation.log",
    }
    if type not in log_files:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown log type '{type}'. Valid values: synthesis, simulation.",
        )

    path = log_files[type]
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No {type} log found for job '{job_id}'. Has the {type} step run yet?",
        )

    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Routes — reports
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/reports", response_model=ReportsResponse, tags=["synthesis"])
def get_reports(_: Auth, job_id: JobId) -> ReportsResponse:
    """
    Return parsed utilization, timing and power reports, plus the raw text
    of each report file for detailed inspection.
    """
    state = _require_job(job_id)
    if state["status"] not in (JobStatus.SYNTHESIZED.value, JobStatus.DONE.value):
        raise HTTPException(
            status_code=409,
            detail=f"Reports not ready: job is in state '{state['status']}'",
        )

    syn  = state.get("synthesis") or {}
    work = job_store.job_dir(job_id)

    raw: dict[str, str] = {}
    for rpt in ("utilization.rpt", "timing.rpt", "power.rpt"):
        path = work / rpt
        if path.exists():
            raw[rpt.replace(".rpt", "")] = path.read_text(
                encoding="utf-8", errors="replace"
            )

    def _util(d): return UtilizationReport(**d) if d else None
    def _timing(d): return TimingReport(**d) if d else None
    def _power(d): return PowerReport(**d) if d else None

    return ReportsResponse(
        utilization=_util(syn.get("utilization")),
        timing=_timing(syn.get("timing")),
        power=_power(syn.get("power")),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Routes — simulation
# ---------------------------------------------------------------------------

@app.post(
    "/jobs/{job_id}/simulate",
    response_model=JobResponse,
    tags=["simulation"],
)
async def trigger_simulation(
    _: Auth,
    job_id: JobId,
    req: SimulationRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JobResponse:
    """
    Upload a testbench file ID and launch behavioral simulation.

    The job must be in ``synthesized`` state (synthesis must complete first).
    Poll ``GET /jobs/{job_id}`` until status is ``done`` or ``failed``,
    then fetch results from ``GET /jobs/{job_id}/simulation``.
    """
    _require_job(job_id)
    # Validate testbench file before touching the job state
    try:
        job_store.get_upload_meta(req.testbench_file_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=f"Testbench file ID '{req.testbench_file_id}' not found",
        )

    try:
        ok, state = job_store.transition_job(
            job_id, {JobStatus.SYNTHESIZED.value}, JobStatus.SIMULATING.value,
            simulation=None,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot simulate: job is in state '{state['status']}'. "
                "Synthesis must complete successfully first."
            ),
        )

    background_tasks.add_task(_background_simulate, job_id, req)
    return _job_response(state)


@app.get(
    "/jobs/{job_id}/simulation",
    response_model=SimulationResult,
    tags=["simulation"],
)
def get_simulation_result(_: Auth, job_id: JobId) -> SimulationResult:
    """
    Return simulation results (log tail, elapsed time, error if any).

    Available once the job status is ``done`` or ``failed``.
    """
    state = _require_job(job_id)
    sim = state.get("simulation") or {}

    return SimulationResult(
        status=JobStatus(state["status"]),
        elapsed_sec=sim.get("elapsed_sec"),
        log_tail=sim.get("log_tail"),
        error=sim.get("error"),
    )
