"""
File-based job persistence.

Every job lives in its own directory under ``{workspace_dir}/jobs/{job_id}/``.
The mutable state is kept in ``state.json``; all Vivado artefacts (logs,
reports) land in the same directory.

Thread-safety: writes are serialised with a per-job threading.Lock so that
concurrent API calls (e.g. a status poll racing with a background synthesis
callback) cannot corrupt state.json.  This is cross-platform (no fcntl /
msvcrt dependency) and sufficient for a single-process uvicorn deployment.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from models import JobStatus
from config import settings

# Per-job write locks — created on first access, held for the duration of a
# read-modify-write cycle only.
_locks: dict[str, threading.Lock] = {}
_locks_meta = threading.Lock()


def _job_lock(job_id: str) -> threading.Lock:
    with _locks_meta:
        if job_id not in _locks:
            _locks[job_id] = threading.Lock()
        return _locks[job_id]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _jobs_root() -> Path:
    p = Path(settings.workspace_dir) / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_dir(job_id: str) -> Path:
    return _jobs_root() / job_id


def uploads_dir() -> Path:
    p = Path(settings.workspace_dir) / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def upload_path(file_id: str) -> Path:
    return uploads_dir() / file_id


def _state_path(job_id: str) -> Path:
    return job_dir(job_id) / "state.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_state(job_id: str) -> dict:
    with open(_state_path(job_id), encoding="utf-8") as fh:
        return json.load(fh)


def _write_state(job_id: str, state: dict) -> None:
    """Write *state* atomically under a per-job threading lock."""
    with _job_lock(job_id):
        tmp = _state_path(job_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(_state_path(job_id))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_job(board_id: str, top_entity: str, file_ids: list[str]) -> dict:
    """
    Create a new job directory and initial state.json.
    Returns the full state dict (which includes the job_id).
    """
    job_id = str(uuid.uuid4())
    jdir = job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    now = _now_iso()
    state = {
        "job_id":     job_id,
        "status":     JobStatus.PENDING.value,
        "board_id":   board_id,
        "top_entity": top_entity,
        "file_ids":   file_ids,
        "created_at": now,
        "updated_at": now,
        "error":      None,
        "synthesis":  None,
        "simulation": None,
    }
    _write_state(job_id, state)
    return state


def get_job(job_id: str) -> dict:
    """Return the current state dict, or raise FileNotFoundError."""
    if not _state_path(job_id).exists():
        raise FileNotFoundError(job_id)
    return _read_state(job_id)


def update_job(job_id: str, **fields) -> dict:
    """
    Merge *fields* into the state dict, update *updated_at*, and persist.
    Returns the updated state.
    """
    state = _read_state(job_id)
    state.update(fields)
    state["updated_at"] = _now_iso()
    _write_state(job_id, state)
    return state


def list_jobs() -> list[dict]:
    """Return all job states (unsorted)."""
    root = _jobs_root()
    result = []
    for jdir in root.iterdir():
        sp = jdir / "state.json"
        if sp.exists():
            try:
                result.append(json.loads(sp.read_text()))
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Uploaded file registry
# ---------------------------------------------------------------------------

def register_upload(filename: str, content: bytes) -> dict:
    """
    Save *content* to the uploads directory.
    Returns a dict with file_id, filename, size_bytes.
    """
    file_id = str(uuid.uuid4())
    dest = upload_path(file_id)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / filename).write_bytes(content)

    # Also store metadata alongside the file
    meta = {"file_id": file_id, "filename": filename, "size_bytes": len(content)}
    (dest / "meta.json").write_text(json.dumps(meta))
    return meta


def get_upload_meta(file_id: str) -> dict:
    meta_path = upload_path(file_id) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(file_id)
    return json.loads(meta_path.read_text())


def resolve_upload_path(file_id: str) -> Path:
    """Return the full filesystem path to the uploaded file."""
    meta = get_upload_meta(file_id)
    return upload_path(file_id) / meta["filename"]
