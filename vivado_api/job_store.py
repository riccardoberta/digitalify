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
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from models import JobStatus, UUID_PATTERN
from config import settings

_UUID_RE = re.compile(UUID_PATTERN)

#: Extensions accepted by POST /upload.
ALLOWED_UPLOAD_EXTENSIONS = frozenset({".vhd", ".vhdl", ".xdc"})

# Uploaded filenames end up in generated TCL scripts and on disk, so only a
# conservative character set is accepted (no spaces, quotes, $, [, ], {, }).
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_MAX_FILENAME_LEN = 128


class InvalidUploadError(ValueError):
    """Raised when an uploaded file has an unacceptable name."""

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

def is_valid_id(value: str) -> bool:
    """True if *value* is a hyphenated UUID (the only form of job/file ID)."""
    return isinstance(value, str) and bool(_UUID_RE.fullmatch(value))


def _check_id(value: str) -> str:
    r"""
    Return *value* unchanged if it is a valid ID, else raise ValueError.

    IDs are used as directory names, so anything other than a UUID (e.g.
    ``..``, ``a/b``, or a Windows ``a\b``) must never reach the filesystem.
    """
    if not is_valid_id(value):
        raise ValueError(f"Invalid ID: {value!r}")
    return value


def _jobs_root() -> Path:
    p = Path(settings.workspace_dir) / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_dir(job_id: str) -> Path:
    return _jobs_root() / _check_id(job_id)


def uploads_dir() -> Path:
    p = Path(settings.workspace_dir) / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def upload_path(file_id: str) -> Path:
    return uploads_dir() / _check_id(file_id)


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
    """Write *state* atomically (temp file + rename). Caller holds the job lock."""
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
    file_ids = [_check_id(fid) for fid in file_ids]
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
    with _job_lock(job_id):
        _write_state(job_id, state)
    return state


def get_job(job_id: str) -> dict:
    """Return the current state dict, or raise FileNotFoundError."""
    if not is_valid_id(job_id) or not _state_path(job_id).exists():
        raise FileNotFoundError(job_id)
    return _read_state(job_id)


def update_job(job_id: str, **fields) -> dict:
    """
    Merge *fields* into the state dict, update *updated_at*, and persist.
    The whole read-modify-write runs under the job lock, so concurrent
    updates cannot overwrite each other. Returns the updated state.
    """
    with _job_lock(_check_id(job_id)):
        state = _read_state(job_id)
        state.update(fields)
        state["updated_at"] = _now_iso()
        _write_state(job_id, state)
        return state


def transition_job(
    job_id: str, allowed_from: set[str] | frozenset[str], to: str, **fields
) -> tuple[bool, dict]:
    """
    Atomically move a job to status *to* if its current status is in
    *allowed_from*.  Returns ``(True, new_state)`` on success, or
    ``(False, current_state)`` if the job was in another state (nothing is
    written in that case).  Raises FileNotFoundError for unknown jobs.
    """
    if not is_valid_id(job_id):
        raise FileNotFoundError(job_id)
    with _job_lock(_check_id(job_id)):
        if not _state_path(job_id).exists():
            raise FileNotFoundError(job_id)
        state = _read_state(job_id)
        if state["status"] not in allowed_from:
            return False, state
        state.update(fields)
        state["status"] = to
        state["updated_at"] = _now_iso()
        _write_state(job_id, state)
        return True, state


def list_jobs() -> list[dict]:
    """Return all job states (unsorted)."""
    root = _jobs_root()
    result = []
    for jdir in root.iterdir():
        sp = jdir / "state.json"
        if sp.exists():
            try:
                result.append(json.loads(sp.read_text(encoding="utf-8")))
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Uploaded file registry
# ---------------------------------------------------------------------------

def sanitize_filename(filename: str | None) -> str:
    r"""
    Reduce a client-supplied filename to a safe basename, or raise
    InvalidUploadError.

    Any directory part is dropped (both ``/`` and ``\`` separators, so a
    Windows-style path sent to a Linux server is handled too), and the
    result must be a plain name with an allowed extension.
    """
    if not filename:
        raise InvalidUploadError("Missing filename")
    name = PurePosixPath(filename.replace("\\", "/")).name
    if name in ("", ".", ".."):
        raise InvalidUploadError(f"Invalid filename: {filename!r}")
    if len(name) > _MAX_FILENAME_LEN or not _FILENAME_RE.fullmatch(name):
        raise InvalidUploadError(
            f"Invalid filename {filename!r}: use only letters, digits, '_', '-' and '.'"
        )
    ext = PurePosixPath(name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise InvalidUploadError(
            f"Unsupported file type {filename!r}; allowed extensions: {allowed}"
        )
    return name


def register_upload(filename: str | None, content: bytes) -> dict:
    """
    Save *content* to the uploads directory under a sanitised filename.
    Returns a dict with file_id, filename, size_bytes.
    Raises InvalidUploadError if *filename* is not acceptable.
    """
    name = sanitize_filename(filename)
    file_id = str(uuid.uuid4())
    dest = upload_path(file_id)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / name).write_bytes(content)

    # Also store metadata alongside the file
    meta = {"file_id": file_id, "filename": name, "size_bytes": len(content)}
    (dest / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def get_upload_meta(file_id: str) -> dict:
    if not is_valid_id(file_id):
        raise FileNotFoundError(file_id)
    meta_path = upload_path(file_id) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(file_id)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def resolve_upload_path(file_id: str) -> Path:
    """Return the full filesystem path to the uploaded file."""
    meta = get_upload_meta(file_id)
    # Re-sanitise: never build a path from stored metadata blindly.
    return upload_path(file_id) / sanitize_filename(meta["filename"])
