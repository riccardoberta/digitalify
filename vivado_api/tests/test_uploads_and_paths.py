"""Upload filename sanitising and ID/path validation."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

import job_store
from job_store import InvalidUploadError, sanitize_filename


# --- sanitize_filename ---------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("top.vhd", "top.vhd"),
    ("my_design-v2.vhdl", "my_design-v2.vhdl"),
    ("constraints.XDC", "constraints.XDC"),
    ("../../evil.vhd", "evil.vhd"),
    ("/etc/evil.vhd", "evil.vhd"),
    ("..\\..\\Windows\\evil.vhd", "evil.vhd"),
    ("C:\\Users\\me\\top.vhd", "top.vhd"),
    ("sub/dir/top.vhd", "top.vhd"),
])
def test_sanitize_filename_keeps_basename(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", [
    None, "", ".", "..", "../", "dir/", "..\\",
    "evil.py", "meta.json", "top.vhd.exe", "noext",
    ".vhd",                       # hidden file, no stem
    "top file.vhd",               # space
    "x[exec calc].vhd",           # TCL command substitution
    "$x.vhd", "{x}.vhd", 'x".vhd',
    "top.vhd\n",
    "a" * 200 + ".vhd",
])
def test_sanitize_filename_rejects(raw):
    with pytest.raises(InvalidUploadError):
        sanitize_filename(raw)


# --- POST /upload ----------------------------------------------------------------------

def test_upload_traversal_is_contained(client):
    r = client.post("/upload", files=[("files", ("../../../evil.vhd", b"x", "text/plain"))])
    assert r.status_code == 200, r.text
    meta = r.json()["files"][0]
    assert meta["filename"] == "evil.vhd"

    path = job_store.resolve_upload_path(meta["file_id"]).resolve()
    uploads = job_store.uploads_dir().resolve()
    assert path.is_relative_to(uploads)
    assert path.read_bytes() == b"x"
    # Nothing escaped into the workspace root or above
    assert not (uploads.parent / "evil.vhd").exists()
    assert not (uploads.parent.parent / "evil.vhd").exists()


@pytest.mark.parametrize("name", ["evil.py", "x[exec calc].vhd", "..", "top file.vhd"])
def test_upload_rejects_bad_names(client, name):
    before = set(job_store.uploads_dir().iterdir())
    r = client.post("/upload", files=[
        ("files", ("good.vhd", b"ok", "text/plain")),
        ("files", (name, b"x", "text/plain")),
    ])
    assert r.status_code in (400, 422), r.text
    # The batch is validated up front: the good file was not stored either
    assert set(job_store.uploads_dir().iterdir()) == before


# --- ID validation --------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["..", "../jobs", "a/b", "a\\b", "x", str(uuid.uuid4()).upper(), ""])
def test_job_store_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        job_store.job_dir(bad)
    with pytest.raises(ValueError):
        job_store.upload_path(bad)
    with pytest.raises(FileNotFoundError):
        job_store.get_job(bad)
    with pytest.raises(FileNotFoundError):
        job_store.get_upload_meta(bad)


@pytest.mark.parametrize("bad", ["not-a-uuid", "..%5C..%5Cetc", "a%5Cb"])
def test_job_routes_reject_bad_ids(client, bad):
    for path in (f"/jobs/{bad}", f"/jobs/{bad}/synthesis", f"/jobs/{bad}/log",
                 f"/jobs/{bad}/reports", f"/jobs/{bad}/simulation"):
        assert client.get(path).status_code == 422, path


def test_unknown_job_is_404(client):
    assert client.get(f"/jobs/{uuid.uuid4()}").status_code == 404


def test_create_job_with_unknown_file_is_400(client):
    r = client.post("/jobs", json={
        "board_id": "basys3", "top_entity": "top", "file_ids": [str(uuid.uuid4())],
    })
    assert r.status_code == 400


# --- locking ---------------------------------------------------------------------------

def test_transition_is_atomic(uploaded):
    state = job_store.create_job("basys3", "top", [uploaded["top.vhd"]])
    jid = state["job_id"]
    results: list[bool] = []
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        ok, _ = job_store.transition_job(jid, {"pending", "failed"}, "synthesizing")
        results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1
    assert job_store.get_job(jid)["status"] == "synthesizing"


def test_concurrent_updates_are_not_lost(uploaded):
    state = job_store.create_job("basys3", "top", [uploaded["top.vhd"]])
    jid = state["job_id"]
    n = 20
    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait()
        job_store.update_job(jid, **{f"field_{i}": i})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = job_store.get_job(jid)
    assert all(final.get(f"field_{i}") == i for i in range(n))


def test_second_synthesize_request_conflicts(client, uploaded, monkeypatch):
    import main

    # Keep the background task from running Vivado
    async def _noop(*a, **k):
        return None
    monkeypatch.setattr(main, "_background_synthesize", _noop)

    r = client.post("/jobs", json={
        "board_id": "basys3", "top_entity": "top", "file_ids": [uploaded["top.vhd"]],
    })
    jid = r.json()["job_id"]
    assert client.post(f"/jobs/{jid}/synthesize").json()["status"] == "synthesizing"
    assert client.post(f"/jobs/{jid}/synthesize").status_code == 409
