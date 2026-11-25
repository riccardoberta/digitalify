"""
Test configuration.

The vivado_api modules use top-level imports (``import job_store``) and read
their settings from the environment at import time, so the environment and
``sys.path`` are prepared here, before any test module imports them.

Run from the ``vivado_api`` directory:

    python -m pytest tests
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent.parent
TEST_API_KEY = "test-api-key-123"

_WORKSPACE = tempfile.mkdtemp(prefix="vivado_api_tests_")
os.environ["VIVADO_API_WORKSPACE_DIR"] = _WORKSPACE
os.environ["VIVADO_API_API_KEY"] = TEST_API_KEY
os.environ.pop("VIVADO_API_ALLOW_NO_AUTH", None)
# Point at a binary that cannot exist: no test may ever launch Vivado.
os.environ["VIVADO_API_VIVADO_PATH"] = str(Path(_WORKSPACE) / "no-vivado-here")

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_WORKSPACE, ignore_errors=True)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main

    with TestClient(main.app) as c:
        c.headers["Authorization"] = f"Bearer {TEST_API_KEY}"
        yield c


@pytest.fixture
def uploaded(client):
    """Upload a design file and a testbench; return their file IDs."""
    r = client.post(
        "/upload",
        files=[
            ("files", ("top.vhd", b"-- design\n", "text/plain")),
            ("files", ("tb_top.vhd", b"-- tb\n", "text/plain")),
        ],
    )
    assert r.status_code == 200, r.text
    ids = {f["filename"]: f["file_id"] for f in r.json()["files"]}
    return ids
