"""Authentication and startup configuration checks."""

from __future__ import annotations

import pytest

import main
from config import settings


def test_missing_token_is_401(client):
    del client.headers["Authorization"]
    assert client.get("/boards").status_code == 401


def test_wrong_token_is_401(client):
    client.headers["Authorization"] = "Bearer wrong"
    assert client.get("/boards").status_code == 401


def test_non_ascii_token_is_401_not_500(client):
    # compare_digest() on str raises TypeError for non-ASCII; we compare bytes.
    del client.headers["Authorization"]
    r = client.get("/boards", headers={"Authorization": "Bearer tést".encode("utf-8")})
    assert r.status_code == 401


def test_correct_token(client):
    assert client.get("/boards").status_code == 200


def test_health_is_public(client):
    del client.headers["Authorization"]
    assert client.get("/health").status_code == 200


def test_startup_refused_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "allow_no_auth", False)
    with pytest.raises(RuntimeError, match="ALLOW_NO_AUTH"):
        main.check_auth_config()

    # Also through the real app lifespan
    from fastapi.testclient import TestClient
    with pytest.raises(RuntimeError):
        with TestClient(main.app):
            pass


def test_startup_refused_with_placeholder_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "change-me-to-a-strong-secret")
    with pytest.raises(RuntimeError, match="placeholder"):
        main.check_auth_config()


def test_explicit_no_auth_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "allow_no_auth", True)
    main.check_auth_config()  # does not raise

    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        assert c.get("/boards").status_code == 200


def test_fails_closed_if_startup_check_bypassed(monkeypatch):
    """Without the lifespan (no `with`), an empty key must not mean open access."""
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "allow_no_auth", False)
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    assert c.get("/boards").status_code == 503
