"""Smoke test for the Studio Flask blueprint."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pytest.importorskip("aiosqlite")
pytest.importorskip("sqlite_vec")


def test_app_boots_and_serves_kit_endpoints():
    from wild_memory.studio.blueprint import create_app

    app = create_app()
    client = app.test_client()

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["ok"] is True
    assert set(health.get_json()["kits"]) == {"1", "2", "3"}

    home = client.get("/")
    assert home.status_code == 200
    body = home.get_data(as_text=True)
    assert "Wild Memory" in body
    assert 'data-kit="1"' in body
    assert 'data-kit="2"' in body
    assert 'data-kit="3"' in body


def test_running_kit_one_returns_pass():
    from wild_memory.studio.blueprint import create_app

    app = create_app()
    client = app.test_client()
    resp = client.post("/api/kit/1/run")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["kit_id"] == "1"
    assert body["passed"] is True, body
    assert body["pass_count"] == body["total_count"]
