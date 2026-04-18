"""FastAPI endpoint tests.

`/solve` is covered with a zero-heuristic net injected in place of the real
trained model, so the whole HTTP path is verified without depending on a
checkpoint existing on disk.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from deepcube import server
from deepcube.cube3 import MOVES, SOLVED, apply_move


class _ZeroHeuristic(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0])


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


@pytest.fixture
def client_with_net(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Inject a zero-heuristic net so /solve works without a real checkpoint."""
    monkeypatch.setattr(server, "_net", _ZeroHeuristic())
    return TestClient(server.app)


@pytest.fixture
def client_no_net(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server, "_net", None)
    return TestClient(server.app)


def test_healthz_shape(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "model_loaded" in body
    assert "checkpoint_path" in body


def test_healthz_reports_model_loaded(client_with_net: TestClient):
    r = client_with_net.get("/healthz")
    assert r.json()["model_loaded"] is True


def test_healthz_reports_no_model(client_no_net: TestClient):
    r = client_no_net.get("/healthz")
    assert r.json()["model_loaded"] is False


def test_scramble_returns_54_stickers_and_move_list(client: TestClient):
    r = client.post("/scramble", json={"depth": 5, "seed": 42})
    assert r.status_code == 200
    body = r.json()
    assert len(body["state"]) == 54
    assert len(body["moves"]) == 5
    for m in body["moves"]:
        assert m in MOVES


def test_scramble_deterministic_with_seed(client: TestClient):
    r1 = client.post("/scramble", json={"depth": 10, "seed": 123}).json()
    r2 = client.post("/scramble", json={"depth": 10, "seed": 123}).json()
    assert r1 == r2


def test_scramble_rejects_bad_depth(client: TestClient):
    r = client.post("/scramble", json={"depth": 0})
    assert r.status_code == 422
    r = client.post("/scramble", json={"depth": 999})
    assert r.status_code == 422


def test_solve_without_model_returns_503(client_no_net: TestClient):
    r = client_no_net.post("/solve", json={"state": SOLVED.tolist()})
    assert r.status_code == 503
    assert "checkpoint" in r.json()["detail"].lower()


def test_solve_trivial_already_solved(client_with_net: TestClient):
    r = client_with_net.post("/solve", json={"state": SOLVED.tolist()})
    assert r.status_code == 200
    body = r.json()
    assert body["solved"] is True
    assert body["moves"] == []
    assert body["path_length"] == 0


def test_solve_one_move_scramble(client_with_net: TestClient):
    start = apply_move(SOLVED, MOVES.index("R"))
    r = client_with_net.post("/solve", json={"state": start.tolist(), "batch_size": 64})
    assert r.status_code == 200
    body = r.json()
    assert body["solved"] is True
    assert body["moves"] == ["R'"]
    assert body["path_length"] == 1
    # Verify by replaying
    s = np.array(start, dtype=np.int8)
    for m in body["moves"]:
        s = apply_move(s, MOVES.index(m))
    assert np.array_equal(s, SOLVED)


def test_solve_rejects_wrong_length_state(client_with_net: TestClient):
    r = client_with_net.post("/solve", json={"state": [0] * 10})
    assert r.status_code == 422


def test_solve_rejects_out_of_range_values(client_with_net: TestClient):
    bad = [0] * 54
    bad[0] = 9
    r = client_with_net.post("/solve", json={"state": bad})
    assert r.status_code == 400
    assert "0..5" in r.json()["detail"]


def test_root_missing_frontend_is_404(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """If the frontend hasn't been written yet, `/` should fail clearly."""
    monkeypatch.setattr(server, "_STATIC_DIR", server._REPO_ROOT / "does-not-exist")
    r = client.get("/")
    assert r.status_code == 404
    assert "frontend" in r.json()["detail"].lower()
