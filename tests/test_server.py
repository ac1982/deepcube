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


def test_meta_shape(client: TestClient):
    r = client.get("/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["n_stickers"] == 54
    assert body["n_moves"] == 12
    assert len(body["moves"]) == 12
    assert len(body["move_perms"]) == 12
    assert all(len(p) == 54 for p in body["move_perms"])
    assert len(body["solved_state"]) == 54
    # Each perm is a bijection on 0..53
    for p in body["move_perms"]:
        assert sorted(p) == list(range(54))
    # Sticker layout
    assert len(body["sticker_positions"]) == 54
    assert len(body["sticker_normals"]) == 54
    assert all(len(p) == 3 for p in body["sticker_positions"])
    assert all(len(n) == 3 for n in body["sticker_normals"])
    # Centers: 6 indices, each paired with a positive center-sticker color
    assert len(body["center_indices"]) == 6
    for face, idx in enumerate(body["center_indices"]):
        assert body["solved_state"][idx] == face
    # (position, normal) pairs must be unique — they're the raycast lookup key.
    seen = set()
    for p, n in zip(body["sticker_positions"], body["sticker_normals"], strict=True):
        key = (tuple(p), tuple(n))
        assert key not in seen, f"duplicate sticker layout entry: {key}"
        seen.add(key)


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


def test_solve_deepcube_without_model_returns_503(client_no_net: TestClient):
    r = client_no_net.post("/solve", json={"state": SOLVED.tolist(), "solver": "deepcube"})
    assert r.status_code == 503
    assert "checkpoint" in r.json()["detail"].lower()


def test_solve_deepcube_trivial_already_solved(client_with_net: TestClient):
    r = client_with_net.post("/solve", json={"state": SOLVED.tolist(), "solver": "deepcube"})
    assert r.status_code == 200
    body = r.json()
    assert body["solver"] == "deepcube"
    assert body["solved"] is True
    assert body["moves"] == []
    assert body["path_length"] == 0
    assert body["path_length_htm"] is None  # BWAS doesn't produce HTM


def test_solve_deepcube_one_move_scramble(client_with_net: TestClient):
    start = apply_move(SOLVED, MOVES.index("R"))
    r = client_with_net.post("/solve", json={
        "state": start.tolist(), "solver": "deepcube", "batch_size": 64,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["solved"] is True
    assert body["moves"] == ["R'"]
    assert body["path_length"] == 1
    s = np.array(start, dtype=np.int8)
    for m in body["moves"]:
        s = apply_move(s, MOVES.index(m))
    assert np.array_equal(s, SOLVED)


def test_solve_kociemba_works_without_model(client_no_net: TestClient):
    """Kociemba path is independent of the trained net — should work even when
    /healthz says model_loaded=false."""
    start = apply_move(SOLVED, MOVES.index("R"))
    r = client_no_net.post("/solve", json={"state": start.tolist(), "solver": "kociemba"})
    assert r.status_code == 200
    body = r.json()
    assert body["solver"] == "kociemba"
    assert body["solved"] is True
    assert body["moves"] == ["R'"]
    assert body["path_length"] == 1
    assert body["path_length_htm"] == 1  # one HTM move


def test_solve_kociemba_is_the_default(client_no_net: TestClient):
    """Omitting `solver` should default to kociemba, which works without the net."""
    start = apply_move(SOLVED, MOVES.index("U"))
    r = client_no_net.post("/solve", json={"state": start.tolist()})
    assert r.status_code == 200
    body = r.json()
    assert body["solver"] == "kociemba"
    assert body["solved"] is True


def test_solve_kociemba_has_no_compare_field(client_no_net: TestClient):
    """Picking Kociemba should NOT also run DeepCubeA — DeepCubeA is slow
    and the whole point of compare is to show AI *vs* classical when the
    user specifically asked for AI."""
    r = client_no_net.post("/solve", json={"state": SOLVED.tolist(), "solver": "kociemba"})
    body = r.json()
    assert body["compare"] is None


def test_solve_deepcube_includes_kociemba_compare(client_with_net: TestClient):
    """When the user picks DeepCubeA, the response must include a Kociemba
    result on the same state, so the UI can render the comparison card."""
    start = apply_move(SOLVED, MOVES.index("U"))
    r = client_with_net.post("/solve", json={
        "state": start.tolist(), "solver": "deepcube", "batch_size": 64,
    })
    body = r.json()
    assert body["solver"] == "deepcube"
    assert body["compare"] is not None
    c = body["compare"]
    assert c["solver"] == "kociemba"
    assert c["solved"] is True
    assert c["path_length"] == 1
    assert c["path_length_htm"] == 1
    assert c["moves"] == ["U'"]
    # The compare result should not depend on the DeepCubeA primary having
    # succeeded — it's an independent run.
    assert c["elapsed_sec"] >= 0


def test_solve_kociemba_reports_htm_on_half_turn(client_no_net: TestClient):
    """A U2 scramble (two quarter turns on the same face) is one HTM move
    but two QTM moves; kociemba should report the difference."""
    start = apply_move(apply_move(SOLVED, MOVES.index("U")), MOVES.index("U"))
    r = client_no_net.post("/solve", json={"state": start.tolist(), "solver": "kociemba"})
    body = r.json()
    assert body["solved"] is True
    assert body["path_length"] == 2         # two quarter-turns to undo U2
    assert body["path_length_htm"] == 1     # counted as one HTM half-turn


def test_solve_rejects_wrong_length_state(client_no_net: TestClient):
    # Uses kociemba path (no net needed) but validation happens before dispatch.
    r = client_no_net.post("/solve", json={"state": [0] * 10})
    assert r.status_code == 422


def test_solve_rejects_out_of_range_values(client_no_net: TestClient):
    bad = [0] * 54
    bad[0] = 9
    r = client_no_net.post("/solve", json={"state": bad})
    assert r.status_code == 400
    assert "0..5" in r.json()["detail"]


def test_solve_rejects_unknown_solver(client_no_net: TestClient):
    r = client_no_net.post("/solve", json={"state": SOLVED.tolist(), "solver": "magic"})
    assert r.status_code == 422  # pydantic Literal rejection


def test_root_missing_frontend_is_404(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """If the frontend hasn't been written yet, `/` should fail clearly."""
    monkeypatch.setattr(server, "_STATIC_DIR", server._REPO_ROOT / "does-not-exist")
    r = client.get("/")
    assert r.status_code == 404
    assert "frontend" in r.json()["detail"].lower()
