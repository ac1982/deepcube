"""Tests for the BWAS solver.

Uses a zero-heuristic (`h(s) = 0` everywhere) so the search reduces to BFS
from the scrambled state to the solved state — this exercises all the search
mechanics (batching, closed set, stale-entry skipping, path reconstruction)
without depending on a trained model.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepcube.cube3 import MOVES, SOLVED, apply_move, inverse_move, scramble
from deepcube.search import bwas_solve


class _ZeroHeuristic(torch.nn.Module):
    """h(s) = 0 for all s. Turns BWAS into batched BFS."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0])


def _verify_path_solves(start: np.ndarray, path: list[int]) -> bool:
    s = start.copy()
    for a in path:
        s = apply_move(s, a)
    return np.array_equal(s, SOLVED)


def test_start_already_solved():
    net = _ZeroHeuristic()
    r = bwas_solve(net, SOLVED.copy())
    assert r.solved
    assert r.path == []
    assert r.path_length == 0


@pytest.mark.parametrize("move_idx", range(12))
def test_one_move_scramble(move_idx: int):
    """Any one-move scramble is solved by applying that move's inverse.

    With a zero heuristic and BFS-ish expansion, the solver finds the optimal
    1-step path.
    """
    net = _ZeroHeuristic()
    start = apply_move(SOLVED, move_idx)
    r = bwas_solve(net, start, batch_size=32)
    assert r.solved, f"failed on 1-move scramble {MOVES[move_idx]}: {r.stop_reason}"
    assert r.path_length == 1
    assert r.path == [inverse_move(move_idx)]
    assert _verify_path_solves(start, r.path)


def test_two_move_scramble_finds_optimal():
    """A 2-move scramble (U F) has the 2-move inverse (F' U') as its unique
    shortest solution with the zero heuristic."""
    net = _ZeroHeuristic()
    start = apply_move(apply_move(SOLVED, MOVES.index("U")), MOVES.index("F"))
    r = bwas_solve(net, start, batch_size=64)
    assert r.solved
    assert r.path_length == 2
    assert _verify_path_solves(start, r.path)


def test_random_short_scramble_solved():
    """BFS with batching should still handle short random scrambles."""
    rng = np.random.default_rng(0)
    net = _ZeroHeuristic()
    for _ in range(5):
        start, _ = scramble(3, rng)
        r = bwas_solve(net, start, batch_size=128, max_iterations=500)
        assert r.solved, f"unsolved 3-move scramble: reason={r.stop_reason}"
        assert r.path_length <= 3
        assert _verify_path_solves(start, r.path)


def test_max_nodes_safety_cap():
    """If max_nodes is tiny, the solver gracefully returns unsolved with the
    right stop_reason instead of OOM-ing or hanging."""
    net = _ZeroHeuristic()
    rng = np.random.default_rng(1)
    start, _ = scramble(6, rng)
    r = bwas_solve(net, start, batch_size=64, max_nodes=100)
    # Either we got lucky and solved it, or we tripped the cap — both fine,
    # we're just checking the termination path.
    assert r.solved or r.stop_reason == "max_nodes"


def test_solve_result_path_names():
    net = _ZeroHeuristic()
    start = apply_move(SOLVED, MOVES.index("R"))
    r = bwas_solve(net, start)
    assert r.path_names == ["R'"]


def test_batch_size_1_equivalent_to_serial():
    """BWAS with batch_size=1 is essentially serial A*. Should still solve."""
    net = _ZeroHeuristic()
    start = apply_move(SOLVED, MOVES.index("D"))
    r = bwas_solve(net, start, batch_size=1)
    assert r.solved and r.path_length == 1


def test_lambda_greater_than_one_still_solves():
    """With a zero heuristic, lambda_weight has no effect; the search should
    still terminate correctly."""
    net = _ZeroHeuristic()
    start = apply_move(SOLVED, MOVES.index("F"))
    r = bwas_solve(net, start, lambda_weight=5.0)
    assert r.solved
    assert r.path_length == 1
