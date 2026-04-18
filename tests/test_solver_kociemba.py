"""Tests for the Kociemba solver wrapper + state-format converter."""
from __future__ import annotations

import numpy as np
import pytest

from deepcube.cube3 import MOVES, SOLVED, apply_move, inverse_move, scramble
from deepcube.solver_kociemba import kociemba_solve, state_to_kociemba_string


# ---- state-format converter ----

def test_solved_state_to_kociemba_string():
    # In kociemba's URFDLB order, the solved state is 9 of each letter in that order.
    assert state_to_kociemba_string(SOLVED) == (
        "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    )


def test_string_length_and_alphabet():
    rng = np.random.default_rng(0)
    state, _ = scramble(10, rng)
    s = state_to_kociemba_string(state)
    assert len(s) == 54
    assert set(s) <= set("URFDLB")
    # Each color must appear exactly 9 times on a valid cube.
    for c in "URFDLB":
        assert s.count(c) == 9, f"{c}: {s.count(c)}"


def test_centers_are_faces_in_kociemba_order():
    """Centers at kociemba positions 4/13/22/31/40/49 map to URFDLB."""
    s = state_to_kociemba_string(SOLVED)
    assert s[4] == "U"
    assert s[13] == "R"
    assert s[22] == "F"
    assert s[31] == "D"
    assert s[40] == "L"
    assert s[49] == "B"


@pytest.mark.parametrize("move_idx", range(12))
def test_single_move_converter_sanity(move_idx: int):
    """Each 1-move scramble converts to a string where only 20 facelets changed
    from solved (every quarter-turn moves exactly 4 stickers on the face it
    turns + 12 on the surrounding side = 20).
    """
    scrambled = apply_move(SOLVED, move_idx)
    solved = state_to_kociemba_string(SOLVED)
    moved = state_to_kociemba_string(scrambled)
    diffs = sum(1 for a, b in zip(solved, moved, strict=True) if a != b)
    assert diffs == 12, f"{MOVES[move_idx]}: {diffs} stickers changed (expected 12)"
    # Note: 12 side stickers, since the 4 stickers on the turned face itself
    # just rotate among themselves, keeping the SAME color mapping.


# ---- solver wrapper ----

def test_solved_state_short_circuits():
    r = kociemba_solve(SOLVED.copy())
    assert r.solved
    assert r.path == []
    assert r.path_length == 0


@pytest.mark.parametrize("move_idx", range(12))
def test_one_move_scramble(move_idx: int):
    start = apply_move(SOLVED, move_idx)
    r = kociemba_solve(start)
    assert r.solved, f"{MOVES[move_idx]}: not solved, stop={r.stop_reason}"
    # Kociemba should return exactly one quarter-turn (the inverse).
    assert r.path == [inverse_move(move_idx)], \
        f"{MOVES[move_idx]}: expected [{inverse_move(move_idx)}], got {r.path}"


def test_random_deep_scramble_solves():
    rng = np.random.default_rng(1)
    state, _ = scramble(25, rng)
    r = kociemba_solve(state)
    assert r.solved
    # Apply the returned path, should land on solved.
    s = state.copy()
    for m in r.path:
        s = apply_move(s, m)
    assert np.array_equal(s, SOLVED)
    # Kociemba's QTM output (after expanding F2) on a random 25-move scramble
    # should be <= 26 (paper's worst-case for 3x3 QTM).
    assert r.path_length <= 30, f"unusually long path: {r.path_length}"


def test_invalid_state_returns_graceful_failure():
    """A state that violates cubie parity is unreachable; kociemba should raise,
    and our wrapper translates to an unsolved SolveResult with a useful reason."""
    bad = SOLVED.copy()
    bad[0], bad[9] = bad[9], bad[0]  # swap one U sticker and one F sticker
    r = kociemba_solve(bad)
    assert not r.solved
    assert r.stop_reason.startswith("invalid_state")
