"""Sanity tests for the 3x3 cube environment.

If any of these fail, the move permutations are wrong — don't train on top of
them and don't touch the inference path until these are green.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepcube import cube3
from deepcube.cube3 import (
    MOVE_PERMS,
    MOVES,
    N_MOVES,
    N_STICKERS,
    SOLVED,
    apply_move,
    batch_scramble,
    format_moves,
    inverse_move,
    is_solved,
    one_hot,
    parse_moves,
    scramble,
)


def test_constants():
    assert N_STICKERS == 54
    assert N_MOVES == 12
    assert SOLVED.shape == (54,)
    assert MOVE_PERMS.shape == (12, 54)
    assert len(MOVES) == 12


def test_solved_state_colors():
    # 9 stickers per face, colors 0..5
    for face in range(6):
        assert np.array_equal(SOLVED[face * 9 : (face + 1) * 9], np.full(9, face))


def test_is_solved():
    assert is_solved(SOLVED)
    assert not is_solved(apply_move(SOLVED, 0))


@pytest.mark.parametrize("move_idx", range(N_MOVES))
def test_face_turn_order_4(move_idx: int):
    """Every face turn has order 4: applying it 4 times returns to the starting state."""
    s = SOLVED.copy()
    for _ in range(4):
        s = apply_move(s, move_idx)
    assert is_solved(s), f"{MOVES[move_idx]} applied 4 times should be identity"


@pytest.mark.parametrize("move_idx", range(N_MOVES))
def test_move_and_inverse_cancel(move_idx: int):
    s = apply_move(SOLVED, move_idx)
    assert not is_solved(s), f"{MOVES[move_idx]} from solved must change state"
    s = apply_move(s, inverse_move(move_idx))
    assert is_solved(s), f"{MOVES[move_idx]} then its inverse should be identity"


def test_sexy_move_order_6():
    """(R U R' U') has order 6 — repeat 6 times returns to solved."""
    sexy = [MOVES.index(x) for x in ("R", "U", "R'", "U'")]
    s = SOLVED.copy()
    for _ in range(6):
        for m in sexy:
            s = apply_move(s, m)
    assert is_solved(s)


def test_random_scramble_then_reverse():
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = int(rng.integers(1, 30))
        s, mvs = scramble(k, rng)
        for m in mvs[::-1]:
            s = apply_move(s, inverse_move(int(m)))
        assert is_solved(s)


def test_apply_move_returns_new_array():
    """apply_move is functional: the input state must not be mutated."""
    s = SOLVED.copy()
    s2 = apply_move(s, 0)
    assert np.array_equal(s, SOLVED), "apply_move mutated its input"
    assert not np.array_equal(s, s2)


def test_batch_scramble_shape_and_range():
    rng = np.random.default_rng(1)
    states, ks = batch_scramble(32, 10, rng)
    assert states.shape == (32, 54)
    assert ks.shape == (32,)
    assert ks.min() >= 1 and ks.max() <= 10
    assert states.min() >= 0 and states.max() <= 5


def test_batch_scramble_reproducible():
    """Same seed -> same states. Guards against subtle loop/dtype regressions."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1, k1 = batch_scramble(64, 8, rng1)
    s2, k2 = batch_scramble(64, 8, rng2)
    assert np.array_equal(s1, s2)
    assert np.array_equal(k1, k2)


def test_batch_scramble_short_walks_can_return_to_solved():
    """Random walks of length up to 10 on the 12-generator cube group return to
    the identity a small but nonzero fraction of the time. This is the exact
    case that pollutes AVI targets if not handled — the training loop zeros
    targets for such states; this test just documents that they exist."""
    rng = np.random.default_rng(0)
    states, _ = batch_scramble(1000, 10, rng)
    n_solved = int((states == SOLVED).all(axis=1).sum())
    # Empirically around 1% for max_k=10. Assert a generous band.
    assert 1 <= n_solved <= 100, f"implausible: {n_solved}/1000 random scrambles back at solved"


def test_one_hot_single():
    oh = one_hot(SOLVED)
    assert oh.shape == (54 * 6,)
    assert oh.dtype == np.float32
    assert oh.sum() == 54.0
    # Each sticker has exactly one 1 in its 6-wide slot
    assert (oh.reshape(54, 6).sum(axis=1) == 1.0).all()


def test_one_hot_batch():
    rng = np.random.default_rng(3)
    states, _ = batch_scramble(8, 5, rng)
    oh = one_hot(states)
    assert oh.shape == (8, 54 * 6)
    assert oh.dtype == np.float32
    assert (oh.reshape(8, 54, 6).sum(axis=2) == 1.0).all()


def test_parse_moves_basic():
    assert parse_moves("R U R' U'") == [MOVES.index(m) for m in ("R", "U", "R'", "U'")]


def test_parse_moves_double():
    # F2 is sugar for F F
    assert parse_moves("F2") == [MOVES.index("F"), MOVES.index("F")]
    # mixed
    seq = parse_moves("R U2 R'")
    assert seq == [MOVES.index("R"), MOVES.index("U"), MOVES.index("U"), MOVES.index("R'")]


def test_parse_moves_rejects_garbage():
    with pytest.raises(ValueError):
        parse_moves("R X R'")
    with pytest.raises(ValueError):
        parse_moves("X2")


def test_format_parse_roundtrip():
    rng = np.random.default_rng(9)
    for _ in range(20):
        seq = [int(x) for x in rng.integers(0, N_MOVES, size=int(rng.integers(1, 15)))]
        assert parse_moves(format_moves(seq)) == seq


def test_move_perms_are_permutations():
    """Each row of MOVE_PERMS must be a bijection on 0..53."""
    for i in range(N_MOVES):
        perm = MOVE_PERMS[i]
        assert sorted(perm.tolist()) == list(range(54)), f"move {MOVES[i]} is not a permutation"


def test_module_public_api_exports():
    for name in cube3.__all__:
        assert hasattr(cube3, name), f"cube3.__all__ lists {name!r} but attribute is missing"
