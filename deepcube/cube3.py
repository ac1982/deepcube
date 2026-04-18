"""3x3 Rubik's cube environment.

State representation
--------------------
54-sticker flat array of color indices 0..5 (U=0, F=1, R=2, B=3, L=4, D=5).
Index layout, standard unfolded net:

            U0 U1 U2
            U3 U4 U5
            U6 U7 U8
    L0..L8  F0..F8  R0..R8  B0..B8
            D0 D1 D2
            D3 D4 D5
            D6 D7 D8

U=0..8, F=9..17, R=18..26, B=27..35, L=36..44, D=45..53.

Moves
-----
12 quarter-turns `U, U', D, D', F, F', B, B', R, R', L, L'`, ordered in pairs
so that `inverse(i) == i ^ 1`. Each move is a 54-permutation compiled once at
import time by rotating the affected layer in 3-space and re-mapping
(cubie-position, sticker-normal) to sticker index.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "N_STICKERS",
    "N_MOVES",
    "SOLVED",
    "MOVES",
    "MOVE_PERMS",
    "apply_move",
    "scramble",
    "batch_scramble",
    "is_solved",
    "one_hot",
    "inverse_move",
    "parse_moves",
    "format_moves",
]

N_STICKERS = 54
N_MOVES = 12

# Solved state: 9 stickers per face, color = face index.
SOLVED: np.ndarray = np.repeat(np.arange(6, dtype=np.int8), 9)
assert SOLVED.shape == (N_STICKERS,)


# ---------------------------------------------------------------------------
# Sticker layout: sticker_index -> (cubie_position, outward_normal)
# ---------------------------------------------------------------------------

def _build_sticker_info() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    info: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    # U (+y): net row order = back-to-front (z: -1 -> 1), col order = left-to-right (x: -1 -> 1)
    for z in (-1, 0, 1):
        for x in (-1, 0, 1):
            info.append(((x, 1, z), (0, 1, 0)))
    # F (+z): y top-to-bottom (1 -> -1), x left-to-right
    for y in (1, 0, -1):
        for x in (-1, 0, 1):
            info.append(((x, y, 1), (0, 0, 1)))
    # R (+x): y top-to-bottom, z front-to-back (1 -> -1)
    for y in (1, 0, -1):
        for z in (1, 0, -1):
            info.append(((1, y, z), (1, 0, 0)))
    # B (-z): y top-to-bottom, x right-to-left (1 -> -1) because B's net is to the right of R
    for y in (1, 0, -1):
        for x in (1, 0, -1):
            info.append(((x, y, -1), (0, 0, -1)))
    # L (-x): y top-to-bottom, z back-to-front (-1 -> 1) because L's net is to the left of F
    for y in (1, 0, -1):
        for z in (-1, 0, 1):
            info.append(((-1, y, z), (-1, 0, 0)))
    # D (-y): z front-to-back (1 -> -1) so D's top row meets F's bottom row, x left-to-right
    for z in (1, 0, -1):
        for x in (-1, 0, 1):
            info.append(((x, -1, z), (0, -1, 0)))
    return info


_STICKER_INFO = _build_sticker_info()
assert len(_STICKER_INFO) == N_STICKERS
_POS_NORMAL_TO_IDX = {key: i for i, key in enumerate(_STICKER_INFO)}


# ---------------------------------------------------------------------------
# Move permutations
# ---------------------------------------------------------------------------

def _rot(axis: int, sign: int) -> np.ndarray:
    """Return the integer 90-degree rotation matrix about `axis` (0=x, 1=y, 2=z).

    `sign = -1` is clockwise by the right-hand rule; `sign = +1` is CCW.
    Uses cos(90) = 0, sin(+/-90) = +/-1, so every entry stays integer.
    """
    c, s = 0, sign
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=int)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=int)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=int)


# (layer_axis, layer_value, rotation_axis, sign). CW from outside = -90 deg about outward normal.
_MOVE_SPECS: dict[str, tuple[int, int, int, int]] = {
    "U":  (1, +1, 1, -1),
    "U'": (1, +1, 1, +1),
    "D":  (1, -1, 1, +1),
    "D'": (1, -1, 1, -1),
    "F":  (2, +1, 2, -1),
    "F'": (2, +1, 2, +1),
    "B":  (2, -1, 2, +1),
    "B'": (2, -1, 2, -1),
    "R":  (0, +1, 0, -1),
    "R'": (0, +1, 0, +1),
    "L":  (0, -1, 0, +1),
    "L'": (0, -1, 0, -1),
}
MOVES: list[str] = list(_MOVE_SPECS.keys())
assert len(MOVES) == N_MOVES
_MOVE_NAME_TO_IDX = {name: i for i, name in enumerate(MOVES)}


def _build_perm(layer_axis: int, layer_value: int, rot_axis: int, sign: int) -> np.ndarray:
    R = _rot(rot_axis, sign)
    perm = np.arange(N_STICKERS, dtype=np.int64)
    for old_idx, (pos, normal) in enumerate(_STICKER_INFO):
        if pos[layer_axis] != layer_value:
            continue
        new_pos = tuple(int(v) for v in R @ np.array(pos))
        new_normal = tuple(int(v) for v in R @ np.array(normal))
        new_idx = _POS_NORMAL_TO_IDX[(new_pos, new_normal)]
        perm[new_idx] = old_idx
    return perm


MOVE_PERMS: np.ndarray = np.stack([_build_perm(*_MOVE_SPECS[m]) for m in MOVES])
assert MOVE_PERMS.shape == (N_MOVES, N_STICKERS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_move(state: np.ndarray, move_idx: int) -> np.ndarray:
    """Return a new state after applying move `move_idx` to `state`."""
    return state[MOVE_PERMS[move_idx]]


def inverse_move(move_idx: int) -> int:
    """Return the index of the inverse of `move_idx` (paired so that `m ^ 1` inverts m)."""
    return move_idx ^ 1


def is_solved(state: np.ndarray) -> bool:
    return bool(np.array_equal(state, SOLVED))


def scramble(k: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply `k` uniformly-random moves to the solved cube.

    Returns (state, moves) where `moves` is the sequence of applied move indices.
    """
    rng = rng if rng is not None else np.random.default_rng()
    mvs = rng.integers(0, N_MOVES, size=k)
    s = SOLVED.copy()
    for m in mvs:
        s = apply_move(s, int(m))
    return s, mvs


def batch_scramble(
    n: int,
    max_k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate `n` scrambled states with per-sample depth `k ~ Uniform[1, max_k]`.

    Fixed `max_k` loop bound so this mirrors the GPU training loop (no branch on
    `ks.max()`). Extra random moves past each sample's own depth are masked out.

    Returns (states: (n, 54) int8, ks: (n,) int64).
    """
    ks = rng.integers(1, max_k + 1, size=n)
    states = np.tile(SOLVED, (n, 1))
    moves = rng.integers(0, N_MOVES, size=(n, max_k))
    idx_b = np.arange(n)[:, None]
    for t in range(max_k):
        perms = MOVE_PERMS[moves[:, t]]
        new_states = states[idx_b, perms]
        active = (t < ks)[:, None]
        states = np.where(active, new_states, states)
    return states, ks


def one_hot(states: np.ndarray) -> np.ndarray:
    """Encode states as a float32 one-hot tensor.

    Accepts either a single state of shape (54,) or a batch of shape (n, 54).
    Returns shape (54*6,) or (n, 54*6).
    """
    states = np.asarray(states)
    if states.ndim == 1:
        oh = np.zeros((N_STICKERS, 6), dtype=np.float32)
        oh[np.arange(N_STICKERS), states] = 1.0
        return oh.reshape(N_STICKERS * 6)
    n = states.shape[0]
    oh = np.zeros((n, N_STICKERS, 6), dtype=np.float32)
    # Fancy index along the last dim
    oh[np.arange(n)[:, None], np.arange(N_STICKERS)[None, :], states] = 1.0
    return oh.reshape(n, N_STICKERS * 6)


def parse_moves(text: str) -> list[int]:
    """Parse a whitespace-separated move string like "R U R' U'" into move indices.

    Accepts `F2` as sugar for `F F` (since the training env is quarter-turn only).
    Raises `ValueError` on unknown tokens.
    """
    out: list[int] = []
    for tok in text.split():
        if tok.endswith("2"):
            base = tok[:-1]
            if base not in _MOVE_NAME_TO_IDX:
                raise ValueError(f"unknown move: {tok!r}")
            idx = _MOVE_NAME_TO_IDX[base]
            out.append(idx)
            out.append(idx)
            continue
        if tok not in _MOVE_NAME_TO_IDX:
            raise ValueError(f"unknown move: {tok!r}")
        out.append(_MOVE_NAME_TO_IDX[tok])
    return out


def format_moves(indices: list[int]) -> str:
    """Format a move index sequence as a human-readable string."""
    return " ".join(MOVES[i] for i in indices)
