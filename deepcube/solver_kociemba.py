"""Kociemba two-phase solver wrapper.

Kociemba's 1992 algorithm is the classic near-optimal cube solver: it
decomposes the problem into two reductions (to a subgroup and then within
it) and uses IDA* with precomputed pruning tables. The C-backed `kociemba`
PyPI package runs in milliseconds and produces solutions of at most 20
moves in the **Half-Turn Metric** (HTM). Included here alongside DeepCubeA
for comparison.

Facelet mapping
---------------
kociemba uses a 54-character string with face order **U R F D L B** and
face-letter colors. Our `cube3` module uses integer state with face order
**U F R B L D**. Within-face ordering is the same (top-to-bottom,
left-to-right of the unfolded net) so only the face order differs.
"""
from __future__ import annotations

import time

import kociemba
import numpy as np

from deepcube.cube3 import SOLVED, parse_moves
from deepcube.search import SolveResult

__all__ = ["state_to_kociemba_string", "kociemba_solve"]

# Our color index (0..5) -> kociemba face letter. Colors correspond to the face
# a sticker would be on in solved state, so this is just (face_idx -> letter).
_COLOR_TO_LETTER = {0: "U", 1: "F", 2: "R", 3: "B", 4: "L", 5: "D"}

# Within our state, slices of each face. In the ORDER kociemba expects (URFDLB).
# Face order: U F R B L D  (ours) -> pick out U, R, F, D, L, B.
_FACE_SLICES_IN_KOCIEMBA_ORDER = [
    slice(0, 9),    # U -> U
    slice(18, 27),  # R -> R
    slice(9, 18),   # F -> F
    slice(45, 54),  # D -> D
    slice(36, 45),  # L -> L
    slice(27, 36),  # B -> B
]


def state_to_kociemba_string(state: np.ndarray) -> str:
    """Convert a 54-int cube state into kociemba's facelet string."""
    assert state.shape == (54,), f"bad state shape: {state.shape}"
    out: list[str] = []
    for sl in _FACE_SLICES_IN_KOCIEMBA_ORDER:
        for v in state[sl]:
            out.append(_COLOR_TO_LETTER[int(v)])
    return "".join(out)


def kociemba_solve(state: np.ndarray) -> SolveResult:
    """Solve a cube using the Kociemba algorithm.

    Returns the same `SolveResult` shape as `bwas_solve` so callers can treat
    the two backends uniformly.
    """
    t0 = time.time()

    # The underlying library doesn't special-case the solved state: asked to
    # solve an already-solved cube it returns a 13-HTM identity sequence
    # ("R L U2 R L' B2 U2 R2 F2 L2 D2 L2 F2") rather than the empty string.
    # Short-circuit that case for sanity.
    if np.array_equal(state, SOLVED):
        return SolveResult(True, [], 0, 0, 1, time.time() - t0, "solved")

    try:
        raw = kociemba.solve(state_to_kociemba_string(state))
    except ValueError as e:
        # kociemba raises on unreachable / malformed states
        return SolveResult(False, [], 0, 0, 0, time.time() - t0, f"invalid_state:{e}")

    if not raw:
        return SolveResult(True, [], 0, 0, 1, time.time() - t0, "solved", path_length_htm=0)

    htm = len(raw.split())
    # Expand HTM half-turns ("U2" -> "U U") into our quarter-turn metric so
    # the path can be animated by the same frontend path as BWAS's output.
    move_indices = parse_moves(raw)
    return SolveResult(
        solved=True,
        path=move_indices,
        path_length=len(move_indices),
        nodes_expanded=0,    # kociemba doesn't expose these stats
        nodes_generated=0,
        elapsed_sec=time.time() - t0,
        stop_reason="solved",
        path_length_htm=htm,
    )
