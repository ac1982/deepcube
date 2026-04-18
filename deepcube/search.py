"""Batch-Weighted A* (BWAS) solver for the 3x3 cube.

From Agostinelli et al., 2019. Differences from vanilla A*:

1. **Batch expansion.** Each iteration pops the `batch_size` best-priority nodes
   at once, generates all 12 * batch_size children, and runs the heuristic on
   them in a single GPU batch. This amortizes the neural net forward cost over
   many nodes.
2. **Weighted priority.** `priority = g + lambda_weight * h`. `lambda_weight = 1`
   is ordinary A* (optimal if h is admissible). `lambda_weight > 1` biases toward
   greedy expansion — much faster, no longer guaranteed optimal.

Both standard lazy-deletion tricks apply: the closed set records the best `g`
seen for each state, and stale heap entries (those whose g exceeds the closed
value) are skipped on pop.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from deepcube.cube3 import MOVE_PERMS, MOVES, N_MOVES, SOLVED, one_hot

if TYPE_CHECKING:
    from deepcube.model import DeepCubeANet

__all__ = ["SolveResult", "bwas_solve"]

_SOLVED_BYTES = SOLVED.tobytes()


@dataclass
class SolveResult:
    solved: bool
    path: list[int]
    path_length: int   # quarter-turn metric (number of quarter turns)
    nodes_expanded: int
    nodes_generated: int
    elapsed_sec: float
    stop_reason: str  # "solved" | "empty_open" | "max_iterations" | "max_nodes"
    path_length_htm: int | None = None   # half-turn metric count when meaningful (Kociemba)

    @property
    def path_names(self) -> list[str]:
        return [MOVES[i] for i in self.path]


@dataclass(order=True)
class _HeapEntry:
    priority: float
    tie_breaker: int  # breaks ties so heapq never tries to compare node_idx payload
    node_idx: int = field(compare=False)


def bwas_solve(
    net: "DeepCubeANet | torch.nn.Module",
    start_state: np.ndarray,
    *,
    lambda_weight: float = 2.0,
    batch_size: int = 1000,
    max_iterations: int = 10_000,
    max_nodes: int = 5_000_000,
    device: str | torch.device = "cpu",
) -> SolveResult:
    """Solve a scrambled 3x3 cube using the trained heuristic `net`.

    Args:
        net: heuristic network. Called as `net(one_hot_batch)` and expected to
             return a (N,) tensor of estimated costs-to-go.
        start_state: (54,) int array of color indices, the cube to solve.
        lambda_weight: weight on heuristic in priority.
        batch_size: number of OPEN nodes popped and expanded per iteration.
        max_iterations, max_nodes: safety caps.
        device: where to run the net forward.
    """
    assert start_state.shape == (54,), f"bad state shape: {start_state.shape}"

    device = torch.device(device)
    net = net.to(device).eval()

    t0 = time.time()

    # Trivial case: already solved.
    if start_state.tobytes() == _SOLVED_BYTES:
        return SolveResult(True, [], 0, 0, 1, time.time() - t0, "solved")

    # Node table, parallel arrays (dataclass-per-node is too slow for 5M entries).
    nodes_state: list[bytes] = []
    nodes_g: list[int] = []
    nodes_parent: list[int] = []   # -1 for root
    nodes_action: list[int] = []   # -1 for root

    # best g known for each state; used both to prune pushes and to detect stale pops
    closed: dict[bytes, int] = {}

    open_heap: list[_HeapEntry] = []
    tie_counter = 0

    def _push(state_bytes: bytes, g: int, parent: int, action: int, h: float) -> None:
        nonlocal tie_counter
        idx = len(nodes_state)
        nodes_state.append(state_bytes)
        nodes_g.append(g)
        nodes_parent.append(parent)
        nodes_action.append(action)
        heapq.heappush(open_heap, _HeapEntry(g + lambda_weight * h, tie_counter, idx))
        tie_counter += 1

    # Seed: query h(start) and push the root.
    with torch.no_grad():
        h0 = float(net(torch.from_numpy(one_hot(start_state)).unsqueeze(0).to(device)).squeeze(0).cpu())
    start_bytes = start_state.tobytes()
    closed[start_bytes] = 0
    _push(start_bytes, 0, -1, -1, h0)

    nodes_expanded = 0

    for _it in range(max_iterations):
        # Pop up to batch_size non-stale nodes.
        batch_ids: list[int] = []
        while open_heap and len(batch_ids) < batch_size:
            entry = heapq.heappop(open_heap)
            state_bytes = nodes_state[entry.node_idx]
            if closed.get(state_bytes, nodes_g[entry.node_idx]) < nodes_g[entry.node_idx]:
                continue  # stale: a better path to this state was recorded later
            batch_ids.append(entry.node_idx)

        if not batch_ids:
            return SolveResult(False, [], 0, nodes_expanded, len(nodes_state),
                               time.time() - t0, "empty_open")

        # Goal check on popped batch. Halting when the goal is popped (not when
        # it's generated) keeps the semantics consistent: we always return a
        # path corresponding to the first goal node that bubbles to the top of
        # OPEN, and for lambda=1 this is optimal.
        for nid in batch_ids:
            if nodes_state[nid] == _SOLVED_BYTES:
                path: list[int] = []
                cur = nid
                while nodes_parent[cur] != -1:
                    path.append(nodes_action[cur])
                    cur = nodes_parent[cur]
                path.reverse()
                return SolveResult(True, path, len(path), nodes_expanded,
                                   len(nodes_state), time.time() - t0, "solved")

        # Expand.
        parent_ids: list[int] = []
        actions: list[int] = []
        children: list[np.ndarray] = []
        children_g: list[int] = []

        for nid in batch_ids:
            parent_state = np.frombuffer(nodes_state[nid], dtype=np.int8)
            g_child = nodes_g[nid] + 1
            for a in range(N_MOVES):
                child = parent_state[MOVE_PERMS[a]]
                child_bytes = child.tobytes()
                prev = closed.get(child_bytes)
                if prev is not None and prev <= g_child:
                    continue
                closed[child_bytes] = g_child
                children.append(child)
                children_g.append(g_child)
                parent_ids.append(nid)
                actions.append(a)

        nodes_expanded += len(batch_ids)

        if not children:
            continue  # every child already has an at-least-as-good path on OPEN/CLOSED

        if len(nodes_state) + len(children) > max_nodes:
            return SolveResult(False, [], 0, nodes_expanded, len(nodes_state),
                               time.time() - t0, "max_nodes")

        # One net forward for all children in this iteration.
        children_arr = np.stack(children)
        with torch.no_grad():
            h_vals = net(torch.from_numpy(one_hot(children_arr)).to(device)).cpu().numpy()

        for i, child in enumerate(children):
            child_bytes = child.tobytes()
            # The goal always has h = 0 regardless of what the net thinks.
            h = 0.0 if child_bytes == _SOLVED_BYTES else float(h_vals[i])
            _push(child_bytes, children_g[i], parent_ids[i], actions[i], h)

    return SolveResult(False, [], 0, nodes_expanded, len(nodes_state),
                       time.time() - t0, "max_iterations")
