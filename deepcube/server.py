"""FastAPI server exposing the DeepCubeA solver over HTTP.

Endpoints
---------
GET  /          -> serves static/index.html (the 3D frontend)
GET  /healthz   -> {"ok": bool, "model_loaded": bool}
POST /scramble  -> {state: [int x 54], moves: [str]}
POST /solve     -> {solved, moves, path_length, nodes_*, elapsed_sec, stop_reason}

Model loading
-------------
At startup, tries to load a trained checkpoint from the path in the
`DEEPCUBE_CHECKPOINT` env var (default: `checkpoints/deepcube_cube3.pt`).
If it's missing, the server still starts so the UI is developable during
training — `/solve` then returns 503 until a checkpoint is dropped in.

Thread safety
-------------
Uvicorn runs sync endpoints in a thread pool. `_SOLVE_LOCK` serializes
`bwas_solve` calls because the solver holds mutable search state and the
same `nn.Module` runs inference concurrently otherwise.
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deepcube.cube3 import (
    CENTER_INDICES,
    MOVE_PERMS,
    MOVES,
    N_MOVES,
    N_STICKERS,
    SOLVED,
    apply_move,
    scramble,
    sticker_layout,
)
from deepcube.model import DeepCubeANet, load_checkpoint
from deepcube.search import bwas_solve
from deepcube.solver_kociemba import kociemba_solve

__all__ = ["app", "main"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STATIC_DIR = _REPO_ROOT / "static"
_DEFAULT_CKPT = _REPO_ROOT / "checkpoints" / "deepcube_cube3.pt"

_net: DeepCubeANet | None = None
_SOLVE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

State54 = Annotated[list[int], Field(min_length=N_STICKERS, max_length=N_STICKERS)]


class ScrambleRequest(BaseModel):
    depth: int = Field(default=20, ge=1, le=50)
    seed: int | None = None


class ScrambleResponse(BaseModel):
    state: State54
    moves: list[str]


class SolveRequest(BaseModel):
    state: State54
    solver: Literal["deepcube", "kociemba"] = "kociemba"
    # BWAS-only knobs; ignored for kociemba.
    lambda_weight: float = Field(default=2.0, ge=1.0, le=10.0)
    batch_size: int = Field(default=1000, ge=1, le=10_000)
    max_nodes: int = Field(default=1_000_000, ge=1_000, le=20_000_000)
    max_iterations: int = Field(default=2_000, ge=1, le=100_000)


class CompareEntry(BaseModel):
    """The *other* solver's result on the same state, included in SolveResponse
    when the primary solver is DeepCubeA so the UI can show a side-by-side
    comparison. Kociemba is always cheap (ms), so we always run it as the
    compare baseline. DeepCubeA is slow (up to ~minute) — we only include it
    as compare if a model is loaded; otherwise this field is None."""
    solver: str
    solved: bool
    moves: list[str]                 # full path, so UI can fall back to it when primary fails
    path_length: int
    path_length_htm: int | None
    elapsed_sec: float
    stop_reason: str


class SolveResponse(BaseModel):
    solver: str
    solved: bool
    moves: list[str]
    path_length: int                 # quarter-turn count — always meaningful
    path_length_htm: int | None      # half-turn count — only for kociemba
    nodes_expanded: int
    nodes_generated: int
    elapsed_sec: float
    stop_reason: str
    compare: CompareEntry | None = None


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    checkpoint_path: str | None


class MetaResponse(BaseModel):
    """Enough env info for the frontend to avoid duplicating cube3.py in JS."""
    n_stickers: int
    n_moves: int
    moves: list[str]
    move_perms: list[list[int]]
    solved_state: list[int]
    # For edit-mode raycast -> sticker-index lookup. sticker_positions[i] and
    # sticker_normals[i] are the (cubie_position, outward_normal) pair of
    # sticker i, each a list of 3 ints in {-1, 0, 1}.
    sticker_positions: list[list[int]]
    sticker_normals: list[list[int]]
    center_indices: list[int]


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def _try_load_checkpoint() -> tuple[DeepCubeANet | None, Path | None]:
    path = Path(os.getenv("DEEPCUBE_CHECKPOINT", _DEFAULT_CKPT))
    if not path.exists():
        return None, path
    net, _ = load_checkpoint(path, device="cpu")
    return net, path


def _reset_state_for_tests() -> None:
    """Hook for tests: reload the checkpoint and clear the lock."""
    global _net
    _net, _ = _try_load_checkpoint()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _net
    _net, path = _try_load_checkpoint()
    if _net is None:
        print(f"[deepcube.server] no checkpoint at {path}; /solve will return 503 until one is provided")
    else:
        print(f"[deepcube.server] loaded checkpoint from {path}")
    yield
    # nothing to tear down


app = FastAPI(title="DeepCubeA", version="0.1.0", lifespan=_lifespan)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    path = os.getenv("DEEPCUBE_CHECKPOINT", str(_DEFAULT_CKPT))
    return HealthResponse(ok=True, model_loaded=_net is not None, checkpoint_path=path)


@app.get("/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    layout = sticker_layout()
    return MetaResponse(
        n_stickers=N_STICKERS,
        n_moves=N_MOVES,
        moves=MOVES,
        move_perms=MOVE_PERMS.tolist(),
        solved_state=SOLVED.tolist(),
        sticker_positions=[list(pos) for pos, _ in layout],
        sticker_normals=[list(nor) for _, nor in layout],
        center_indices=list(CENTER_INDICES),
    )


@app.post("/scramble", response_model=ScrambleResponse)
def post_scramble(req: ScrambleRequest) -> ScrambleResponse:
    rng = np.random.default_rng(req.seed)
    state, move_indices = scramble(req.depth, rng)
    return ScrambleResponse(
        state=state.tolist(),
        moves=[MOVES[int(m)] for m in move_indices],
    )


@app.post("/solve", response_model=SolveResponse)
def post_solve(req: SolveRequest) -> SolveResponse:
    state = np.array(req.state, dtype=np.int8)
    if state.shape != (N_STICKERS,):
        raise HTTPException(status_code=400, detail=f"state must be {N_STICKERS} ints")
    if state.min() < 0 or state.max() > 5:
        raise HTTPException(status_code=400, detail="state values must be in 0..5")

    compare: CompareEntry | None = None

    if req.solver == "kociemba":
        res = kociemba_solve(state)
    else:
        if _net is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="deepcube model not loaded — drop a trained checkpoint into "
                       "checkpoints/deepcube_cube3.pt (or set DEEPCUBE_CHECKPOINT) and restart, "
                       "or pass solver=\"kociemba\" to use the two-phase solver.",
            )
        with _SOLVE_LOCK:
            res = bwas_solve(
                _net, state,
                lambda_weight=req.lambda_weight,
                batch_size=req.batch_size,
                max_iterations=req.max_iterations,
                max_nodes=req.max_nodes,
                device="cpu",
            )
        # Always also run Kociemba for the comparison card. It's a few
        # milliseconds, so no meaningful latency cost.
        k_res = kociemba_solve(state)
        compare = CompareEntry(
            solver="kociemba",
            solved=k_res.solved,
            moves=[MOVES[i] for i in k_res.path],
            path_length=k_res.path_length,
            path_length_htm=k_res.path_length_htm,
            elapsed_sec=k_res.elapsed_sec,
            stop_reason=k_res.stop_reason,
        )

    return SolveResponse(
        solver=req.solver,
        solved=res.solved,
        moves=[MOVES[i] for i in res.path],
        path_length=res.path_length,
        path_length_htm=res.path_length_htm,
        nodes_expanded=res.nodes_expanded,
        nodes_generated=res.nodes_generated,
        elapsed_sec=res.elapsed_sec,
        stop_reason=res.stop_reason,
        compare=compare,
    )


# Static frontend: index.html at "/", assets mounted under "/static".
@app.get("/")
def root(request: Request) -> FileResponse:
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="frontend not built yet (static/index.html missing)")
    return FileResponse(index)


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def main() -> None:
    """Entry point for the `deepcube-serve` console script."""
    import uvicorn

    host = os.getenv("DEEPCUBE_HOST", "127.0.0.1")
    port = int(os.getenv("DEEPCUBE_PORT", "8000"))
    uvicorn.run("deepcube.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
