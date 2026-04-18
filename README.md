# DeepCubeA — PyTorch reimplementation

A clean-room reimplementation of **DeepCubeA** (Agostinelli et al., *Nature Machine Intelligence* 2019) for the 3×3 Rubik's cube, in modern PyTorch + Python 3.12. Reference: [yeates/deepcube-full](https://github.com/yeates/deepcube-full) (original, TF1/Keras era).

The project is split into two sides:

| side | machine | purpose |
|---|---|---|
| **training** | rented GPU (Vast.ai 5090) | run `train.ipynb` → produce `deepcube_cube3.pt` |
| **inference + UI** | local | FastAPI backend + cuber.js 3D frontend, loads `.pt` to solve scrambles interactively |

The two sides share a common cube environment (54-sticker state, 12 quarter-turn moves) but otherwise communicate only through the trained checkpoint.

---

## Algorithm

**Approximate Value Iteration (AVI).** The heuristic network `h_θ(s)` is trained to estimate the cost-to-go from state `s` to the solved state. No external dataset is needed — training states are generated on the fly by applying random quarter-turns to the solved cube.

At each iteration:

1. Sample a batch of scrambled states `s`, each from `k ~ Uniform[1, K]` random moves.
2. For every `s`, expand all 12 children `s'` through the target network and compute
   `y(s) = min_a (1 + h_θ⁻(s'))`, where `h = 0` at the goal.
3. Minimise `MSE(h_θ(s), y(s))` on the online network.
4. Periodically sync `θ → θ⁻`.

Solving (inference) uses **Batch-Weighted A\*** with `h_θ` as the heuristic.

## Architecture

Residual MLP (per paper):

- Input: one-hot 54 stickers × 6 colors = **324** dim
- FC 324 → 5000 → 1000 (BatchNorm + ReLU)
- 4 residual blocks, each 1000 → 1000 → 1000 with skip
- Scalar head
- ~14.7 M parameters

## State representation

The cube is laid out as 54 stickers (0 = U, 1 = F, 2 = R, 3 = B, 4 = L, 5 = D) in standard net order:

```
        U0 U1 U2
        U3 U4 U5
        U6 U7 U8
L0..L8  F0..F8  R0..R8  B0..B8
        D0 D1 D2
        D3 D4 D5
        D6 D7 D8
```

The 12 quarter-turn moves (`U, U', D, D', F, F', B, B', R, R', L, L'`) are compiled into 54-element gather permutations by rotating cubies in 3-space and re-mapping (position, sticker-normal) → sticker index. See cell 2 of `train.ipynb`.

## Training

### On Vast.ai (RTX 5090, recommended)

1. Rent a 1× RTX 5090 instance with the `vastai/pytorch` template (on-demand).
2. Upload `train.ipynb` to Jupyter Lab.
3. Run all cells top-to-bottom.
4. Download `deepcube_cube3.pt` before destroying the instance.

**Presets** (switch in cell 10 by changing `PRESET`):

| preset | iters | batch | max scramble | ~ time on 5090 | ~ cost @ $0.30/h |
|---|---|---|---|---|---|
| `smoke` | 1 000 | 2 000 | 10 | ~5 min | ~$0.03 |
| `demo` | 50 000 | 5 000 | 20 | ~1–2 h | ~$0.50 |
| `paper` | 1 000 000 | 10 000 | 30 | ~12 h | ~$3.60 |

The notebook writes `ckpt_latest.pt` every `ckpt_every` iterations and plots an inline loss curve. To resume after a disconnect, set `RESUME_FROM = "ckpt_latest.pt"` in cell 10.

### Blackwell (sm_120) note

RTX 5090 is Blackwell and requires PyTorch **2.7+**. If you hit `no kernel image available for sm_120`, cell 2 will print the fallback install command (PyTorch nightly + cu128).

## Inference + UI (coming)

Planned (not yet implemented):

```
deepcube/
├── cube3.py       # same env as the notebook, promoted to a module
├── model.py       # same architecture as the notebook
├── search.py      # Batch-Weighted A* solver
└── server.py      # FastAPI: POST /scramble, POST /solve
static/
└── index.html     # cuber.js 3D cube, drives the backend
checkpoints/
└── deepcube_cube3.pt   # drop the trained file here
```

Run locally with `uv run uvicorn deepcube.server:app` → open `http://127.0.0.1:8000`.

## Project status

- [x] Cube environment with verified moves (sanity tests pass)
- [x] PyTorch model (14.66 M params)
- [x] AVI training loop with AMP, `torch.compile`, checkpointing, resume
- [x] Greedy-rollout sanity check
- [ ] Batch-Weighted A\* solver
- [ ] FastAPI backend
- [ ] cuber.js 3D frontend

## Repo layout

```
.
├── README.md
├── train.ipynb           # upload this to Vast.ai
├── build_notebook.py     # regenerate train.ipynb from source
└── .gitignore
```

## References

- Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). *Solving the Rubik's cube with deep reinforcement learning and search*. Nature Machine Intelligence, 1(8), 356–363.
- [yeates/deepcube-full](https://github.com/yeates/deepcube-full) — original TF1/Keras implementation + web demo.
