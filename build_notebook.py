"""Generate train.ipynb for DeepCubeA training on Vast.ai 5090."""
import json
from pathlib import Path

cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


md("""# DeepCubeA — 3×3 Rubik's Cube (PyTorch)

Trains a neural heuristic for Rubik's cube via **Approximate Value Iteration (AVI)** on randomly scrambled states. No external dataset — states are generated on the fly from the solved cube.

## Quick start on Vast.ai
1. Rent an instance with template `vastai/pytorch` (latest), 1× RTX 5090, on-demand.
2. Upload this notebook to the Jupyter Lab file browser.
3. Run all cells top-to-bottom.
4. When done, download `deepcube_cube3.pt` before destroying the instance.

## Expected runtime on single RTX 5090
| preset | iters | approx time | cost @ $0.30/hr |
|---|---|---|---|
| `smoke` | 1,000 | ~5 min | ~$0.03 |
| `demo` | 50,000 | ~1–2 h | ~$0.50 |
| `paper` | 1,000,000 | ~12 h | ~$3.60 |

## Outputs
- `ckpt_latest.pt` — saved every `ckpt_every` iterations (resume with `RESUME_FROM`).
- `deepcube_cube3.pt` — final model weights (this is what you download).
- Inline loss curves at each checkpoint.
""")

# ----------------------------------------------------------------------
md("""## 1. Setup & GPU check

Verifies CUDA, checks that Blackwell kernels work. If the smoke matmul fails on a 5090 with a "no kernel image" error, install PyTorch nightly (cell will print the command).
""")

code("""import subprocess, sys
subprocess.run(["nvidia-smi", "-L"], check=False)

import torch, numpy as np
print(f"torch={torch.__version__}  numpy={np.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
    try:
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"gpu matmul OK  (sum={y.sum().item():.2f})")
    except Exception as e:
        print("gpu matmul FAILED:", e)
        print("If you see 'no kernel image available for sm_120', run:")
        print("  !pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
        raise
else:
    raise RuntimeError("No CUDA device — this notebook is GPU-only.")

# Ensure tqdm + matplotlib available (usually preinstalled in vastai/pytorch)
try:
    import tqdm, matplotlib  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tqdm", "matplotlib"])
    import tqdm, matplotlib  # noqa: F401
""")

# ----------------------------------------------------------------------
md("""## 2. Cube environment

**State:** 54 stickers, each holding a color in `0..5` (U=0, F=1, R=2, B=3, L=4, D=5).

**Index layout** (standard net):

```
        U0 U1 U2
        U3 U4 U5
        U6 U7 U8
L0..L8  F0..F8  R0..R8  B0..B8
        D0 D1 D2
        D3 D4 D5
        D6 D7 D8
```

U=0..8, F=9..17, R=18..26, B=27..35, L=36..44, D=45..53.

**Moves:** 12 quarter-turns `U,U',D,D',F,F',B,B',R,R',L,L'` (ordered in pairs so `move^1` gives inverse).

Each move is a 54-permutation built algorithmically by rotating the affected cubies in 3-space and re-mapping (cubie-position, sticker-normal) back to sticker index.
""")

code("""import numpy as np

# Solved state: 9 stickers per face, color = face index 0..5
SOLVED = np.repeat(np.arange(6, dtype=np.int8), 9)
assert SOLVED.shape == (54,)

# ---- sticker_index -> (cubie_position, sticker_normal) ----
# coords in {-1, 0, 1}^3; normal is a unit vector like (0,1,0) for U face.

def _build_sticker_info():
    info = []
    # U (+y): iterate z (net top→bottom = back→front = -1→1), then x (-1→1)
    for z in (-1, 0, 1):
        for x in (-1, 0, 1):
            info.append(((x, 1, z), (0, 1, 0)))
    # F (+z): y (top→bottom = 1→-1), x (-1→1)
    for y in (1, 0, -1):
        for x in (-1, 0, 1):
            info.append(((x, y, 1), (0, 0, 1)))
    # R (+x): y (1→-1), z (left→right in net of R = 1→-1)
    for y in (1, 0, -1):
        for z in (1, 0, -1):
            info.append(((1, y, z), (1, 0, 0)))
    # B (-z): y (1→-1), x (1→-1)  (B's net is to the right of R; going right in net = -x in world)
    for y in (1, 0, -1):
        for x in (1, 0, -1):
            info.append(((x, y, -1), (0, 0, -1)))
    # L (-x): y (1→-1), z (-1→1)  (L's net is to the left of F; going right in net = +z in world)
    for y in (1, 0, -1):
        for z in (-1, 0, 1):
            info.append(((-1, y, z), (-1, 0, 0)))
    # D (-y): z (top of D net = shared with F bottom = +z; 1→-1), x (-1→1)
    for z in (1, 0, -1):
        for x in (-1, 0, 1):
            info.append(((x, -1, z), (0, -1, 0)))
    return info

STICKER_INFO = _build_sticker_info()
assert len(STICKER_INFO) == 54
POS_NORMAL_TO_IDX = {key: i for i, key in enumerate(STICKER_INFO)}

def _rot(axis: int, sign: int) -> np.ndarray:
    \"\"\"90° rotation matrix about axis (0=x,1=y,2=z). sign=-1 is CW (right-hand rule).\"\"\"
    c, s = 0, sign  # cos(90°)=0, sin(±90°)=±1
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=int)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=int)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=int)

# Face-turn CW (viewed from outside the face) is -90° about the outward-normal axis.
# Inverse (') is +90°.
MOVE_SPECS = {
    "U":  (1, +1, 1, -1),   # layer y=+1, rotate about +y by -90°
    "U'": (1, +1, 1, +1),
    "D":  (1, -1, 1, +1),   # layer y=-1, CW from -y = +90° about +y
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
MOVES = list(MOVE_SPECS.keys())  # pairs: (U,U'), (D,D'), ... inverse(m) = m ^ 1

def _build_perm(layer_axis: int, layer_value: int, rot_axis: int, sign: int) -> np.ndarray:
    R = _rot(rot_axis, sign)
    perm = np.arange(54, dtype=np.int64)
    for old_idx, (pos, normal) in enumerate(STICKER_INFO):
        if pos[layer_axis] != layer_value:
            continue
        new_pos = tuple(int(v) for v in R @ np.array(pos))
        new_normal = tuple(int(v) for v in R @ np.array(normal))
        new_idx = POS_NORMAL_TO_IDX[(new_pos, new_normal)]
        perm[new_idx] = old_idx
    return perm

MOVE_PERMS = np.stack([_build_perm(*MOVE_SPECS[m]) for m in MOVES])  # (12, 54)
print(f"built {len(MOVES)} move permutations, shape={MOVE_PERMS.shape}")

def apply_move(state: np.ndarray, move_idx: int) -> np.ndarray:
    return state[MOVE_PERMS[move_idx]]

def scramble(k: int, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    mvs = rng.integers(0, 12, size=k)
    s = SOLVED.copy()
    for m in mvs:
        s = apply_move(s, int(m))
    return s, mvs

def batch_scramble(B: int, max_k: int, rng):
    \"\"\"Generate B states with ks[i] ~ Uniform[1, max_k] random moves.

    Returns (states: (B,54) int8, ks: (B,) int64).\"\"\"
    ks = rng.integers(1, max_k + 1, size=B)
    max_total = int(ks.max())
    states = np.tile(SOLVED, (B, 1))
    moves = rng.integers(0, 12, size=(B, max_total))
    idx_b = np.arange(B)[:, None]
    for t in range(max_total):
        perms = MOVE_PERMS[moves[:, t]]            # (B, 54)
        new = states[idx_b, perms]                 # (B, 54)
        active = (t < ks)[:, None]
        states = np.where(active, new, states)
    return states, ks
""")

# ----------------------------------------------------------------------
md("""## 3. Sanity tests

If any assertion fails, the env has a bug — don't train on it.
""")

code("""def _run_env_tests():
    rng = np.random.default_rng(0)

    # Every face turn has order 4.
    for m in range(12):
        s = SOLVED.copy()
        for _ in range(4):
            s = apply_move(s, m)
        assert (s == SOLVED).all(), f"{MOVES[m]}^4 should be identity"

    # Each move and its inverse cancel.
    for m in range(12):
        s = apply_move(SOLVED, m)
        assert not (s == SOLVED).all(), f"{MOVES[m]} must change state"
        s = apply_move(s, m ^ 1)
        assert (s == SOLVED).all(), f"{MOVES[m]} · {MOVES[m ^ 1]} should be identity"

    # The 'sexy move' (R U R' U') has order 6.
    sexy = [MOVES.index(x) for x in ("R", "U", "R'", "U'")]
    s = SOLVED.copy()
    for _ in range(6):
        for m in sexy:
            s = apply_move(s, m)
    assert (s == SOLVED).all(), "(R U R' U')^6 should be identity"

    # Random scramble + reverse -> solved.
    for _ in range(50):
        k = int(rng.integers(1, 30))
        mvs = rng.integers(0, 12, size=k)
        s = SOLVED.copy()
        for m in mvs:
            s = apply_move(s, int(m))
        for m in mvs[::-1]:
            s = apply_move(s, int(m) ^ 1)
        assert (s == SOLVED).all(), "scramble + reverse failed"

    # batch_scramble shape.
    bs, ks = batch_scramble(32, 20, rng)
    assert bs.shape == (32, 54) and ks.shape == (32,)
    assert bs.min() >= 0 and bs.max() <= 5

    print("[OK] all env sanity tests passed")

_run_env_tests()
""")

# ----------------------------------------------------------------------
md("""## 4. Model

Residual MLP per the DeepCubeA paper (Agostinelli et al., 2019):

- Input: one-hot 54×6 = **324** dim.
- FC 324 → 5000 → 1000 (BatchNorm + ReLU after each).
- 4 residual blocks, each 1000 → 1000 → 1000 with a skip connection.
- Scalar output head (estimated cost-to-go).

~10M parameters.
""")

code("""import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim: int = 1000):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)), inplace=True)
        h = self.bn2(self.fc2(h))
        return F.relu(x + h, inplace=True)


class DeepCubeANet(nn.Module):
    def __init__(self, n_res_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(324, 5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(inplace=True),
            nn.Linear(5000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResBlock(1000) for _ in range(n_res_blocks)])
        self.head = nn.Linear(1000, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)


_tmp = DeepCubeANet()
n_params = sum(p.numel() for p in _tmp.parameters())
print(f"model params: {n_params / 1e6:.2f}M")
del _tmp
""")

# ----------------------------------------------------------------------
md("""## 5. Config

Three presets. **Change `PRESET` here** if you want to swap.

- `smoke` (~5 min): verifies the full pipeline works. Trains a weak heuristic on scrambles ≤10.
- `demo` (~1–2 h): decent heuristic for common scrambles.
- `paper` (~12 h): full paper configuration on 1M iterations.

To resume from a crash / disconnect, set `RESUME_FROM = "ckpt_latest.pt"`.
""")

code("""PRESET = "paper"          # "smoke" | "demo" | "paper"
RESUME_FROM = None        # e.g. "ckpt_latest.pt" to continue training

CONFIGS = {
    "smoke": dict(
        n_iters=1_000,
        batch_size=2_000,
        max_scramble=10,
        target_sync_every=500,
        ckpt_every=500,
        lr=1e-3,
    ),
    "demo": dict(
        n_iters=50_000,
        batch_size=5_000,
        max_scramble=20,
        target_sync_every=2_500,
        ckpt_every=5_000,
        lr=5e-4,
    ),
    "paper": dict(
        n_iters=1_000_000,
        batch_size=10_000,
        max_scramble=30,
        target_sync_every=5_000,
        ckpt_every=10_000,
        lr=1e-4,
    ),
}

cfg = CONFIGS[PRESET]
print(f"preset={PRESET}")
for k, v in cfg.items():
    print(f"  {k}: {v}")
""")

# ----------------------------------------------------------------------
md("""## 6. Training loop (AVI)

**Approximate Value Iteration:**

1. Sample batch of scrambled states `s ~ scramble(k), k ~ Uniform[1, K]`.
2. For each `s`, expand all 12 children `s'` and compute target:
   $$y(s) = \\min_a \\Big(1 + h_{\\theta^-}(s')\\Big)$$
   where `h_{θ⁻}` is the target network. If `s'` is solved, `h=0`.
3. Minimize `MSE(h_θ(s), y(s))` wrt online network `θ`.
4. Every `target_sync_every` iters, copy `θ → θ⁻`.

Uses AMP (fp16) + `torch.compile` for speed. Training keeps `ckpt_latest.pt` up to date and plots the loss curve inline at each checkpoint.
""")

code("""import time
import matplotlib.pyplot as plt
from tqdm.auto import trange
from IPython.display import clear_output

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

SOLVED_T = torch.from_numpy(SOLVED).long().to(device)            # (54,)
MOVE_PERMS_T = torch.from_numpy(MOVE_PERMS).long().to(device)     # (12, 54)

# ---- networks ----
net = DeepCubeANet().to(device)
target_net = DeepCubeANet().to(device)
target_net.load_state_dict(net.state_dict())
target_net.eval()
for p in target_net.parameters():
    p.requires_grad = False

opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
scaler = torch.amp.GradScaler("cuda")

loss_hist: list[float] = []
start_iter = 0

if RESUME_FROM is not None:
    ck = torch.load(RESUME_FROM, map_location=device)
    net.load_state_dict(ck["net"])
    target_net.load_state_dict(ck["target"])
    opt.load_state_dict(ck["opt"])
    loss_hist = list(ck.get("loss_hist", []))
    start_iter = ck["iter"]
    print(f"resumed from {RESUME_FROM} at iter {start_iter}  (loss_hist len={len(loss_hist)})")

# ---- optional torch.compile ----
try:
    net_c = torch.compile(net, mode="default")
    target_c = torch.compile(target_net, mode="default")
    print("torch.compile enabled")
except Exception as e:
    print(f"torch.compile skipped: {e}")
    net_c, target_c = net, target_net


def _scramble_batch_gpu(B: int, max_k: int) -> torch.Tensor:
    ks = torch.randint(1, max_k + 1, (B,), device=device)
    max_total = int(ks.max().item())
    states = SOLVED_T.unsqueeze(0).expand(B, 54).clone()
    moves = torch.randint(0, 12, (B, max_total), device=device)
    for t in range(max_total):
        perms = MOVE_PERMS_T[moves[:, t]]                    # (B, 54)
        new_states = states.gather(1, perms)                 # (B, 54)
        active = (t < ks).unsqueeze(1)
        states = torch.where(active, new_states, states)
    return states


def _compute_target(states: torch.Tensor) -> torch.Tensor:
    \"\"\"Given (B,54) states, return (B,) target cost-to-go from target net.\"\"\"
    B = states.shape[0]
    # Expand to all 12 children: (B, 12, 54)
    next_states = states.unsqueeze(1).expand(B, 12, 54).gather(
        2, MOVE_PERMS_T.unsqueeze(0).expand(B, 12, 54)
    )
    next_flat = next_states.reshape(B * 12, 54)
    is_terminal = (next_flat == SOLVED_T).all(dim=-1)
    next_oh = F.one_hot(next_flat, num_classes=6).reshape(B * 12, 324).float()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        h_next = target_c(next_oh)
    h_next = torch.where(is_terminal, torch.zeros_like(h_next), h_next)
    costs = (1.0 + h_next).reshape(B, 12)
    targets = costs.min(dim=1).values
    return targets


def _save_ckpt(path: str, it: int) -> None:
    torch.save({
        "net": net.state_dict(),
        "target": target_net.state_dict(),
        "opt": opt.state_dict(),
        "iter": it,
        "cfg": cfg,
        "loss_hist": loss_hist,
    }, path)


def _plot_loss(title: str) -> None:
    if not loss_hist:
        return
    plt.figure(figsize=(10, 3.5))
    plt.plot(loss_hist, linewidth=0.6, alpha=0.6, label="per-iter")
    w = max(1, len(loss_hist) // 200)
    if len(loss_hist) > w:
        smooth = np.convolve(loss_hist, np.ones(w) / w, mode="valid")
        plt.plot(range(w - 1, len(loss_hist)), smooth, linewidth=1.5, label=f"MA-{w}")
    plt.yscale("log")
    plt.xlabel("iter")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---- main loop ----
pbar = trange(start_iter, cfg["n_iters"], initial=start_iter, total=cfg["n_iters"])
t0 = time.time()
for it in pbar:
    states = _scramble_batch_gpu(cfg["batch_size"], cfg["max_scramble"])

    with torch.no_grad():
        targets = _compute_target(states)

    states_oh = F.one_hot(states, num_classes=6).reshape(states.shape[0], 324).float()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = net_c(states_oh)
        loss = F.mse_loss(pred, targets.float())

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    loss_hist.append(float(loss.detach()))

    if (it + 1) % 50 == 0:
        pbar.set_postfix(loss=f"{np.mean(loss_hist[-50:]):.4f}")

    if (it + 1) % cfg["target_sync_every"] == 0:
        target_net.load_state_dict(net.state_dict())

    if (it + 1) % cfg["ckpt_every"] == 0:
        _save_ckpt("ckpt_latest.pt", it + 1)
        clear_output(wait=True)
        _plot_loss(f"iter {it + 1}/{cfg['n_iters']}  last-100 mean loss: {np.mean(loss_hist[-100:]):.4f}")
        print(f"saved ckpt_latest.pt  |  elapsed {(time.time() - t0) / 60:.1f} min")

# Final plot + save
_plot_loss(f"training complete ({cfg['n_iters']} iters)")
torch.save({"net": net.state_dict(), "cfg": cfg, "loss_hist": loss_hist}, "deepcube_cube3.pt")
print(f"saved deepcube_cube3.pt  ({(time.time() - t0) / 60:.1f} min total)")
""")

# ----------------------------------------------------------------------
md("""## 7. End-to-end sanity: greedy rollout

Loads the trained net and greedily picks the action with lowest estimated cost-to-go. Works for short scrambles even with a weak heuristic — if this fails on 3-step scrambles, training didn't converge.

**Note:** full-strength solving uses **Batch-Weighted A\\*** (BWAS), which is in the local inference repo, not here — this is only a sanity check.
""")

code("""@torch.no_grad()
def greedy_solve(net_module: nn.Module, state_np: np.ndarray, max_steps: int = 50) -> list[str] | None:
    net_module.eval()
    s = torch.from_numpy(state_np).long().to(device)
    path: list[str] = []
    for _ in range(max_steps):
        if torch.equal(s, SOLVED_T):
            return path
        nexts = s[MOVE_PERMS_T]                                         # (12, 54)
        oh = F.one_hot(nexts, num_classes=6).reshape(12, 324).float()
        h = net_module(oh)
        terms = (nexts == SOLVED_T).all(dim=-1)
        h = torch.where(terms, torch.zeros_like(h), h)
        costs = 1.0 + h
        best = int(costs.argmin().item())
        path.append(MOVES[best])
        s = nexts[best]
    return None


rng = np.random.default_rng(123)
print(f"{'scramble_len':>12} {'solved_in':>10}  path")
print("-" * 50)
successes = 0
for depth in (1, 2, 3, 5, 7, 10):
    for trial in range(3):
        s = SOLVED.copy()
        mvs = rng.integers(0, 12, size=depth)
        for m in mvs:
            s = apply_move(s, int(m))
        sol = greedy_solve(net, s, max_steps=max(50, depth * 4))
        if sol is None:
            print(f"{depth:>12} {'FAIL':>10}")
        else:
            successes += 1
            print(f"{depth:>12} {len(sol):>10}  {' '.join(sol)}")
print(f"\\n{successes}/18 greedy solves succeeded")
""")

# ----------------------------------------------------------------------
md("""---

## Done

- Final weights saved at `deepcube_cube3.pt`.
- Right-click in the Jupyter file browser → **Download** to pull it to your local machine.
- Put it in the local repo at `checkpoints/deepcube_cube3.pt` — the inference server will pick it up from there.
- You can now destroy the Vast.ai instance.
""")

# ----------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "train.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
