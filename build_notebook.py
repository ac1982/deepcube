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


md("""# DeepCubeA 训练

用**近似值迭代（AVI）**在线生成扰动状态，训练神经网络估计到还原态的距离。不需要任何外部数据集。

## 两档预设（第 5 格改 `PRESET`）

| 预设 | 迭代数 | 用途 |
|---|---|---|
| `smoke` | 1 000 | 验证代码跑通，约 5 分钟 |
| `paper` | 1 000 000 | 正式训练（RTX 5090 ~12h，T4/Colab 免费档跑不完） |

## 产物

- `ckpt_latest.pt` — 每 `ckpt_every` 步保存，断开后把 `RESUME_FROM = "ckpt_latest.pt"` 可续训
- `deepcube_cube3.pt` — 训练结束产出的最终权重，下载到本地放进 `checkpoints/`

## Colab 使用

`Runtime → Change runtime type → Hardware accelerator: GPU`，依次运行所有 cell。
""")

# ----------------------------------------------------------------------
md("""## 1. GPU 检查

没有 CUDA 就直接报错，告诉你去哪改 runtime。
""")

code("""import subprocess, shutil, sys

# `nvidia-smi` is only present on hosts with an NVIDIA driver installed.
# On Colab with the default CPU runtime it's missing — treat that as a
# clear signal that the user needs to switch runtimes.
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "-L"], check=False)
else:
    print("[setup] nvidia-smi not found on PATH — no NVIDIA driver on this host.")

import torch, numpy as np
print(f"torch={torch.__version__}  numpy={np.__version__}  cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "没有可用的 CUDA 设备。本 notebook 训练神经网络，必须有 GPU。\\n"
        "\\n"
        "  Colab:     Runtime -> Change runtime type -> Hardware accelerator: GPU\\n"
        "             然后 Runtime -> Reconnect，再重跑这一格。\\n"
        "  Vast.ai:   租实例时选带 GPU 的。\\n"
        "  本机:       在装了 NVIDIA 驱动 + CUDA 的机器上跑。\\n"
    )

print(f"device: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
try:
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"gpu matmul OK  (sum={y.sum().item():.2f})")
except Exception as e:
    print("gpu matmul FAILED:", e)
    print("If you see 'no kernel image available for sm_120' (RTX 5090 / Blackwell), run:")
    print("  !pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
    raise

# Ensure tqdm + matplotlib available (usually preinstalled, but Colab installs
# can drift between images).
try:
    import tqdm, matplotlib  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tqdm", "matplotlib"])
    import tqdm, matplotlib  # noqa: F401
""")

# ----------------------------------------------------------------------
md("""## 2. 魔方环境

54 格状态（0=U, 1=F, 2=R, 3=B, 4=L, 5=D）+ 12 个四分之一转。下面构建 12 个置换表。
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
    states = np.tile(SOLVED, (B, 1))
    moves = rng.integers(0, 12, size=(B, max_k))
    idx_b = np.arange(B)[:, None]
    for t in range(max_k):
        perms = MOVE_PERMS[moves[:, t]]            # (B, 54)
        new = states[idx_b, perms]                 # (B, 54)
        active = (t < ks)[:, None]
        states = np.where(active, new, states)
    return states, ks
""")

# ----------------------------------------------------------------------
md("""## 3. Sanity 测试

置换表正确性检查。任何断言失败就不要训练。
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
md("""## 4. 模型

残差 MLP：324 → 5000 → 1000 + 4× 残差块(1000) + 标量头。约 14.7M 参数。
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
md("""## 5. 配置

两档：`smoke` 先跑通、`paper` 正式训练。`RESUME_FROM = "ckpt_latest.pt"` 可续训。
""")

code("""PRESET = "smoke"          # "smoke" (~5 min, 验证) or "paper" (~12h on 5090, 正式)
RESUME_FROM = None        # 从 checkpoint 续训：填 "ckpt_latest.pt"

CONFIGS = {
    "smoke": dict(
        n_iters=1_000,
        batch_size=2_000,
        max_scramble=10,
        target_sync_every=500,
        ckpt_every=500,
        lr=1e-3,
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
md("""## 6. 训练（AVI）

每步：扰动 → target 网给所有 12 个子状态估代价 → `y = min_a(1 + h(s'))` → MSE 回归 → 周期同步 target。AMP 用 bf16（fp16 早期会溢出）。
""")

code("""import time
import matplotlib.pyplot as plt
from tqdm.auto import trange

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

loss_hist: list[float] = []
start_iter = 0
elapsed_before = 0.0   # wall-clock seconds accumulated in prior sessions (for resume)

if RESUME_FROM is not None:
    ck = torch.load(RESUME_FROM, map_location=device)
    net.load_state_dict(ck["net"])
    target_net.load_state_dict(ck["target"])
    opt.load_state_dict(ck["opt"])
    loss_hist = list(ck.get("loss_hist", []))
    start_iter = ck["iter"]
    elapsed_before = float(ck.get("elapsed", 0.0))
    print(f"resumed from {RESUME_FROM} at iter {start_iter}  "
          f"(loss_hist len={len(loss_hist)}, prior elapsed={elapsed_before / 60:.1f} min)")

# ---- compile online net only; target net is never compiled ----
# load_state_dict on a compiled module has surprising interactions with BN buffers,
# and the target forward is no_grad so the speedup is small.
try:
    net_c = torch.compile(net, mode="default")
    print("torch.compile enabled on online network")
except Exception as e:
    print(f"torch.compile skipped: {e}")
    net_c = net
target_c = target_net


def _scramble_batch_gpu(B: int, max_k: int) -> torch.Tensor:
    \"\"\"Generate B scrambled states with ks[i] ~ Uniform[1, max_k] random moves.

    Loop runs `max_k` times unconditionally (fixed bound, no GPU->CPU sync);
    samples with t >= ks are masked out via `torch.where`.\"\"\"
    ks = torch.randint(1, max_k + 1, (B,), device=device)
    states = SOLVED_T.unsqueeze(0).expand(B, 54).clone()
    moves = torch.randint(0, 12, (B, max_k), device=device)
    for t in range(max_k):
        perms = MOVE_PERMS_T[moves[:, t]]                    # (B, 54)
        new_states = states.gather(1, perms)                 # (B, 54)
        active = (t < ks).unsqueeze(1)
        states = torch.where(active, new_states, states)
    return states


def _compute_target(states: torch.Tensor) -> torch.Tensor:
    \"\"\"Given (B,54) states, return (B,) target cost-to-go.

    Two terminal cases, both set to 0:
      (a) child state is solved  -> h_target(s') = 0
      (b) current state is solved -> target(s) = 0 (occurs when a random
          scramble cancels, e.g. `U U'`; would otherwise teach h(goal) > 0).\"\"\"
    B = states.shape[0]
    next_states = states.unsqueeze(1).expand(B, 12, 54).gather(
        2, MOVE_PERMS_T.unsqueeze(0).expand(B, 12, 54)
    )
    next_flat = next_states.reshape(B * 12, 54)
    is_child_terminal = (next_flat == SOLVED_T).all(dim=-1)
    next_oh = F.one_hot(next_flat, num_classes=6).reshape(B * 12, 324).float()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h_next = target_c(next_oh).float()
    h_next = torch.where(is_child_terminal, torch.zeros_like(h_next), h_next)
    costs = (1.0 + h_next).reshape(B, 12)
    targets = costs.min(dim=1).values

    # (b) zero out target when current state is already solved.
    is_current_terminal = (states == SOLVED_T).all(dim=-1)
    targets = torch.where(is_current_terminal, torch.zeros_like(targets), targets)
    return targets


def _save_ckpt(path: str, it: int, elapsed_total: float) -> None:
    torch.save({
        "net": net.state_dict(),
        "target": target_net.state_dict(),
        "opt": opt.state_dict(),
        "iter": it,
        "cfg": cfg,
        "loss_hist": loss_hist,
        "elapsed": elapsed_total,
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
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = net_c(states_oh)
        loss = F.mse_loss(pred.float(), targets)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    loss_hist.append(float(loss.detach()))

    if (it + 1) % 50 == 0:
        pbar.set_postfix(loss=f"{np.mean(loss_hist[-50:]):.4f}")

    if (it + 1) % cfg["target_sync_every"] == 0:
        target_net.load_state_dict(net.state_dict())

    if (it + 1) % cfg["ckpt_every"] == 0:
        elapsed_total = elapsed_before + (time.time() - t0)
        _save_ckpt("ckpt_latest.pt", it + 1, elapsed_total)
        # Show loss without clearing output — tqdm bar stays put and plots stack.
        _plot_loss(f"iter {it + 1}/{cfg['n_iters']}  last-100 mean loss: {np.mean(loss_hist[-100:]):.4f}")
        print(f"saved ckpt_latest.pt  |  elapsed {elapsed_total / 60:.1f} min")

# Final plot + save
elapsed_total = elapsed_before + (time.time() - t0)
_plot_loss(f"training complete ({cfg['n_iters']} iters)")
torch.save({"net": net.state_dict(), "cfg": cfg, "loss_hist": loss_hist,
            "elapsed": elapsed_total}, "deepcube_cube3.pt")
print(f"saved deepcube_cube3.pt  ({elapsed_total / 60:.1f} min total)")
""")

# ----------------------------------------------------------------------
md("""## 7. 端到端验证：贪心求解

用贪心策略（每步选 `h` 最小的动作）在浅扰动上跑一下，验证网络学到了有效的启发式。完整的 BWAS 求解器在本地推理栈里，这里只是 sanity check。
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

## 完成

- 最终权重在 `deepcube_cube3.pt`
- Jupyter 文件栏右键 → **Download** 下载到本地
- 放到仓库的 `checkpoints/deepcube_cube3.pt`，推理服务器会自动加载
- 可以销毁 Colab / Vast.ai 实例了
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
