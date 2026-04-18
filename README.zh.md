# DeepCubeA — PyTorch 重写版

> 🌐 [English](README.md) · 简体中文

一个从零重写的 **DeepCubeA** 实现（Agostinelli 等，*Nature Machine Intelligence* 2019），专门求解 3×3 魔方。用的是现代 PyTorch + Python 3.12。参考原实现：[yeates/deepcube-full](https://github.com/yeates/deepcube-full)（TF1/Keras 时代的）。

项目分两块：

| 部分 | 机器 | 目的 |
|---|---|---|
| **训练** | 租用 GPU (Vast.ai 5090) | 跑 `train.ipynb` → 产出 `deepcube_cube3.pt` |
| **推理 + UI** | 本机 | FastAPI 后端 + three.js 3D 前端，加载 `.pt` 实时求解 |

两边共用同一套魔方环境（54 格状态、12 个四分之一转动作），除此之外只通过训练好的权重文件通信。

---

## 算法

**近似值迭代（Approximate Value Iteration, AVI）**。启发式网络 `h_θ(s)` 学习估计从状态 `s` 到 solved 状态所需的代价。**不需要任何外部数据集**——训练状态全部在线生成，从 solved 魔方随机应用若干四分之一转即可。

每次迭代：

1. 采样一个 batch 的扰动状态 `s`，每个用 `k ~ Uniform[1, K]` 步随机走
2. 对每个 `s`，通过 target 网络展开全部 12 个子状态 `s'`，计算
   `y(s) = min_a (1 + h_θ⁻(s'))`，其中 goal 处 `h = 0`
3. 在 online 网络上最小化 `MSE(h_θ(s), y(s))`
4. 周期性把 `θ → θ⁻` 同步

求解阶段用 **Batch-Weighted A\***，`h_θ` 当启发式。

UI 里另外集成了 **Kociemba 两阶段算法**（1992，经典的近最优魔方求解器）作为第二个后端。这样在 `.pt` 还没训好之前 app 已经可用，训好之后还可以横向对比。进入 Solve 面板的求解器 radio 切换。

## 网络结构

残差 MLP（与论文一致）：

- 输入：54 贴纸 × 6 色 one-hot = **324** 维
- FC 324 → 5000 → 1000（每层后接 BatchNorm + ReLU）
- 4 个残差块，每块 1000 → 1000 → 1000 带 skip
- 标量输出头
- 约 **14.7M 参数**

## 状态表示

54 个贴纸的扁平数组，每位存 0..5 的颜色索引（0=U 上, 1=F 前, 2=R 右, 3=B 后, 4=L 左, 5=D 下），标准展开图顺序：

```
        U0 U1 U2
        U3 U4 U5
        U6 U7 U8
L0..L8  F0..F8  R0..R8  B0..B8
        D0 D1 D2
        D3 D4 D5
        D6 D7 D8
```

12 个四分之一转（`U, U', D, D', F, F', B, B', R, R', L, L'`）通过在 3D 空间里旋转小块并把 (位置, 贴纸法向) → 贴纸索引 反推得到，编译成 54 元素的 gather 置换。详见 `train.ipynb` 第 2 格。

## 训练

### Vast.ai 上跑（推荐 RTX 5090）

1. 租一台 1× RTX 5090 实例，选 `vastai/pytorch` 模板（on-demand）
2. 上传 `train.ipynb` 到 Jupyter Lab
3. 从上到下依次运行所有 cell
4. 销毁实例前下载 `deepcube_cube3.pt`

**预设档位**（改第 10 格的 `PRESET` 切换）：

| 档位 | 迭代数 | batch | 最大扰动 | 5090 上耗时 | 估算成本 @ $0.30/小时 |
|---|---|---|---|---|---|
| `smoke` | 1 000 | 2 000 | 10 | ~5 分钟 | ~$0.03 |
| `demo` | 50 000 | 5 000 | 20 | ~1–2 小时 | ~$0.50 |
| `paper` | 1 000 000 | 10 000 | 30 | ~12 小时 | ~$3.60 |

notebook 每 `ckpt_every` 迭代保存 `ckpt_latest.pt` 并画出 loss 曲线。训练中断后想继续，把第 10 格里的 `RESUME_FROM` 设为 `"ckpt_latest.pt"` 即可。

### Blackwell (sm_120) 注意事项

RTX 5090 是 Blackwell 架构，需要 PyTorch **2.7+**。如果看到 `no kernel image available for sm_120` 错误，第 2 格会打印 fallback 安装命令（PyTorch nightly + cu128）。

## 推理 + UI

本机技术栈：uv 管理的 Python 包 + FastAPI 后端 + three.js 3D 前端。

### 首次安装

**用 uv（推荐）:**
```bash
uv sync              # 安装依赖
uv run pytest        # 应该 144 个测试全过
```

**用 conda + pip:**
```bash
conda create -n deepcube python=3.12 -y
conda activate deepcube
pip install -e .                 # 读 pyproject.toml 装所有依赖
pip install pytest httpx ruff    # 测试相关 (可选)
pytest
```

### 启动服务器

```bash
uv run deepcube-serve
# 或自定义 host/port:
DEEPCUBE_HOST=0.0.0.0 DEEPCUBE_PORT=8000 uv run deepcube-serve
```

浏览器打开 `http://127.0.0.1:8000`——加载 3D 魔方，**Scramble** 按钮动画播放随机扰动，**Solve** 把当前状态发给服务器并动画展示求解路径。

### 三个操作

- **Scramble** — 发起随机扰动
- **Edit mode** — 进入编辑模式后，自动 Reset 到 solved，点击任意非中心贴纸循环切换颜色（U→F→R→B→L→D→U），拖拽旋转视角；用来把 UI 状态对齐你手里的实体魔方
- **Solve** — 用选中的求解器（Kociemba 或 DeepCubeA）求解当前状态并动画展示

### 求解器选择

Solve 面板有 2 个 radio：

- **⦿ Kociemba** — 默认。两阶段算法，毫秒级响应，HTM 最优。**无需 `.pt`，开箱即用**
- **○ DeepCubeA** — 习得的启发式 + A\*。**未加载 `.pt` 时自动禁用**

切到 DeepCubeA 才会出现 `λ` 滑块（Kociemba 没可调参数）。

> **度量差异说明:** 两个算法不是直接可比的步数。
> - Kociemba 优化 **HTM (Half-Turn Metric)**，把 `U2` 这种 180° 转算 1 步，最长 20 步
> - DeepCubeA 优化 **QTM (Quarter-Turn Metric)**，只算 90° 转，最长 26 步
>
> 结果栏同时展示两个数字，方便对比。

### 放入训练好的权重

把从 Vast.ai 下载的 `.pt` 文件放到 `checkpoints/deepcube_cube3.pt`（或者用 `DEEPCUBE_CHECKPOINT` 指向别处），然后重启服务器即可。**没有 `.pt` 服务器也能启动**——只是 DeepCubeA 求解器会处于禁用状态，Kociemba 照常工作。

### HTTP API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 3D 前端 |
| `/healthz` | GET | `{ok, model_loaded, checkpoint_path}` |
| `/meta` | GET | 魔方常量（moves, MOVE_PERMS, solved state, 贴纸布局），给前端消费 |
| `/scramble` | POST | `{depth, seed?}` → `{state, moves}` |
| `/solve` | POST | `{state, solver?, lambda_weight?, batch_size?, ...}` → `{solver, solved, moves, path_length, path_length_htm, nodes_expanded, nodes_generated, elapsed_sec, stop_reason}` |

`solver` 取值 `"kociemba"`（默认）或 `"deepcube"`。`path_length` 永远是 QTM；`path_length_htm` 只有 Kociemba 会填。

## 实测数据（服务器端 HTTP 测量）

| 扰动深度 | QTM | HTM | Kociemba 耗时 |
|---|---|---|---|
| 5 | 5 | 5 | 2 ms |
| 15 | 29 | 20 | 8 ms |
| 25 | 31 | 21 | 23 ms |

## 项目进度

- [x] 魔方环境 + 移动正确性测试
- [x] PyTorch 模型（14.66M 参数）
- [x] AVI 训练循环，AMP (bf16) + `torch.compile` + checkpointing + 断点续训
- [x] 贪心 rollout 健全性检查
- [x] Batch-Weighted A\* 求解器（`deepcube/search.py`，19 个测试）
- [x] Kociemba 两阶段求解器（`deepcube/solver_kociemba.py`，30 个测试）
- [x] FastAPI 后端（`deepcube/server.py`，17 个测试）
- [x] three.js 3D 前端（`static/index.html`），支持 click-to-edit 和求解器选择

## 仓库结构

```
.
├── README.md / README.zh.md
├── train.ipynb             # 上传到 Vast.ai 用
├── build_notebook.py       # 重新生成 train.ipynb 的生成器
├── pyproject.toml          # uv 管理的 Python 3.12 包
├── deepcube/
│   ├── cube3.py            # 环境: 54 贴纸状态、12 个移动、scramble、one-hot
│   ├── model.py            # DeepCubeANet, load_checkpoint
│   ├── search.py           # Batch-Weighted A*
│   ├── solver_kociemba.py  # Kociemba 两阶段 wrapper + 状态格式转换
│   └── server.py           # FastAPI: /healthz, /meta, /scramble, /solve
├── static/
│   └── index.html          # three.js 3D 前端
├── tests/                  # pytest 测试（144 个 case）
├── checkpoints/            # 把 deepcube_cube3.pt 放这里
└── .gitignore
```

## 常见问题

**Q: `pip install -e .` 装 kociemba 失败？**
A: kociemba 是 C 扩展，需要编译器。Mac 装 Xcode CLT：`xcode-select --install`；Linux 装 `gcc` / `build-essential`。

**Q: torch 下载巨慢？**
A: 默认从 PyPI 拉 ~800MB。换源：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Q: 浏览器打开页面黑屏？**
A: 看 F12 Console 有没有报错。通常是 three.js CDN 被墙——可以换镜像源（改 `static/index.html` 里的 `importmap`）。

## 参考

- Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). *Solving the Rubik's cube with deep reinforcement learning and search*. Nature Machine Intelligence, 1(8), 356–363.
- [yeates/deepcube-full](https://github.com/yeates/deepcube-full) — 原始 TF1/Keras 实现 + web demo
- [Kociemba's two-phase algorithm](http://kociemba.org/cube.htm) — 经典的近最优求解器
- [muodov/kociemba](https://github.com/muodov/kociemba) — 我们用的 C 后端 Python 包
