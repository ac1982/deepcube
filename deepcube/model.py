"""Residual MLP heuristic for DeepCubeA.

Architecture (per Agostinelli et al., 2019):

    one-hot(54*6=324) -> 5000 -> 1000 -> [ResBlock(1000) x4] -> 1

Each linear layer is followed by BatchNorm1d + ReLU. Residual blocks are two
linear layers with a skip connection.

~14.7 M parameters. Matches the network used in `train.ipynb`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepcube.cube3 import N_STICKERS

__all__ = ["INPUT_DIM", "ResBlock", "DeepCubeANet", "load_checkpoint", "count_parameters"]

INPUT_DIM = N_STICKERS * 6  # 324


class ResBlock(nn.Module):
    """Two FC+BN layers with a ReLU-after-add skip."""

    def __init__(self, dim: int = 1000) -> None:
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
    def __init__(self, n_res_blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(INPUT_DIM, 5000),
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


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DeepCubeANet, dict[str, Any]]:
    """Load a trained model from a `.pt` file written by `train.ipynb`.

    Accepts both the final save (`deepcube_cube3.pt`, keys: net/cfg/loss_hist/elapsed)
    and intermediate checkpoints (`ckpt_latest.pt`, which also include opt/target/iter).
    Returns the net in eval mode plus the raw checkpoint dict (for inspecting cfg,
    loss history, etc).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "net" not in ckpt:
        raise ValueError(f"checkpoint at {path} has no 'net' state_dict")
    net = DeepCubeANet().to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()
    return net, ckpt
