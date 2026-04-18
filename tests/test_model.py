from __future__ import annotations

import numpy as np
import pytest
import torch

from deepcube.cube3 import SOLVED, batch_scramble, one_hot
from deepcube.model import INPUT_DIM, DeepCubeANet, count_parameters, load_checkpoint


def test_input_dim():
    assert INPUT_DIM == 324


def test_forward_shape_and_finite():
    net = DeepCubeANet().eval()
    rng = np.random.default_rng(0)
    states, _ = batch_scramble(8, 5, rng)
    x = torch.from_numpy(one_hot(states))
    with torch.no_grad():
        y = net(x)
    assert y.shape == (8,)
    assert torch.isfinite(y).all()


def test_forward_single_state():
    """Single state must work too — used by the greedy rollout path."""
    net = DeepCubeANet().eval()
    x = torch.from_numpy(one_hot(SOLVED)).unsqueeze(0)  # (1, 324)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1,)


def test_param_count():
    """Guard against accidental architecture changes — we target the paper's ~14.7M."""
    n = count_parameters(DeepCubeANet())
    assert 14_000_000 < n < 16_000_000, f"unexpected param count: {n}"


def test_save_load_roundtrip(tmp_path):
    net = DeepCubeANet().eval()
    x = torch.randn(4, INPUT_DIM)
    with torch.no_grad():
        y_before = net(x)

    ckpt_path = tmp_path / "roundtrip.pt"
    torch.save({"net": net.state_dict(), "cfg": {"preset": "test"}}, ckpt_path)

    loaded, meta = load_checkpoint(ckpt_path, device="cpu")
    with torch.no_grad():
        y_after = loaded(x)

    assert torch.allclose(y_before, y_after), "weights changed across save/load"
    assert meta["cfg"]["preset"] == "test"


def test_load_checkpoint_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_load_checkpoint_rejects_bad_format(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"not_net": {}}, bad)
    with pytest.raises(ValueError, match="no 'net' state_dict"):
        load_checkpoint(bad)
