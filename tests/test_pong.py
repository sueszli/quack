import pytest
import torch

from src.pong import FRAMES, SKINS, Clean, Flowers, H, Pong, W, clamp_params


@pytest.mark.parametrize("name", list(SKINS))
def test_skin_contract(name):
    env = Pong(8, SKINS[name](), device="cpu")
    assert env.obs().shape == (8, FRAMES * 3, H, W) and env.obs().dtype == torch.uint8
    for _ in range(50):
        obs, r, _, _ = env.step(torch.randint(0, 2, (8,)))
    assert obs.shape == (8, FRAMES * 3, H, W)
    assert torch.isfinite(r).all()
    assert env.frames.amax() == 255


def test_deterministic_given_seed():
    outs = []
    for _ in range(2):
        env = Pong(16, Flowers(), device="cpu", seed=3)
        acts = torch.randint(0, 2, (100, 16), generator=torch.Generator().manual_seed(0))
        for a in acts:
            obs, *_ = env.step(a)
        outs.append(obs.clone())
    assert torch.equal(outs[0], outs[1])


def test_tracker_ramps_then_loses():
    env = Pong(32, Clean(), device="cpu")
    max_hits, misses = torch.zeros(32), 0
    for _ in range(1000):
        _, _, _, info = env.step((env.s.by > env.s.py).long())
        max_hits = torch.maximum(max_hits, info["hits"].float())
        misses += int(info["miss"].sum())
    assert (max_hits >= 5).all()
    assert 0 < misses < 32 * 20


def test_static_paddle_misses():
    env = Pong(64, Clean(), device="cpu")
    misses = sum(int(env.step(torch.zeros(64, dtype=torch.long))[3]["miss"].sum()) for _ in range(400))
    assert misses > 30


def test_params_clamped():
    assert clamp_params({"ball_speed_mult": 10.0, "paddle_half_h": 0.0, "junk": 1}) == {"ball_speed_mult": 2.0, "paddle_half_h": 0.04}
    assert clamp_params({}) == {"ball_speed_mult": 1.0, "paddle_half_h": 0.08}
