import math

import torch

from src import task_mdp as mdp


def test_crouch_height_target_endpoints_are_high():
    phase = torch.tensor([0.0, 0.999])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([0.11, 0.11]), atol=2e-3)


def test_crouch_height_target_plateau_is_low():
    phase = torch.tensor([0.375, 0.5, 0.624])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.full((3,), 0.075), atol=1e-6)


def test_crouch_height_target_descent_midpoint():
    phase = torch.tensor([0.1875])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([(0.11 + 0.075) / 2]), atol=1e-6)


def test_crouch_height_target_rise_midpoint():
    phase = torch.tensor([0.8125])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([(0.11 + 0.075) / 2]), atol=1e-6)


_BLEND = {"descent_end": 0.10, "hold_end": 0.50, "rise_end": 0.60}


def test_blend_zero_standing_at_start_and_top_hold():
    phase = torch.tensor([0.0, 0.6, 0.8, 0.999])
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.zeros(4), atol=1e-6)


def test_blend_one_on_low_hold():
    phase = torch.tensor([0.10, 0.3, 0.499])
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.ones(3), atol=1e-6)


def test_blend_descent_and_rise_midpoints():
    phase = torch.tensor([0.05, 0.55])
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.tensor([0.5, 0.5]), atol=1e-6)


def test_reward_is_one_when_height_matches_target():
    cmd_cos = torch.tensor([math.cos(2 * math.pi * 0.5)])
    cmd_sin = torch.tensor([math.sin(2 * math.pi * 0.5)])
    com_height = torch.tensor([0.075])
    r = mdp.crouch_glide_reward_from_values(com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-3)


def test_reward_decays_when_off_by_one_std():
    cmd_cos = torch.tensor([math.cos(2 * math.pi * 0.5)])
    cmd_sin = torch.tensor([math.sin(2 * math.pi * 0.5)])
    com_height = torch.tensor([0.075 + 0.02])
    r = mdp.crouch_glide_reward_from_values(com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02)
    assert torch.allclose(r, torch.tensor([math.exp(-1.0)]), atol=1e-3)


def test_reward_at_phase_zero_expects_high_stance():
    cmd_cos = torch.tensor([1.0, 1.0])
    cmd_sin = torch.tensor([0.0, 0.0])
    com_height = torch.tensor([0.11, 0.075])
    r = mdp.crouch_glide_reward_from_values(com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02)
    assert r[0] > 0.99
    assert r[1] < 0.2
