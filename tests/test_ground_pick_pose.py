import torch
from mjlab_microduck.tasks.mdp import phase_pose_blend

DESCENT_END, HOLD_END, RISE_END = 0.15, 0.50, 0.65


def test_phase_pose_blend_keypoints():
    phase = torch.tensor([0.0, 0.075, 0.15, 0.30, 0.50, 0.575, 0.65, 0.80])
    b = phase_pose_blend(phase, DESCENT_END, HOLD_END, RISE_END)
    expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    assert torch.allclose(b, expected, atol=1e-6), b


def test_phase_pose_blend_range():
    phase = torch.linspace(0.0, 1.0, 101)
    b = phase_pose_blend(phase, DESCENT_END, HOLD_END, RISE_END)
    assert b.min() >= 0.0 and b.max() <= 1.0
