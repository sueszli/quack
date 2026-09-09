import torch

from src.task_mdp import slope_move_masks


def test_move_up_when_reached_bottom():
    # Promotion above size_x*0.4 = 3.2.
    dist = torch.tensor([5.0, 4.1])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert bool(up[0]) and bool(up[1])
    assert not bool(down[0]) and not bool(down[1])


def test_move_down_when_stuck_early():
    # Demotion below size_x*0.2 = 1.6.
    dist = torch.tensor([0.5, 1.0])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(up[1])
    assert bool(down[0]) and bool(down[1])


def test_stay_in_middle_band():
    dist = torch.tensor([2.5])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(down[0])


def test_move_up_boundary_at_04():
    # 0.4*size_x means the robot covered most of the ramp before the runout flat.
    dist = torch.tensor([3.3])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert bool(up[0])
    assert not bool(down[0])

    dist_mid = torch.tensor([3.0])
    up_mid, down_mid = slope_move_masks(dist_mid, size_x=8.0)
    assert not bool(up_mid[0]) and not bool(down_mid[0])
