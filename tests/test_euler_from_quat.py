import math

import pytest
import torch
from mjlab.utils.lab_api.math import euler_xyz_from_quat

from src import task_mdp as microduck_mdp

_EDGE_QUATS = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.0, 0.70710678, 0.0], [0.70710678, 0.0, -0.70710678, 0.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5]])


def _euler_xyz_inline(quat):
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def _normalize(quat):
    return quat / quat.norm(dim=-1, keepdim=True)


def _quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor([[cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]])


class _Data:
    def __init__(self, quat):
        self.root_link_pos_w = torch.zeros(len(quat), 3)
        self.root_link_quat_w = quat


class _Scene:
    def __init__(self, data):
        self._asset = type("A", (), {"data": data})()
        self.terrain = type("T", (), {"env_origins": torch.zeros(len(data.root_link_quat_w), 3)})()

    def __getitem__(self, _):
        return self._asset


class _Env:
    def __init__(self, quat, cmd):
        self.scene = _Scene(_Data(quat))
        self.command_manager = type("C", (), {"get_command": lambda _s, _n: cmd})()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_inline_expansion(dtype):
    torch.manual_seed(0)
    quat = torch.cat([_EDGE_QUATS.to(dtype), _normalize(torch.randn(20000, 4, dtype=dtype))])
    for got, want in zip(euler_xyz_from_quat(quat), _euler_xyz_inline(quat), strict=True):
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)


def test_gimbal_lock_pitch_is_exactly_quarter_turn():
    _, pitch, _ = euler_xyz_from_quat(_normalize(_EDGE_QUATS[1:3]))
    torch.testing.assert_close(pitch.abs(), torch.full_like(pitch, torch.pi / 2))


@pytest.mark.parametrize(("axis", "index"), [("roll", 3), ("pitch", 4), ("yaw", 5)])
def test_body_pose_tracking_6d_maps_each_angle_to_its_command_slot(axis, index):
    angle = math.radians(10)
    quat = _quat_from_rpy(**{"roll": 0.0, "pitch": 0.0, "yaw": 0.0, axis: angle})

    matched = torch.zeros(1, 6)
    matched[0, index] = angle
    swapped = torch.zeros(1, 6)
    swapped[0, 3 + (index - 3 + 1) % 3] = angle

    kwargs = {"nominal_height": 0.0, "xy_std": 1.0, "z_std": 1.0, "angle_std": math.radians(2)}
    reward_matched = microduck_mdp.body_pose_tracking_6d(_Env(quat, matched), **kwargs)
    reward_swapped = microduck_mdp.body_pose_tracking_6d(_Env(quat, swapped), **kwargs)

    assert reward_matched.item() == pytest.approx(1.0, abs=1e-4)
    assert reward_swapped.item() < 0.9
