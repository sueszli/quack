"""wheel_glide_reward: rewards forward ROLLING of the wheels (gliding under
gravity), capped at cap_speed, zero if the wheels roll backward, NaN-safe.
Independent of any command (the slope task has a zero command).
"""

import re

import torch

from src.task_mdp import wheel_glide_reward

_WHEELS = {"passive_LF_wheel": 0, "passive_LR_wheel": 1, "passive_RF_wheel": 2, "passive_RR_wheel": 3}


class _Data:
    def __init__(self, omegas):
        self.joint_vel = torch.tensor([omegas], dtype=torch.float32)


class _Asset:
    def __init__(self, data):
        self.data = data

    def find_joints(self, pattern):
        ids = [i for name, i in _WHEELS.items() if re.fullmatch(pattern, name)]
        assert ids, pattern
        return ids, None


class _Env:
    def __init__(self, omegas):
        self._a = _Asset(_Data(omegas))

    def __getitem__(self, _k):
        return self._a

    @property
    def scene(self):
        return self


def test_rewards_forward_roll_below_cap():
    out = wheel_glide_reward(_Env([10.0, 10.0, 10.0, 10.0]), cap_speed=0.35)
    assert abs(float(out[0]) - 0.175) < 1e-6


def test_caps_fast_roll():
    out = wheel_glide_reward(_Env([40.0, 40.0, 40.0, 40.0]), cap_speed=0.35)
    assert abs(float(out[0]) - 0.35) < 1e-6


def test_zero_when_wheels_roll_backward():
    out = wheel_glide_reward(_Env([-10.0, -10.0, -10.0, -10.0]), cap_speed=0.35)
    assert float(out[0]) == 0.0


def test_nan_safe():
    out = wheel_glide_reward(_Env([float("nan"), 10.0, 10.0, 10.0]), cap_speed=0.35)
    assert float(out[0]) == 0.0
