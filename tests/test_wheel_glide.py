"""wheel_glide_reward prices forward ROLLING, not base speed, and is independent
of any command (the slope task commands zero).
"""

import re

import torch

from src.task_mdp import wheel_glide_reward

_WHEELS = {"passive_LF_wheel": 0, "passive_LR_wheel": 1, "passive_RF_wheel": 2, "passive_RR_wheel": 3}


class _Data:
    def __init__(self, omegas):
        # columns 0..3 = LF, LR, RF, RR
        self.joint_vel = torch.tensor([omegas], dtype=torch.float32)


class _Asset:
    def __init__(self, data):
        self.data = data

    def find_joints(self, pattern):
        # Real Entity.find_joints resolves regexes; mdp queries use
        # spelling-tolerant patterns such as "passive_LF_?wheel".
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
    # 10 rad/s * 0.0175 m wheel radius = 0.175 m/s, under the cap.
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
