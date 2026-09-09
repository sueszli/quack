import math

import torch

from src import task_mdp as mdp

_ENV = {"rate_max": 6.0, "accel_end": 0.125, "hold_end": 0.525, "brake_end": 0.650}


def test_spin_rate_segment_boundaries():
    phase = torch.tensor([0.0, 0.125, 0.30, 0.525, 0.650, 0.80, 0.999])
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    expected = torch.tensor([0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0])
    assert torch.allclose(w, expected, atol=1e-6)


def test_spin_rate_accel_ramp_is_increasing():
    phase = torch.linspace(0.0, 0.125, 20)
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.all(w[1:] >= w[:-1])
    mid = mdp.spin_rate_by_phase(torch.tensor([0.0625]), **_ENV)
    assert torch.allclose(mid, torch.tensor([3.0]), atol=1e-6)


def test_spin_rate_brake_ramp_is_decreasing():
    phase = torch.linspace(0.525, 0.6499, 20)
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.all(w[1:] <= w[:-1])
    mid = mdp.spin_rate_by_phase(torch.tensor([0.5875]), **_ENV)
    assert torch.allclose(mid, torch.tensor([3.0]), atol=1e-6)


def test_spin_rate_integral_matches_trapezoid_shape_at_rate_max_6():
    n = 100_000
    phase = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    integral = float(w.mean()) * 4.0
    assert abs(integral - 4 * math.pi) / (4 * math.pi) < 0.01


def test_spin_rate_max_integrates_to_2_1_times_itself_per_cycle():
    n = 100_000
    phase = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    w = mdp.spin_rate_by_phase(phase, rate_max=mdp.SPIN_RATE_MAX, accel_end=mdp.SPIN_ACCEL_END, hold_end=mdp.SPIN_HOLD_END, brake_end=mdp.SPIN_BRAKE_END)
    integral = float(w.mean()) * mdp.SPIN_PERIOD
    expected = 2.1 * mdp.SPIN_RATE_MAX
    assert abs(integral - expected) / expected < 0.01


def test_spin_gate_is_normalized_rate():
    phase = torch.tensor([0.0, 0.0625, 0.30, 0.5875, 0.80])
    gate = mdp.spin_gate_by_phase(phase, **_ENV)
    rate = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.allclose(gate, rate / 6.0, atol=1e-6)
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


def test_spin_gate_is_zero_over_the_whole_rest_segment():
    phase = torch.linspace(0.650, 0.999, 50)
    gate = mdp.spin_gate_by_phase(phase, **_ENV)
    assert torch.allclose(gate, torch.zeros_like(gate), atol=1e-6)


class _FakeData:
    def __init__(self, ang_vel_b=None, lin_vel_b=None, joint_pos=None, joint_vel=None):
        self.root_link_ang_vel_b = ang_vel_b
        self.root_link_lin_vel_b = lin_vel_b
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel


class _FakeEntity:
    """Minimal entity: find_joints() resolves by name from a {name: index} dict."""

    def __init__(self, data, joint_ids=None):
        self.data = data
        self._joint_ids = joint_ids or {}

    def find_joints(self, pattern):
        import re

        names = list(self._joint_ids.keys())
        if isinstance(pattern, (list, tuple)):
            matched = [n for n in names if n in pattern]
        else:
            matched = [n for n in names if re.fullmatch(pattern, n)]
        assert matched, f"no joint matches {pattern!r} among {names}"
        return [self._joint_ids[n] for n in matched], matched


class _FakeCommandManager:
    def __init__(self, cmd):
        self._cmd = cmd

    def get_command(self, name):
        return self._cmd


class _FakeSensorData:
    def __init__(self, current_contact_time):
        self.current_contact_time = current_contact_time


class _FakeSensor:
    def __init__(self, current_contact_time):
        self.data = _FakeSensorData(current_contact_time)


class _FakeEnv:
    def __init__(self, entity, cmd=None, sensors=None):
        self.scene = {"robot": entity, **(sensors or {})}
        self.command_manager = _FakeCommandManager(cmd)
        self.device = "cpu"


def _phase_cmd(phases):
    """Slot command as the policy sees it: [cos(2*pi*phi), sin(...), 0]."""
    p = torch.as_tensor(phases, dtype=torch.float32)
    return torch.stack([torch.cos(2 * math.pi * p), torch.sin(2 * math.pi * p), torch.zeros_like(p)], dim=-1)


def test_spin_phase_from_command_roundtrip():
    phases = torch.tensor([0.0, 0.125, 0.4, 0.65, 0.9])
    got = mdp.spin_phase_from_command(_phase_cmd(phases))
    assert torch.allclose(got, phases, atol=1e-5)


def test_spin_rate_reward_peaks_on_exact_match():
    w = torch.tensor([6.0, 6.0])
    target = torch.tensor([6.0, 4.5])
    r = mdp.spin_rate_reward_from_values(w, target, std=1.5)
    assert torch.allclose(r, torch.tensor([1.0, math.exp(-1.0)]), atol=1e-6)


def test_spin_rate_track_uses_yaw_and_phase():
    ang = torch.tensor([[0.0, 0.0, mdp.SPIN_RATE_MAX], [0.0, 0.0, 0.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30]))
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] > 0.99
    assert r[1] < 0.05


def test_spin_rate_track_wants_stillness_during_rest():
    ang = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.80, 0.80]))
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] > 0.99
    assert r[1] < 0.01


def test_spin_rate_track_penalizes_wrong_direction():
    ang = torch.tensor([[0.0, 0.0, -mdp.SPIN_RATE_MAX], [0.0, 0.0, 0.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30]))
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] < r[1]


def test_spin_rate_l1_is_negative_absolute_error():
    ang = torch.tensor([[0.0, 0.0, mdp.SPIN_RATE_MAX], [0.0, 0.0, 1.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30]))
    r = mdp.spin_rate_l1(env)
    expected = torch.tensor([0.0, -(mdp.SPIN_RATE_MAX - 1.0)])
    assert torch.allclose(r, expected, atol=1e-5)


def test_spin_stay_in_place_is_squared_planar_speed():
    lin = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.4, 9.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(lin_vel_b=lin)), cmd=_phase_cmd([0.30, 0.30]))
    c = mdp.spin_stay_in_place(env)
    assert torch.allclose(c, torch.tensor([0.0, 0.25]), atol=1e-6)


def test_spin_stay_in_place_is_attenuated_during_the_launch_ramp():
    lin = torch.tensor([[0.3, 0.4, 0.0], [0.3, 0.4, 0.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(lin_vel_b=lin)), cmd=_phase_cmd([0.05, 0.30]))
    c = mdp.spin_stay_in_place(env, launch_scale=0.2, accel_end=0.125)
    assert torch.allclose(c, torch.tensor([0.05, 0.25]), atol=1e-6)
    assert c[0] < c[1]


def test_spin_stay_in_place_is_full_price_during_rest():
    lin = torch.tensor([[0.3, 0.4, 0.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(lin_vel_b=lin)), cmd=_phase_cmd([0.80]))
    c = mdp.spin_stay_in_place(env)
    assert torch.allclose(c, torch.tensor([0.25]), atol=1e-6)


_WHEEL_IDS = {"passive_LF_wheel": 0, "passive_LR_wheel": 1, "passive_RF_wheel": 2, "passive_RR_wheel": 3}


def _wheel_env(vel_rows, phases):
    vel = torch.tensor(vel_rows, dtype=torch.float32)
    entity = _FakeEntity(_FakeData(joint_vel=vel), joint_ids=_WHEEL_IDS)
    return _FakeEnv(entity, cmd=_phase_cmd(phases))


def test_wheel_differential_rewards_counter_rolling_wheels():
    env = _wheel_env([[-10.0, -10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0], [10.0, 10.0, -10.0, -10.0]], [0.30, 0.30, 0.30])
    r = mdp.spin_wheel_differential(env, omega_scale=20.0)
    assert r[0] > 0.5
    assert torch.allclose(r[1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(r[2], torch.tensor(0.0), atol=1e-6)


def test_wheel_differential_is_gated_off_during_rest():
    env = _wheel_env([[-10.0, -10.0, 10.0, 10.0]], [0.80])
    r = mdp.spin_wheel_differential(env, omega_scale=20.0)
    assert torch.allclose(r, torch.zeros(1), atol=1e-6)


def test_wheel_differential_saturates():
    env = _wheel_env([[-10.0, -10.0, 10.0, 10.0], [-100.0, -100.0, 100.0, 100.0]], [0.30, 0.30])
    r = mdp.spin_wheel_differential(env, omega_scale=20.0)
    assert r[1] > r[0]
    assert r[1] <= 1.0


def test_wheel_differential_from_values_is_pure():
    diff = torch.tensor([20.0, 0.0, -20.0])
    gate = torch.ones(3)
    r = mdp.spin_wheel_differential_from_values(diff, gate, omega_scale=20.0)
    expected = torch.tensor([math.tanh(1.0), 0.0, 0.0])
    assert torch.allclose(r, expected, atol=1e-6)


def test_spin_grounded_rewards_both_blades_down_and_is_gated():
    contact = torch.tensor([[0.2, 0.3], [0.2, 0.0], [0.0, 0.0], [0.2, 0.3]])
    entity = _FakeEntity(_FakeData())
    env = _FakeEnv(entity, cmd=_phase_cmd([0.30, 0.30, 0.30, 0.80]), sensors={"feet_ground_contact": _FakeSensor(contact)})
    r = mdp.spin_grounded(env, sensor_name="feet_ground_contact")
    assert torch.allclose(r, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


_LEG_IDS = {"left_hip_pitch": 0, "left_knee": 1, "right_hip_pitch": 2, "right_knee": 3}


def _leg_env(pos_rows, phases):
    pos = torch.tensor(pos_rows, dtype=torch.float32)
    entity = _FakeEntity(_FakeData(joint_pos=pos), joint_ids=_LEG_IDS)
    return _FakeEnv(entity, cmd=_phase_cmd(phases))


def test_leg_antisymmetry_prefers_scissor_over_mirror():
    env = _leg_env([[0.4, 0.3, 0.4, 0.3], [0.4, 0.3, -0.4, -0.3]], [0.30, 0.30])
    r = mdp.leg_antisymmetry(env)
    assert torch.allclose(r, torch.tensor([0.0, -0.7]), atol=1e-6)
    assert r[0] > r[1]


def test_leg_antisymmetry_is_gated_off_during_rest():
    env = _leg_env([[0.4, 0.3, -0.4, -0.3]], [0.80])
    r = mdp.leg_antisymmetry(env)
    assert torch.allclose(r, torch.zeros(1), atol=1e-6)


_NECK_IDS = {"neck_pitch": 0, "head_pitch": 1, "head_roll": 2, "head_yaw": 3}


def test_neck_joint_pos_l2_pattern_can_exclude_head_yaw():
    class _NeckData(_FakeData):
        def __init__(self, joint_pos, default_joint_pos):
            super().__init__(joint_pos=joint_pos)
            self.default_joint_pos = default_joint_pos

    pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    default = torch.zeros(1, 4)
    entity = _FakeEntity(_NeckData(pos, default), joint_ids=_NECK_IDS)
    env = _FakeEnv(entity)

    assert torch.allclose(mdp.neck_joint_pos_l2(env), torch.tensor([1.0]), atol=1e-6)
    assert torch.allclose(mdp.neck_joint_pos_l2(env, pattern=r"^(neck_pitch|head_pitch|head_roll)$"), torch.tensor([0.0]), atol=1e-6)
