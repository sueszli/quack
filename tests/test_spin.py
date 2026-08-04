import math

import torch

from mjlab_microduck.tasks import mdp

# Enveloppe du spec : accel 0.5s / régime 1.6s / freinage 0.5s / repos 1.4s sur 4s.
_ENV = dict(rate_max=6.0, accel_end=0.125, hold_end=0.525, brake_end=0.650)


def test_spin_rate_segment_boundaries():
    # bornes des 4 segments : 0 au départ, plein régime sur [accel_end, hold_end],
    # encore plein régime au tout début du freinage, 0 dès le segment de repos.
    phase = torch.tensor([0.0, 0.125, 0.30, 0.525, 0.650, 0.80, 0.999])
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    expected = torch.tensor([0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0])
    assert torch.allclose(w, expected, atol=1e-6)


def test_spin_rate_accel_ramp_is_increasing():
    phase = torch.linspace(0.0, 0.125, 20)
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.all(w[1:] >= w[:-1])
    # milieu de la rampe de lancement -> moitié de la cible
    mid = mdp.spin_rate_by_phase(torch.tensor([0.0625]), **_ENV)
    assert torch.allclose(mid, torch.tensor([3.0]), atol=1e-6)


def test_spin_rate_brake_ramp_is_decreasing():
    phase = torch.linspace(0.525, 0.6499, 20)
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.all(w[1:] <= w[:-1])
    # milieu du freinage -> moitié de la cible
    mid = mdp.spin_rate_by_phase(torch.tensor([0.5875]), **_ENV)
    assert torch.allclose(mid, torch.tensor([3.0]), atol=1e-6)


def test_spin_rate_integral_is_two_turns():
    # LE test qui protège la cible du spec : l'aire sous l'enveloppe sur un cycle
    # de 4 s doit valoir ~4*pi rad = 2 tours. Enveloppe exacte = 12.6 rad,
    # 4*pi = 12.566 -> tolérance 1 %.
    n = 100_000
    phase = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    integral = float(w.mean()) * 4.0
    assert abs(integral - 4 * math.pi) / (4 * math.pi) < 0.01


def test_spin_gate_is_normalized_rate():
    phase = torch.tensor([0.0, 0.0625, 0.30, 0.5875, 0.80])
    gate = mdp.spin_gate_by_phase(phase, **_ENV)
    rate = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.allclose(gate, rate / 6.0, atol=1e-6)
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


def test_spin_gate_is_zero_over_the_whole_rest_segment():
    # pendant le repos aucune amorce ne doit pousser au ciseau -> porte nulle,
    # c'est ce qui donne une sortie de trick propre vers la policy roller.
    phase = torch.linspace(0.650, 0.999, 50)
    gate = mdp.spin_gate_by_phase(phase, **_ENV)
    assert torch.allclose(gate, torch.zeros_like(gate), atol=1e-6)


# ── faux env minimal : permet de tester les wrappers de reward sans MuJoCo ────
class _FakeData:
    def __init__(self, ang_vel_b=None, lin_vel_b=None, joint_pos=None, joint_vel=None):
        self.root_link_ang_vel_b = ang_vel_b
        self.root_link_lin_vel_b = lin_vel_b
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel


class _FakeEntity:
    """Entity minimale : find_joints() résout par nom depuis un dict {nom: index}."""

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
        assert matched, f"aucun joint ne matche {pattern!r} parmi {names}"
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
    """Commande du slot telle que la voit la policy : [cos(2*pi*phi), sin(...), 0]."""
    p = torch.as_tensor(phases, dtype=torch.float32)
    return torch.stack(
        [torch.cos(2 * math.pi * p), torch.sin(2 * math.pi * p), torch.zeros_like(p)],
        dim=-1,
    )


# ── phase recover ────────────────────────────────────────────────────────────
def test_spin_phase_from_command_roundtrip():
    phases = torch.tensor([0.0, 0.125, 0.4, 0.65, 0.9])
    got = mdp.spin_phase_from_command(_phase_cmd(phases))
    assert torch.allclose(got, phases, atol=1e-5)


# ── spin_rate_track ──────────────────────────────────────────────────────────
def test_spin_rate_reward_peaks_on_exact_match():
    w = torch.tensor([6.0, 6.0])
    target = torch.tensor([6.0, 4.5])
    r = mdp.spin_rate_reward_from_values(w, target, std=1.5)
    # erreur nulle -> 1.0 ; erreur = 1 std -> exp(-1)
    assert torch.allclose(r, torch.tensor([1.0, math.exp(-1.0)]), atol=1e-6)


def test_spin_rate_track_uses_yaw_and_phase():
    # phase 0.30 = plein régime -> cible 6 rad/s. Un robot qui tourne à 6 rad/s
    # doit toucher 1.0 ; un robot immobile doit être largement en dessous.
    ang = torch.tensor([[0.0, 0.0, 6.0], [0.0, 0.0, 0.0]])
    env = _FakeEnv(
        _FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30])
    )
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] > 0.99
    assert r[1] < 0.01


def test_spin_rate_track_wants_stillness_during_rest():
    # phase 0.80 = repos -> cible 0 : tourner encore est puni, être immobile payé.
    ang = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
    env = _FakeEnv(
        _FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.80, 0.80])
    )
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] > 0.99
    assert r[1] < 0.01


def test_spin_rate_track_penalizes_wrong_direction():
    # tourner à -6 rad/s (horaire) quand on demande +6 doit être pire qu'immobile
    ang = torch.tensor([[0.0, 0.0, -6.0], [0.0, 0.0, 0.0]])
    env = _FakeEnv(
        _FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30])
    )
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] < r[1]


# ── spin_rate_l1 ─────────────────────────────────────────────────────────────
def test_spin_rate_l1_is_negative_absolute_error():
    ang = torch.tensor([[0.0, 0.0, 6.0], [0.0, 0.0, 2.0]])
    env = _FakeEnv(
        _FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30])
    )
    r = mdp.spin_rate_l1(env)
    assert torch.allclose(r, torch.tensor([0.0, -4.0]), atol=1e-5)


# ── spin_stay_in_place ───────────────────────────────────────────────────────
def test_spin_stay_in_place_is_squared_planar_speed():
    lin = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.4, 9.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(lin_vel_b=lin)))
    c = mdp.spin_stay_in_place(env)
    # 0.3^2 + 0.4^2 = 0.25 ; la composante z est ignorée
    assert torch.allclose(c, torch.tensor([0.0, 0.25]), atol=1e-6)
