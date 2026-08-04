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
