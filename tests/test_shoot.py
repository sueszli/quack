import torch
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg, kick_pose_target


def test_phase_cmd_randomize_flag_default_true():
    cfg = GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(0.1, 0.5),
        ranges=[[0, 0], [0, 0], [0, 0]],
    )
    assert cfg.randomize_phase is True


def test_phase_cmd_randomize_flag_settable_false():
    cfg = GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(0.1, 0.5),
        ranges=[[0, 0], [0, 0], [0, 0]],
        randomize_phase=False,
    )
    assert cfg.randomize_phase is False


W, K, R = 0.35, 0.45, 0.75  # windup_end, kick_end, return_end
STAND = torch.tensor([0.0, 0.0])
BACK = torch.tensor([1.0, -1.0])
FWD = torch.tensor([-1.0, 2.0])


def _t(phase):
    return kick_pose_target(torch.tensor([phase]), STAND, BACK, FWD, W, K, R)[0]


def test_kick_target_keypoints():
    assert torch.allclose(_t(0.0), STAND, atol=1e-6)          # début: STAND
    assert torch.allclose(_t(W), BACK, atol=1e-6)             # fin armement: BACK
    assert torch.allclose(_t(K), FWD, atol=1e-6)              # fin frappe: FORWARD
    assert torch.allclose(_t(R), STAND, atol=1e-6)            # fin retour: STAND
    assert torch.allclose(_t(0.9), STAND, atol=1e-6)          # repos: STAND


def test_kick_target_midsegments():
    assert torch.allclose(_t(W / 2), 0.5 * BACK, atol=1e-6)                    # mi-armement
    assert torch.allclose(_t((W + K) / 2), 0.5 * (BACK + FWD), atol=1e-6)      # mi-frappe
    assert torch.allclose(_t((K + R) / 2), 0.5 * FWD, atol=1e-6)              # mi-retour


def test_kick_target_batch_shape():
    phase = torch.linspace(0.0, 1.0, 50)
    out = kick_pose_target(phase, STAND, BACK, FWD, W, K, R)
    assert out.shape == (50, 2)
    # chaque composante reste dans l'enveloppe des 3 poses
    lo = torch.minimum(torch.minimum(STAND, BACK), FWD)
    hi = torch.maximum(torch.maximum(STAND, BACK), FWD)
    assert (out >= lo - 1e-6).all() and (out <= hi + 1e-6).all()
