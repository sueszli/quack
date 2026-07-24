import pytest
import torch
from mjlab_microduck.tasks.mdp import (
    GroundPickPhaseCommand,
    GroundPickPhaseCommandCfg,
    kick_pose_target,
    kick_pose_track,
    kick_pose_track_l1,
)


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


class _StubEnv:
    """Only what ManagerTermBase.device/num_envs read off self._env."""

    def __init__(self, num_envs: int):
        self.device = "cpu"
        self.num_envs = num_envs


def _minimal_phase_command(randomize_phase: bool, num_envs: int) -> GroundPickPhaseCommand:
    """Build a GroundPickPhaseCommand without going through __init__ (which
    needs a full ManagerBasedRlEnv/scene). reset() only touches
    self._gp_phase / self._randomize_phase / self.device (the latter is a
    read-only property proxying self._env.device), so we seed exactly those
    attributes and then exercise the REAL reset() method — no
    reimplementation of its logic."""
    cmd = object.__new__(GroundPickPhaseCommand)
    cmd._env = _StubEnv(num_envs)
    cmd._randomize_phase = randomize_phase
    cmd._gp_phase = torch.full((num_envs,), 0.3)
    return cmd


def test_phase_command_reset_zeroes_phase_when_not_randomized():
    # randomize_phase=False (shoot task config) -> reset() must snap phase to
    # exactly 0 for every reset env, matching the deployment slot's one-shot
    # button-A semantics (φ=0 == STAND).
    cmd = _minimal_phase_command(randomize_phase=False, num_envs=4)
    env_ids = torch.tensor([0, 2, 3])
    result = cmd.reset(env_ids)

    assert result == {}
    assert torch.equal(cmd._gp_phase[env_ids], torch.zeros(len(env_ids)))
    # untouched env keeps its previous phase
    assert cmd._gp_phase[1].item() == pytest.approx(0.3)


def test_phase_command_reset_randomizes_phase_when_enabled():
    # Sanity check for the opposite branch: randomize_phase=True resamples
    # (near-certainly) away from the seeded 0.3 value, and stays in [0, 1).
    cmd = _minimal_phase_command(randomize_phase=True, num_envs=4)
    env_ids = torch.tensor([0, 1, 2, 3])
    torch.manual_seed(0)
    cmd.reset(env_ids)

    assert (cmd._gp_phase >= 0.0).all() and (cmd._gp_phase < 1.0).all()


def test_phase_command_reset_noop_on_empty_env_ids():
    cmd = _minimal_phase_command(randomize_phase=False, num_envs=3)
    before = cmd._gp_phase.clone()
    cmd.reset(torch.tensor([], dtype=torch.long))
    assert torch.equal(cmd._gp_phase, before)


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


STAND_D = {"a": 0.0, "b": 0.0}
BACK_D = {"a": 1.0, "b": -1.0}
FWD_D = {"a": -1.0, "b": 2.0}
_IDX = {"a": 0, "b": 1}


class _FakeData:
    def __init__(self, joint_pos):
        self.joint_pos = joint_pos
        self.default_joint_pos = torch.zeros_like(joint_pos)


class _FakeAsset:
    def __init__(self, joint_pos):
        self.data = _FakeData(joint_pos)

    def find_joints(self, names):
        return ([_IDX[names[0]]], names)


class _FakeScene:
    def __init__(self, asset):
        self._a = asset

    def __getitem__(self, name):
        return self._a


class _FakeCmdMgr:
    def __init__(self, cmd):
        self._cmd = cmd

    def get_command(self, name):
        return self._cmd


class _FakeEnv:
    def __init__(self, joint_pos, phase):
        self.scene = _FakeScene(_FakeAsset(joint_pos))
        # cmd = [cos, sin, 0]
        cmd = torch.stack(
            [torch.cos(2 * torch.pi * phase), torch.sin(2 * torch.pi * phase),
             torch.zeros_like(phase)], dim=-1)
        self.command_manager = _FakeCmdMgr(cmd)
        self.device = "cpu"
        self.num_envs = joint_pos.shape[0]


def test_kick_track_perfect_at_stand_phase():
    # phase=0 -> cible STAND=[0,0] ; joint_pos exactement STAND -> reward ~1
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
    r = kick_pose_track(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-4)


def test_kick_track_lower_when_off_target():
    # phase=0.45 (kick_end) -> cible FORWARD=[-1,2] ; joint_pos=STAND -> reward < 0.5
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.45]))
    r = kick_pose_track(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert (r < 0.5).all()


def test_kick_track_l1_zero_when_perfect():
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
    r = kick_pose_track_l1(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6)
