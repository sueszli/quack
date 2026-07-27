from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
)
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand


def test_ground_pick_cfg_builds_with_pose_rewards():
    cfg = make_microduck_ground_pick_env_cfg()
    rewards = cfg.rewards
    # Suivi de pose splitté tête/jambes (relevage séquencé).
    assert "phase_pose_track_head" in rewards
    assert "phase_pose_track_legs" in rewards
    assert "phase_pose_track_l1_head" in rewards
    assert "phase_pose_track_l1_legs" in rewards
    assert rewards["phase_pose_track_head"].weight == 2.0
    assert rewards["phase_pose_track_legs"].weight == 4.0
    # tête remonte AVANT les jambes : rise_end tête < rise_end jambes
    assert (
        rewards["phase_pose_track_head"].params["rise_end"]
        < rewards["phase_pose_track_legs"].params["rise_end"]
    )
    # mouth_ground_proximity retiré : il poussait la bouche dans le sol (slam) —
    # le positionnement est porté par le suivi de pose (pose DOWN réelle).
    assert "mouth_ground_proximity" not in rewards
    # anciennes mécaniques retirées
    assert "mouth_perpendicular_to_ground" not in rewards
    assert "ground_pick_return_pose_legs" not in rewards
    assert "ground_pick_return_pose_neck" not in rewards


def test_ground_pick_cfg_command_is_phase_no_randomize():
    cfg = make_microduck_ground_pick_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is GroundPickPhaseCommand
    assert cmd.period == 6.0
    assert cmd.randomize_phase is False


def test_ground_pick_rough_variant_builds():
    cfg = make_microduck_ground_pick_env_cfg(rough=True)
    assert "phase_pose_track_legs" in cfg.rewards


def test_ground_pick_play_variant_builds():
    cfg = make_microduck_ground_pick_env_cfg(play=True)
    assert cfg.commands["twist"].randomize_phase is False
