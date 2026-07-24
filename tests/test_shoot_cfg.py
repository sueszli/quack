import re

import pytest

from mjlab_microduck.tasks.microduck_shoot_env_cfg import (
    make_microduck_shoot_env_cfg,
    STAND_POSE, KICK_BACK_POSE, KICK_FWD_POSE, SHOOT_PERIOD,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.robot.microduck_constants import HOME_FRAME


def test_poses_have_same_14_keys():
    assert set(STAND_POSE) == set(KICK_BACK_POSE) == set(KICK_FWD_POSE)
    assert len(STAND_POSE) == 14
    assert "mouth" not in STAND_POSE


def test_stand_pose_matches_home_standing_pose():
    """Reset-coherence invariant: at phase=0 the STAND target must equal the
    robot's HOME/reset standing joint configuration, or the support leg gets
    pulled off its stance the instant the episode starts."""
    for joint_name, target_value in STAND_POSE.items():
        matches = [
            value
            for regex, value in HOME_FRAME.joint_pos.items()
            if re.match(regex, joint_name)
        ]
        assert len(matches) == 1, (
            f"expected exactly one HOME_FRAME regex to match '{joint_name}', "
            f"got {len(matches)}"
        )
        assert target_value == pytest.approx(matches[0], abs=1e-3), (
            f"STAND_POSE['{joint_name}']={target_value} does not match "
            f"HOME_FRAME value {matches[0]}"
        )


def test_shoot_cfg_builds_with_phase_command():
    cfg = make_microduck_shoot_env_cfg()
    twist = cfg.commands["twist"]
    assert isinstance(twist, microduck_mdp.GroundPickPhaseCommandCfg)
    assert twist.randomize_phase is False
    assert twist.period == SHOOT_PERIOD


def test_shoot_cfg_has_kick_rewards_and_no_walking():
    cfg = make_microduck_shoot_env_cfg()
    assert "kick_pose_track" in cfg.rewards
    assert "kick_pose_l1" in cfg.rewards
    assert "support_foot_grounded" in cfg.rewards
    for gone in ("track_linear_velocity", "track_angular_velocity",
                 "mouth_ground_proximity", "ground_pick_return_pose_legs"):
        assert gone not in cfg.rewards
