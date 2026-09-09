from mjlab.tasks.velocity import mdp

from src.task_velocity_rollers import make_microduck_velocity_rollers_env_cfg
from src.task_velocity_swizzle import make_microduck_velocity_swizzle_env_cfg


def test_swizzle_head_control_wired():
    cfg = make_microduck_velocity_swizzle_env_cfg()
    roller_cfg = make_microduck_velocity_rollers_env_cfg()

    assert "head_pose" in cfg.commands

    # head_command carries the REAL command here, not the usual zero padding.
    for group in ("actor", "critic"):
        term = cfg.observations[group].terms["head_command"]
        assert term.func is mdp.generated_commands
        assert term.params["command_name"] == "head_pose"

    assert "head_pose_tracking" in cfg.rewards

    # Both HOME-pullers that would fight the head command must be handled: drop
    # neck_joint_pos_l2, and scope the pose reward away from neck/head.
    assert "neck_joint_pos_l2" not in cfg.rewards
    pose_joints = cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert any("(?!" in j and "neck" in j and "head" in j for j in pose_joints), f"pose reward not scoped away from neck/head: {pose_joints}"

    assert cfg.rewards["pose"].func is roller_cfg.rewards["pose"].func, "pose reward function was swapped (should only scope asset_cfg)"

    assert "head_pose_tracking_weight" in cfg.curriculum
    assert "head_pose_range" in cfg.curriculum
