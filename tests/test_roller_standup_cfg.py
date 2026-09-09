import pytest

from src.task_roller_standup import EPISODE_LENGTH_S, make_microduck_roller_standup_env_cfg
from src.task_velocity_rollers import make_microduck_velocity_rollers_env_cfg

SKATING_REWARDS = ("wheel_speed", "braking", "skating_air_time", "glide", "single_support", "gait_symmetry", "forward_lean", "heading_hold", "feet_flat", "hip_roll_neutral", "pose", "com_height_target", "upright")


def test_env_builds_train_and_play():
    assert make_microduck_roller_standup_env_cfg() is not None
    assert make_microduck_roller_standup_env_cfg(play=True) is not None


def test_episode_is_short():
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0


def test_no_skating_rewards_survive():
    cfg = make_microduck_roller_standup_env_cfg()
    for name in SKATING_REWARDS:
        assert name not in cfg.rewards, f"surviving skating reward: {name}"


def test_smoothness_regularisers_kept():
    # body_ang_vel must stay LIGHT: at -0.15 it froze the stand-up (standup).
    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("action_over_limit", "self_collisions", "body_ang_vel", "angular_momentum", "action_rate_l2", "neck_action_rate_l2", "neck_joint_pos_l2", "joint_torques_l2"):
        assert name in cfg.rewards, f"lost regularizer: {name}"
    assert cfg.rewards["body_ang_vel"].weight == -0.05


def test_twist_command_is_neutralised():
    # Deployed as --standing: the runtime leaves the twist slot at zero.
    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_twist_command_is_not_heading_relative():
    # The roller env's RelativeHeadingVelocityCommandCfg makes cmd[2] a heading
    # error; here cmd[2] must be a true noisy zero.
    from src import task_mdp as microduck_mdp

    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert not isinstance(cmd, microduck_mdp.RelativeHeadingVelocityCommandCfg)


def test_obs_nan_policy_sanitize():
    # A rare contact diverges the free joint to NaN: sanitize, don't kill training.
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"


def test_obs_parity_with_roller_env():
    # Parity is mandatory: otherwise the ONNX does not load into a runtime slot.
    standup = make_microduck_roller_standup_env_cfg()
    roller = make_microduck_velocity_rollers_env_cfg()
    for grp in ("actor", "critic"):
        assert list(standup.observations[grp].terms.keys()) == list(roller.observations[grp].terms.keys()), f"observation layout diverges on group {grp}"


def test_terrain_is_plain_plane():
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import src.task_registry  # noqa: F401  (import registers the tasks)

    assert "Mjlab-RollerStandUp-Flat-MicroDuck" in list_tasks()


def test_joint_indices_match_actual_roller_model():
    """Passive wheels interleave: standup's indices ([0-4, 9-13]) would aim
    rewards at wheels."""
    import mujoco

    from src.robot import get_walk_rollers_spec
    from src.task_roller_standup import _LEG_JOINTS, _NECK_JOINTS, _WHEEL_JOINTS

    model = get_walk_rollers_spec().compile()
    articulated = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]

    assert [articulated[i] for i in _LEG_JOINTS] == ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
    assert [articulated[i] for i in _NECK_JOINTS] == ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    assert [articulated[i] for i in _WHEEL_JOINTS] == ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]
    assert len(set(_LEG_JOINTS) | set(_NECK_JOINTS) | set(_WHEEL_JOINTS)) == len(articulated)


def test_recovery_rewards_present_with_expected_weights():
    cfg = make_microduck_roller_standup_env_cfg()
    expected = {
        "pose_stand_legs": 8.0,
        "pose_stand_l1": 5.0,
        "height_stand": 4.0,
        "height_stand_sharp": 4.0,
        "height_stand_l1": 30.0,
        "com_upward_velocity": 3.0,
        # POSITIVE: trunk_vertical_accel_penalty already returns -|a_z|.
        "gentle_rise": +0.02,
        "upright_linear": 6.0,
        "upright_sharp": 6.0,
        "standing_composite": 15.0,
        "joint_torque_rate_l2": -0.2,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, f"missing stand-up reward: {name}"
        assert cfg.rewards[name].weight == weight, f"unexpected weight on {name}"


def test_recovery_rewards_use_roller_heights_not_walker_heights():
    from src.task_roller_standup import ROLLER_PRONE_Z, ROLLER_STAND_Z

    cfg = make_microduck_roller_standup_env_cfg()
    assert ROLLER_STAND_Z == 0.138  # wheels add 23 mm over the wheel-less 0.115
    for name in ("height_stand", "height_stand_sharp", "height_stand_l1"):
        assert cfg.rewards[name].params["target_height"] == ROLLER_STAND_Z
    assert cfg.rewards["standing_composite"].params["target_height"] == ROLLER_STAND_Z
    # Cut off 10 mm ABOVE the target, else the policy parks at the cutoff altitude.
    assert cfg.rewards["com_upward_velocity"].params["max_height"] == ROLLER_STAND_Z + 0.010
    assert cfg.rewards["upright_sharp"].params["height_low"] == ROLLER_PRONE_Z
    assert cfg.rewards["upright_sharp"].params["height_high"] == ROLLER_STAND_Z


def test_pose_rewards_target_legs_only_at_roller_indices():
    from src.task_roller_standup import _LEG_JOINTS

    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("pose_stand_legs", "pose_stand_l1", "standing_composite"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        # None → the target is HOME (default_joint_pos).
        assert cfg.rewards[name].params["target_overrides"] is None


def test_trunk_asset_cfgs_are_distinct_objects():
    """mjlab resolves and MUTATES SceneEntityCfg in place: sharing one object
    between terms leaves stale indices."""
    cfg = make_microduck_roller_standup_env_cfg()
    names = ("height_stand", "height_stand_sharp", "height_stand_l1", "com_upward_velocity", "gentle_rise", "upright_linear", "upright_sharp", "standing_composite")
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg shared between several terms"


def test_starts_from_ground_states():
    # No sitting bucket: standup's sitting_joint_overrides index the wheel-less model.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "set_ground_state" in cfg.events
    params = cfg.events["set_ground_state"].params
    assert params["sitting_prob"] == 0.0
    assert params["sitting_joint_overrides"] is None
    assert params["face_down_prob"] > 0.0
    assert params["standing_prob"] > 0.0
    # The back is introduced late by the curriculum.
    assert params["face_up_prob"] == 0.0


def test_ground_state_heights_are_roller_specific():
    cfg = make_microduck_roller_standup_env_cfg()
    params = cfg.events["set_ground_state"].params
    # Belly and back share one z range; the belly is the binding side.
    assert (params["prone_z_min"], params["prone_z_max"]) == (0.076, 0.09)
    # Below 0.0752 (measured belly contact, HOME pose) the spawn is INSIDE the floor,
    # and the pushout is charged to gentle_rise / joint_torque_rate_l2.
    assert params["prone_z_min"] >= 0.0752
    assert params["standing_z_min"] == 0.134
    assert params["standing_z_max"] == 0.144
    assert params["standing_z_min"] < 0.138 < params["standing_z_max"]


def test_ground_state_event_runs_after_base_reset():
    # It overwrites reset_base / reset_robot_joints, and event order is insertion order.
    cfg = make_microduck_roller_standup_env_cfg()
    order = list(cfg.events.keys())
    assert order.index("set_ground_state") > order.index("reset_base")
    assert order.index("set_ground_state") > order.index("reset_robot_joints")


def test_no_fall_termination():
    # The robot STARTS fallen: a tilt termination would kill step 1.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_ground_state_curriculum_ramps_easy_to_hard():
    cfg = make_microduck_roller_standup_env_cfg()
    assert "ground_state_mix" in cfg.curriculum
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert cfg.curriculum["ground_state_mix"].params["event_name"] == "set_ground_state"
    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)
    face_up = [s["params"]["face_up_prob"] for s in stages]
    assert face_up[0] == 0.0
    assert face_up == sorted(face_up)
    assert face_up[-1] >= 0.35
    # "Already standing" must never disappear, or the policy never learns to hold.
    for stage in stages:
        p = stage["params"]
        total = p["standing_prob"] + p["sitting_prob"] + p["face_down_prob"] + p["face_up_prob"]
        assert abs(total - 1.0) < 1e-9
        assert p["sitting_prob"] == 0.0
        assert p["standing_prob"] > 0.0


def test_wheel_friction_curriculum_is_decreasing():
    """Wheels BRAKED → FREE: rolling wheels give no longitudinal grip, so we
    bootstrap on near-locked bearings. The roller env ramps the other way."""
    cfg = make_microduck_roller_standup_env_cfg()
    stages = cfg.curriculum["wheel_friction"].params["ranges_stages"]
    assert cfg.curriculum["wheel_friction"].params["event_name"] == "randomize_wheel_friction"

    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)

    lows = [s["ranges"][0] for s in stages]
    assert lows == sorted(lows, reverse=True), "friction must DECREASE"
    assert lows[0] >= 0.02, "start firmly braked to bootstrap the motion"
    assert stages[-1]["ranges"] == (0.0015, 0.0015)  # the true bearing value
    for stage in stages:
        assert stage["ranges"][0] == stage["ranges"][1]


def test_wheel_friction_event_default_matches_stage_zero():
    # The curriculum runs before the reset events, so this default is never read;
    # keep it consistent with stage 0 in case the curriculum is ever dropped.
    cfg = make_microduck_roller_standup_env_cfg()
    stage0 = cfg.curriculum["wheel_friction"].params["ranges_stages"][0]["ranges"]
    assert cfg.events["randomize_wheel_friction"].params["ranges"] == stage0


def test_action_rate_ramp_is_the_standup_one_not_the_roller_one():
    # The roller env's -2.0 is a motion blocker: getting up from the back needs
    # fast action, so reuse standup's ramp, capped at -1.0.
    cfg = make_microduck_roller_standup_env_cfg()
    weights = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert weights == [-0.4, -0.8, -1.0]
    assert cfg.rewards["action_rate_l2"].weight == -0.6


def test_push_curriculum_ramps_from_zero():
    # Ramped: a shove from step 0 disturbs the stand-up bootstrap.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "push_robot" in cfg.events
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert cfg.curriculum["push_magnitude"].params["event_name"] == "push_robot"
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)
    assert stages[-1]["velocity_range"]["x"] == (-0.2, 0.2)
    highs = [s["velocity_range"]["x"][1] for s in stages]
    assert highs == sorted(highs), "push must INCREASE"


def test_inherited_dr_curricula_survive():
    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("com_range", "head_com_range"):
        assert name in cfg.curriculum, f"lost DR curriculum: {name}"
    for name in ("randomize_com", "randomize_head_com", "randomize_armature", "randomize_joint_friction", "randomize_mass_inertia", "randomize_wheel_friction", "encoder_bias"):
        assert name in cfg.events, f"lost DR event: {name}"


# STANDUP_PLAY_FACE_UP exists because a play env is rebuilt from scratch:
# common_step_counter restarts at 0, so the curriculum applies stage 0 and a
# play would never show the hardest case, a start on the back.


def test_play_face_up_override_forces_back_starts(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    params = cfg.events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0
    assert params["face_down_prob"] == 0.0
    assert params["standing_prob"] == 0.0
    # The curriculum runs before the reset events and would rewrite these.
    assert "ground_state_mix" not in cfg.curriculum


def test_play_face_up_override_splits_remainder_like_final_stage(monkeypatch):
    # The remainder is split in the final stage's 2:1 belly:standing ratio.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "0.4")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == pytest.approx(0.40)
    assert params["face_down_prob"] == pytest.approx(0.40)
    assert params["standing_prob"] == pytest.approx(0.20)
    total = params["face_up_prob"] + params["face_down_prob"] + params["standing_prob"]
    assert total == pytest.approx(1.0)


def test_play_face_up_override_is_clamped(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "3.0")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0


def test_play_face_up_override_ignored_during_training(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=False)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_without_override_keeps_curriculum_mix(monkeypatch):
    monkeypatch.delenv("STANDUP_PLAY_FACE_UP", raising=False)
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "bogus")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_none_keyword_disables(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "none")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_already_negative_penalties_use_positive_weights():
    """task_mdp.py mixes two sign conventions: cost functions return a positive
    magnitude (negative weight), self-negating ones return ≤ 0 (POSITIVE weight).
    A negative weight on the latter double-negates into a reward for violence."""
    cfg = make_microduck_roller_standup_env_cfg()
    # height_l1_penalty, pose_l1_penalty, trunk_vertical_accel_penalty.
    for name in ("height_stand_l1", "pose_stand_l1", "gentle_rise"):
        assert cfg.rewards[name].weight > 0, f"{name} calls a function that already returns a negative value: a negative weight would turn it into a reward"
    for name in ("joint_torques_l2", "joint_torque_rate_l2", "action_rate_l2"):
        assert cfg.rewards[name].weight < 0, f"{name} expects a negative weight"


def test_no_ungated_head_impact_penalty():
    """Getting up from the back PIVOTS on the head: the head is the fulcrum, not
    collateral damage. At -1.0 the policy stayed lying down, inert. Any
    replacement must be HEIGHT-GATED so it spares the ground phase."""
    cfg = make_microduck_roller_standup_env_cfg()
    assert "head_impact_penalty" not in cfg.rewards
    assert "head_impact_contact" not in [s.name for s in cfg.scene.sensors]


def test_inherited_sensors_intact():
    cfg = make_microduck_roller_standup_env_cfg()
    names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in names
    assert "self_collision" in names


def test_lazy_prone_optimum_is_documented_risk():
    """Lying down keeps the legs at HOME, so pose_stand_legs pays ~full price for
    doing nothing (+7.72 of 8 measured). height_stand_l1 is the term that makes
    staying on the floor net negative: it must stay strong."""
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.rewards["height_stand_l1"].weight >= 30.0
    assert cfg.rewards["com_upward_velocity"].weight > 0.0


def test_damping_terms_are_not_numerically_negligible():
    """At -2e-3 the dampers measured -0.0002/step against +41.6 of task reward.
    joint_torque_rate_l2 is the safe lever to raise: it prices torque VARIATION,
    not motion, unlike body_ang_vel and action_rate which froze the stand-up."""
    cfg = make_microduck_roller_standup_env_cfg()
    assert abs(cfg.rewards["joint_torque_rate_l2"].weight) >= 0.1
    assert cfg.rewards["body_ang_vel"].weight == -0.05
    weights = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert min(weights) >= -1.0, "action_rate beyond -1.0 froze the stand-up (standup)"
