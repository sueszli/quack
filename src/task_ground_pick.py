# Symmetry — disabled for v1.5: SYMMETRY_CFG's _OBS_PERM is hardcoded for the
# old 51D obs layout and breaks on the new 61D obs (which includes the
# head_command/body_command padding). All v1.5 envs run with symmetry off
# until SYMMETRY_CFG gets rewritten for the new obs structure.
ENABLE_SYMMETRY = False

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import task_dr
from . import task_mdp as microduck_mdp
from .robot import MICRODUCK_GROUND_PICK_ROBOT_CFG
from .task_symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg
from .task_velocity import HEAD_BODY_NAMES, LOCAL_CHECKPOINTS_ONLY, MICRODUCK_ROUGH_TERRAINS_CFG

DR = dataclasses.replace(task_dr.DEFAULT_DR, push_range=(-0.15, 0.15), push_play_interval_s=(2.0, 4.0))
ENCODER_BIAS_RANGE = DR.encoder_bias_range

# Durations at GP_PERIOD = 4 s:
#   descent    [0, DESCENT_END)        1.5 s  transition STAND->low
#   low hold   [DESCENT_END, HOLD_END) 0.2 s  brush (short)
#   rise       [HOLD_END, RISE_END)    1.5 s  transition low->STAND
#   rest       [RISE_END, 1)           0.8 s  standing
# ⚠️ RISE_END=0.80 > the φ=0.7 cutoff of the infer script: the rise is
# only complete if the slot plays up to φ~1.0 (the whole period). Check the
# actual runtime window.  ⚠️ --ground-pick-period at deployment = 4.0.
GP_PERIOD = 4.0
DESCENT_END = 0.375
HOLD_END = 0.425
RISE_END = 0.80


def make_microduck_ground_pick_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)

    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    head_impact_cfg = ContactSensorCfg(name="head_impact_contact", primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("force",), reduce="netforce", num_slots=1)

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_GROUND_PICK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, head_impact_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # No NeckOffsetJointPositionAction — head joints are part of the task motion

    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",  # replaced by phase-conditioned ground_pick_return_pose
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # target_height=0 pulls the mouth towards the ground; std=0.10 gives gradient from
    # ~20 cm (from the standing stance). The "WITHOUT TOUCHING" is ensured by
    # head_impact_penalty (strong) below -> the equilibrium is the mouth just
    # above the ground. Weight raised 2.0 -> 3.0 to pull closer.
    cfg.rewards["mouth_ground_proximity"] = RewardTermCfg(func=microduck_mdp.mouth_ground_proximity_phased, weight=3.0, params={"asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]), "std": 0.10, "target_height": 0.0, "command_name": "twist", "descent_end": DESCENT_END, "hold_end": HOLD_END, "rise_end": RISE_END})

    cfg.rewards["mouth_perpendicular_to_ground"] = RewardTermCfg(func=microduck_mdp.mouth_perpendicular_phased, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]), "command_name": "twist", "descent_end": DESCENT_END, "hold_end": HOLD_END, "rise_end": RISE_END})

    _LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    cfg.rewards["ground_pick_return_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,  # 4->6: reinforces the leg extension when standing up
        params={"std": 0.3, "command_name": "twist", "joint_indices": _LEG_JOINTS, "hold_end": HOLD_END, "rise_end": RISE_END},
    )

    # Return phase — neck/head (joints 5-8): tight std to prevent backward overshoot
    # and head-body collision (head geoms have no collision mesh, so self_collisions
    # can't catch it — the pose reward is the only guard).
    _NECK_JOINTS = [5, 6, 7, 8]
    cfg.rewards["ground_pick_return_pose_neck"] = RewardTermCfg(func=microduck_mdp.ground_pick_return_pose_phased, weight=6.0, params={"std": 0.15, "command_name": "twist", "joint_indices": _NECK_JOINTS, "hold_end": HOLD_END, "rise_end": RISE_END})

    cfg.rewards["return_upright"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_upright_phased,
        weight=4.0,  # 2->4: helps trunk balance more strongly when standing up
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.4, "command_name": "twist", "hold_end": HOLD_END, "rise_end": RISE_END},
    )

    cfg.rewards["neck_vel_descent"] = RewardTermCfg(func=microduck_mdp.neck_vel_descent_penalty, weight=-0.1, params={"command_name": "twist", "joint_indices": _NECK_JOINTS, "hold_end": HOLD_END})

    # Weight-0 reward: serves as a per-step hook that applies the object's WEIGHT
    # as an external force at mouth_tip, gated on the rise (phase >= HOLD_END).
    cfg.rewards["mouth_payload_force"] = RewardTermCfg(func=microduck_mdp.apply_mouth_payload_force, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["jaw_soft"], site_names=["mouth_tip"]), "command_name": "twist", "hold_end": HOLD_END})

    # Upright: reduced weight — the robot needs to lean forward during approach.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 0.2

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["soft_landing"].weight = -1e-5

    cfg.rewards["feet_grounded"] = RewardTermCfg(func=microduck_mdp.feet_grounded_reward, weight=3.0, params={"sensor_name": feet_ground_cfg.name})

    # FLAT feet. feet_grounded only sees CONTACT (found per foot): a foot
    # that PIVOTS on the ankle (tips onto its edge/tip) while keeping one contact
    # point slips through -> "it rolls its foot over". feet_flat_penalty
    # projects gravity into the foot site frame: when flat the site Z is
    # vertical (xy²≈0); any tipping -> xy²>0. Thus forbids rolling the
    # foot over on the ankle axis.
    cfg.rewards["feet_flat"] = RewardTermCfg(func=microduck_mdp.feet_flat_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", site_names=["left_foot", "right_foot"])})

    # Deliberately kept heavier than the velocity env: the ground-pick motion is
    # slow and precise, so strong smoothness aids transfer (unlike the dynamic
    # standup recovery, where heavy regularisation blocked the motion).

    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-2.0)

    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(func=microduck_mdp.neck_action_rate_l2, weight=-1.0)

    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=microduck_mdp.joint_torques_l2, weight=-5e-3)

    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name})

    cfg.rewards["head_impact_penalty"] = RewardTermCfg(func=microduck_mdp.body_impact_cost, weight=-2.0, params={"sensor_name": head_impact_cfg.name, "threshold": 1.0})

    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    # Ground-pick has no terrain-height sensor (and drops the walking foot
    # rewards), so remove these terms.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    microduck_mdp.wire_sim2real_obs(cfg, imu_delay_max_lag=3, imu_misalignment_deg=DR.imu_orientation_angle_deg if DR.imu_orientation else None, encoder_bias_range=DR.encoder_bias_range if DR.encoder_bias else None, sanitize_critic_sensors=False)

    # Ground-pick doesn't use head/body pose commands (the head is driven by the
    # task's phase motion), but all microduck policies share the same 61D obs
    # shape so the runtime can feed a single command buffer. The 10 trailing
    # slots (head 4 + body 6) are constant zero.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 4})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 6})

    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(**{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand, "period": GP_PERIOD})

    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")

    cfg.events["sample_mouth_payload"] = EventTermCfg(func=microduck_mdp.sample_mouth_payload, mode="reset", params={"min_kg": 0.01, "max_kg": 0.04})
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    task_dr.apply_dr(cfg, DR, HEAD_BODY_NAMES, play=play)

    # NOTE: IMU mounting-misalignment is applied at the OBSERVATION level above
    # (matching velocity) — the old event-based randomize_imu_orientation wrote
    # site_quat, a no-op under mjlab 1.3.0.

    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Action-rate curriculum: warm up light so the gross reaching motion can form,
    # then clamp down HARD (-2.0, heavier than velocity's -1.0) for smoothness.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.8}, {"step": 250 * 24, "weight": -1.5}, {"step": 500 * 24, "weight": -2.0}]})

    if DR.com:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}, {"step": 2000 * 24, "range": 0.02}]})

    if DR.head_com:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    return cfg


MicroduckGroundPickRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches velocity; normalizer baked into ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None),
    logger=LOCAL_CHECKPOINTS_ONLY,
    experiment_name="ground_pick",
    run_name="ground_pick",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
