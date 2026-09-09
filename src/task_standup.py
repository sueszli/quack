"""Microduck stand task — rise from the sitting keyframe to standing.

One fixed target rewarded from t=0: no trajectory waypoints and no
episode-progress gating, so the rise path is the policy's to discover.
Companion to the sit env, one policy per direction.

Once standing, a body_pose command steers a trunk delta [z, roll, pitch];
body control is introduced at iter 2500, after ground_state_mix finishes.
"""

import math
from copy import deepcopy

ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to the velocity env for sim2real parity) ────
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = False
ENABLE_KD_RANDOMIZATION = False
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

# ── Ranges (matched to the velocity env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE = 0.003  # ramped to 0.015 by the com_range curriculum
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ramped to 0.01 by head_com_range
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
# Ramped in by push_magnitude: unlike velocity (which starts standing), the
# sit-rise bootstrap must not be shoved around from step 0.
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
# The real IMU has ~5° systematic pitch error + estimator drift; 2° trained
# too narrow a band.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

EPISODE_LENGTH_S = 6.0

# ── Sitting source pose (joint_pos index → rad) ───────────────────────────────
# Must match the *actual end-state* of the sit policy (this reset IS the
# sit→stand hand-off) — keep in sync with task_sitstand.SITTING_TARGET_OVERRIDES.
# Neck/head omitted → they stay at HOME, where the sit policy converges.
# Indices are the clean 14-joint order (0-4 left leg, 5-8 neck/head, 9-13 right
# leg); the passive jaw joints are excluded from qpos.
SITTING_JOINT_OVERRIDES = {
    1: 0.0,  # left  hip_roll   (HOME -0.0873)
    2: -0.4079,  # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3: 1.35,  # left  knee       (HOME -0.0049)
    4: 0.0,  # left  ankle      (HOME +0.4530)
    10: 0.0,  # right hip_roll   (HOME +0.0873)
    11: 0.4079,  # right hip_pitch  (HOME +0.4579)
    12: -1.35,  # right knee       (HOME +0.0049)
    13: 0.0,  # right ankle      (HOME -0.4530)
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk heights (m), both MEASURED in sim — never carried across model
# revisions. SIT_Z is the seated equilibrium of the pose above (keep in sync
# with task_sitstand.py); STAND_Z is the trunk z at the natural HOME standing
# equilibrium. The old 0.120 was 5 mm above what HOME can reach and forced a
# back-lean compromise.
SIT_Z = 0.060
STAND_Z = 0.115

# ── Body pose command ─────────────────────────────────────────────────────────
# OFF ⇒ zero-padded body_command obs slot (obs stays 61D either way), no
# tracking reward, no body-control curricula.
ENABLE_BODY_CONTROL = True
# 6D slot [x, y, z, roll, pitch, yaw] for obs parity, but only z/roll/pitch are
# tracked — the axes the runtime exposes. x/y/yaw keep a tiny "alive" range
# forever so their input weights stay trained. z is ASYMMETRIC: STAND_Z is the
# HOME equilibrium, so there is crouch room below but only ~1 cm of leg
# extension above. Angles capped at ±15°: ±20° trained twitchy tilting.
BODY_CMD_MAX_Z_DOWN = 0.04  # m, crouch below STAND_Z
BODY_CMD_MAX_Z_UP = 0.030  # m, extend above STAND_Z
BODY_CMD_MAX_ANGLE = math.radians(15)  # rad, trunk pitch/roll
BODY_CMD_ALIVE_XY = 0.005  # m
BODY_CMD_ALIVE_ANGLE = 0.05  # rad
# Trains the deployment idle case ("stand at nominal") — uniform sampling
# never produces the all-zero command.
BODY_CMD_ZERO_PROB = 0.3

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import task_mdp as microduck_mdp
from .robot import MICRODUCK_STANDUP_ROBOT_CFG
from .task_symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg
from .task_velocity import BODY_POSE_CMD_RESAMPLE_S, HEAD_BODY_NAMES, HEAD_POSE_CMD_RESAMPLE_S, MICRODUCK_ROUGH_TERRAINS_CFG


def make_microduck_standup_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck stand environment configuration (sit-keyframe start)."""

    site_names = ["left_foot", "right_foot"]

    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)

    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in ["track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance", "foot_swing_height", "foot_slip", "pose"]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: single fixed target (HOME pose + STAND_Z), active from t=0 ──
    # All task weights here are velocity's ÷4 so the total task mass (~12)
    # matches velocity's (~11) and the shared regularisers bite at the same
    # RELATIVE strength. At the old ~49 task mass they were ~4× weaker and
    # jitter around the standing point was nearly free (the real robot came
    # out violent and shaky). Reward numbers quoted below are pre-÷4.

    # target_overrides=None → HOME.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(func=microduck_mdp.pose_target_match, weight=2.0, params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None})

    # The neck/head are steered by this command, so they are excluded from
    # pose_stand_l1 / standing_composite — nothing may fight its gradient.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(func=microduck_mdp.head_pose_tracking, weight=0.75, params={"command_name": "head_pose", "std": 0.5})

    # Head DC-droop penalty: L1 on a 1 s EMA of the tracking error, so only
    # the sustained gravity sag the policy can cancel is priced and transient
    # motion averages out. The upright gate (arrival_damping's values) zeroes
    # the error feeding the EMA while prone/rising, so the head-pivot flip is
    # never taxed — the retired head_impact_penalty froze the policy that way.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by head_pose_bias_weight curriculum
        params={"command_name": "head_pose", "tau_s": 1.0, "gate_height_low": 0.09, "gate_height_high": 0.11, "gate_tilt_full_deg": 20.0, "gate_tilt_zero_deg": 45.0},
    )

    # L1 bootstrap — constant gradient even when far from HOME. Sized (pre-÷4:
    # 5, not 2) so parking ~0.18 rad off-HOME on bent knees costs -0.9/step
    # rather than an ignorable -0.35/step. Legs only (see head_pose_tracking).
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(func=microduck_mdp.pose_l1_penalty, weight=1.25, params={"joint_indices": _LEG_JOINTS, "target_overrides": None})

    # Two-layer height target. With the wide layer alone runs converged at
    # z ≈ 0.109: already saturated (0.93/1.0), so nothing pulled the last cm.
    # The sharp layer adds a 0.36→1.0 jump over that range, ~3× the pull.
    cfg.rewards["height_stand"] = RewardTermCfg(func=microduck_mdp.height_target_gaussian, weight=1.0, params={"std": 0.04, "target_height": STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(func=microduck_mdp.height_target_gaussian, weight=1.0, params={"std": 0.015, "target_height": STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    # L1 sized (pre-÷4: 30, not 10) so "stay sitting" is net NEGATIVE: at 10
    # the static-sit basin was net positive and runs plateaued sitting still.
    cfg.rewards["height_stand_l1"] = RewardTermCfg(func=microduck_mdp.height_l1_penalty, weight=7.5, params={"target_height": STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Pays for the MOTION of rising, not just the destination: with destination
    # rewards only, "sit upright collecting most-of-pose" was the dominant
    # local optimum. max_height sits just above STAND_Z so the reward stays
    # active through the final cm — at 0.11 the policy parked at the gate-off
    # altitude (~0.108) and never finished the climb.
    # NO max_vz cap: any cap (0.15, 0.30 both tried) shrinks the payoff of
    # noisy recovery ATTEMPTS and face-up/face-down recovery never gets
    # learned. Smoothing comes from the late curricula instead.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(func=microduck_mdp.com_upward_velocity, weight=0.75, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "max_height": 0.125})

    # Pairs with com_upward_velocity: constant positive vz collects that reward
    # AND has a_z = 0, so together they select a smooth constant-velocity rise.
    # GLOBAL (not phase-gated), so prone flips pay it in full. 0.005 is the
    # ceiling unless it gains a height/tilt gate like arrival_damping — 0.01
    # contributed to the face-up freeze.
    # ⚠️ POSITIVE weight: trunk_vertical_accel_penalty already returns -|a_z|,
    # so a negative weight double-negates into a reward for vertical shocks.
    # Keep the magnitude small: |a_z| is unavoidable during prone flips.
    cfg.rewards["gentle_rise"] = RewardTermCfg(func=microduck_mdp.trunk_vertical_accel_penalty, weight=0.005, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Arrival damper — trunk ω_xy², gated on height AND tilt (zero above 45°
    # tilt / below 0.09 m, full below 20° tilt / above 0.11 m). Targets the
    # real-robot failure loop: rise → overshoot vertical → tip → retry.
    #
    # Introduced at iter 3000: any attempt-tax active during recovery DISCOVERY
    # (ground_state_mix ramps prone spawns until 2500) makes the noisy flip
    # attempts net-negative and "do nothing" wins. Gate/magnitude refinements
    # did not change that failure signature — the fix is timing.
    cfg.rewards["arrival_damping"] = RewardTermCfg(func=microduck_mdp.body_ang_vel_at_height, weight=0.0, params={"height_low": 0.09, "height_high": 0.11, "tilt_full_deg": 20.0, "tilt_zero_deg": 45.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Two-layer upright. cos(tilt) has a strong gradient when inverted but runs
    # out of steam near vertical; without the sharp layer runs converged at a
    # ~37° back-lean.
    cfg.rewards["upright_linear"] = RewardTermCfg(func=microduck_mdp.body_upright_linear, weight=1.5, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    # z-gated so it pays only at standing height — blocks the "crouch low and
    # vertical" exploit. std 0.3 (≈17°), not 0.1: tighter scored near-zero at
    # the lean basin (no gradient); at 0.3 that basin scores ~0.1.
    cfg.rewards["upright_sharp"] = RewardTermCfg(func=microduck_mdp.upright_gaussian_at_height, weight=1.5, params={"std": 0.3, "height_low": SIT_Z, "height_high": STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Multiplicative goal score — a lean that satisfies 2 of 3 factors pays
    # nothing. Stds are BROAD on purpose: at tight stds (0.015/0.15/0.20) the
    # product was ~5e-5 at the lean basin, i.e. zero gradient.
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=3.75,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,  # ≈ 23° — lean basin scores ~0.3
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Locomotion variant (not body_pose_tracking_6d) so the unused x/y axes
    # don't reference the spawn origin, which the robot leaves during prone
    # flips. Ramped in from iter 2500. While prone/rising it is ≈0 on all
    # tracked axes, so unlike a motion penalty it cannot tax flip attempts.
    # Tight stds: at 1 cm z error z_std=0.01 gives 0.37 vs 0.78 at 0.02.
    if ENABLE_BODY_CONTROL:
        cfg.rewards["body_pose_tracking"] = RewardTermCfg(func=microduck_mdp.body_pose_tracking_locomotion, weight=0.0, params={"command_name": "body_pose", "nominal_height": STAND_Z, "z_std": 0.01, "angle_std": math.radians(5), "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 0.0), "vel_gate_command_name": None})

    # ── Sim2real regularisers — velocity's exact set and absolute weights ───
    # RISK: at the old (×4) task scale, body_ang_vel -0.15 and an action_rate
    # end of -1.2 killed back-recovery (both block the flip). -0.05 here ≈
    # -0.2 in old units — watch prone recovery as ground_state_mix ramps it in
    # (iters 600–2500). If recovery freezes: halve body_ang_vel first, then
    # soften the action_rate curriculum end to -0.6.
    # joint_torque_rate_l2 and arrival_damping start at 0 and come in at iter
    # 3000 (see arrival_damping: timing, not magnitude, protects discovery).
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05  # motion-blocker: kept LIGHT
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name})

    # Replaced by upright_linear/upright_sharp above.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / sit policies) ─────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    # No terrain-height sensor in this env. foot_air_time/foot_contact(_forces)
    # use feet_ground_contact, which standup does define, so they stay.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # NaN-safe wrappers: a non-finite contact force slips past robot_state_is_nan
    # (joint + root state only) and one NaN kills the run via rsl_rl's check_nan.
    # Standup lands and flips constantly, so degenerate contacts are likely.
    for _term, _safe in (("foot_contact_forces", microduck_mdp.foot_contact_forces_safe), ("foot_air_time", microduck_mdp.foot_air_time_safe)):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])

    # IMU obs delay: max_lag 1 — the real dxl IMU path is fast (±20 ms).
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Obs noise matched to the velocity env.
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (match velocity): per-env constant rotation of
    # the IMU-derived actor obs; critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Deepcopy joint_pos/joint_vel per group (they share base-template objects) so
    # the encoder-bias `biased` flag below applies to the actor only.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR (match velocity): actor sees joint_pos + per-env bias; critic
    # keeps the true joint pos. Requires the base-template encoder_bias event.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Head pose command ─────────────────────────────────────────────────────
    # 4D deltas-from-HOME on the neck/head joints; ranges widened by the
    # head_pose_range curriculum.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # ── Body pose command (6D delta from nominal standing) ───────────────────
    # Ranges start tiny; body_pose_range widens z/roll/pitch once the recovery
    # skills exist (ground_state_mix final at 2500).
    if ENABLE_BODY_CONTROL:
        cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
            resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
            zero_command_prob=BODY_CMD_ZERO_PROB,
            ranges=(
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),  # x (m)
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),  # y (m)
                (-0.005, 0.005),  # z (m)
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # roll
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # pitch
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # yaw
            ),
        )

    # Obs layout parity: [twist(3), head_pose(4), body_pose(6)]. The body slot
    # is zero-padded when body control is off — obs shape is 61D either way.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "head_pose"})
        if ENABLE_BODY_CONTROL:
            cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "body_pose"})
        else:
            cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 6})

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # Robot starts seated — tilt-based fall termination doesn't apply here.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False, params={"sensor_names": ("feet_ground_contact",)})

    # ── Events ────────────────────────────────────────────────────────────────
    # BAM writes per-env dof_frictionloss/dof_damping every step; this no-op
    # event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(func=microduck_mdp.expand_bam_friction_fields, mode="startup")

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Joint/tilt noise on the sitting start: the real hand-off from the sit
    # policy won't reproduce the keyframe exactly, and without noise the policy
    # overfits to the exact canonical SIT pose.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            # Stage 0 of the ground_state_mix curriculum, which ramps these
            # easy→hard. Standing spawns teach HOLDING a stand, not only rising.
            "face_down_prob": 0.20,  # belly to floor (+90° pitch)
            "face_up_prob": 0.00,  # back to floor (-90° pitch) — introduced late
            "sitting_prob": 0.40,  # sit keyframe (deployment hand-off)
            "standing_prob": 0.40,  # already upright at standing height
            # Trunk rests at ~0.044 m face-down (measured); spawn just above the
            # ground instead of the 0.20–0.25 default (a ~15 cm free-fall).
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            # Partial-roll noise on face-up spawns: flat supine→prone has no
            # reward gradient until the roll completes, which made back-recovery
            # seed-lucky. Near-on-side spawns start partway along the roll.
            "face_up_roll_max": math.radians(90),
            "sitting_joint_overrides": SITTING_JOINT_OVERRIDES,
            "sitting_joint_noise_std": 0.12,  # ≈ 7° per joint
            "sitting_tilt_max": math.radians(10),  # ±10° pitch/roll
            # Band around the SIT_Z=0.060 seated equilibrium.
            "sitting_z_min": 0.05,
            "sitting_z_max": 0.09,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=interval, params={"velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}, "asset_cfg": SceneEntityCfg("robot")})

    if ENABLE_COM_RANDOMIZATION:
        # dr.body_ipos with operation="add" re-reads the compile-time default
        # each reset, so it is non-accumulating.
        cfg.events["randomize_com"] = EventTermCfg(func=dr.body_ipos, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "operation": "add", "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE)})

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(func=dr.body_ipos, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES), "operation": "add", "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE)})

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(func=dr.joint_armature, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)), "operation": "scale", "ranges": ARMATURE_RANDOMIZATION_RANGE})

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(func=microduck_mdp.randomize_delayed_actuator_gains, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "operation": "scale", "kp_range": kp_range, "kd_range": kd_range})

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # Physics-consistent mass+inertia (alpha scales both by e^(2α), CoM
        # untouched).
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(func=dr.pseudo_inertia, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0)})

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # Scale BAM's friction budget per-env — dof_frictionloss is zeroed
        # under BAM, so randomizing it directly would be a silent no-op.
        cfg.events["randomize_joint_friction"] = EventTermCfg(func=microduck_mdp.randomize_bam_friction, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE})

    # IMU mounting-misalignment is applied at the OBSERVATION level above: an
    # event writing site_quat is neither per-env nor read by the obs.

    # ── Terrain ───────────────────────────────────────────────────────────────
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

    # ── Curriculum ────────────────────────────────────────────────────────────
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Ramp the spawn mix EASY → HARD. Under a flat split the policy optimized
    # the easy majority (hold-stand + sit-rise) and left the hard poses
    # under-trained — face-up froze into "do nothing". Face-up comes last and
    # is biased heavily at the end so it gets the most practice.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                # step,          standing, sitting, face_down(front), face_up(back)
                {"step": 0, "params": {"standing_prob": 0.40, "sitting_prob": 0.40, "face_down_prob": 0.20, "face_up_prob": 0.00}},
                {"step": 600 * 24, "params": {"standing_prob": 0.25, "sitting_prob": 0.30, "face_down_prob": 0.35, "face_up_prob": 0.10}},
                {"step": 1500 * 24, "params": {"standing_prob": 0.20, "sitting_prob": 0.25, "face_down_prob": 0.30, "face_up_prob": 0.25}},
                {"step": 2500 * 24, "params": {"standing_prob": 0.15, "sitting_prob": 0.20, "face_down_prob": 0.30, "face_up_prob": 0.35}},
            ],
        },
    )

    # 5% → 100% of each joint's reachable delta from HOME over ~2000 iters.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(func=microduck_mdp.pose_command_range_curriculum, params={"command_name": "head_pose", "range_stages": [{"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))}, {"step": 500 * 24, "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))}, {"step": 1000 * 24, "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))}, {"step": 1500 * 24, "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))}, {"step": 2000 * 24, "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))}]})

    # Trunk CoM is capped at ±15 mm: beyond that the randomized CoM can leave
    # the foot support polygon entirely, which trains hyper-reactive correction.
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]})

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(func=microduck_mdp.push_curriculum, params={"event_name": "push_robot", "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}, {"step": 500 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}}, {"step": 1000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}}]})

    # Velocity's exact ramp: the rise is discovered under light smoothing,
    # then damping tightens. A -1.2 end once blocked back-recovery, so -1.0
    # is the ceiling.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.1}, {"step": 500 * 24, "weight": -0.2}, {"step": 750 * 24, "weight": -0.4}, {"step": 1000 * 24, "weight": -0.6}, {"step": 1250 * 24, "weight": -0.8}, {"step": 1500 * 24, "weight": -1.0}]})

    # Anti-violence terms, introduced only AFTER the recovery skills exist
    # (ground_state_mix finishes at 2500). The same weights active from step 0
    # prevent the flips from ever being DISCOVERED. If recovery degrades after
    # 3000, soften the last stage — never move the introduction earlier.
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "arrival_damping", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 3000 * 24, "weight": -0.025}, {"step": 4000 * 24, "weight": -0.05}]})
    # head_pose_bias lands at 1.5 (vs velocity's 3.0) because standup runs
    # head_pose_tracking at 0.75 vs velocity's 2.0. At 1.5 a 15° standing droop
    # costs 0.39/step. If the head is still down, raise the last stage only.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "head_pose_bias", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 3000 * 24, "weight": 0.5}, {"step": 4000 * 24, "weight": 1.5}]})
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "joint_torque_rate_l2", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 3000 * 24, "weight": -1e-3}]})

    # ── Body-control curricula ────────────────────────────────────────────────
    # NOTE the early return: add any unrelated cfg ABOVE this line.
    if not ENABLE_BODY_CONTROL:
        return cfg

    # Ramps in at 2500, when ground_state_mix reaches its hardest mix, so
    # recovery discovery trains without body-command pressure. Final weight
    # 4.0: at full command the fixed-stand terms oppose tracking by ~2/step
    # after the relax stages below, vs a ~0.65/step marginal gain per unit
    # weight. Without those relax stages the opposition is ~4.3/step and even
    # weight 5 loses.
    cfg.curriculum["body_pose_tracking_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "body_pose_tracking", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 2500 * 24, "weight": 1.5}, {"step": 3000 * 24, "weight": 3.0}, {"step": 4000 * 24, "weight": 4.0}]})

    # Only z/roll/pitch widen; x/y/yaw stay at their untracked alive ranges.
    _alive_xy = (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY)
    _alive_ang = (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE)
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                # ranges = (x, y, z, roll, pitch, yaw)
                {"step": 0, "ranges": (_alive_xy, _alive_xy, (-0.005, 0.005), _alive_ang, _alive_ang, _alive_ang)},
                {"step": 2500 * 24, "ranges": (_alive_xy, _alive_xy, (-0.010, 0.005), (-math.radians(8), math.radians(8)), (-math.radians(8), math.radians(8)), _alive_ang)},
                {"step": 3000 * 24, "ranges": (_alive_xy, _alive_xy, (-0.018, 0.008), (-math.radians(12), math.radians(12)), (-math.radians(12), math.radians(12)), _alive_ang)},
                {"step": 4000 * 24, "ranges": (_alive_xy, _alive_xy, (-BODY_CMD_MAX_Z_DOWN, BODY_CMD_MAX_Z_UP), (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE), (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE), _alive_ang)},
            ],
        },
    )

    # Conflict relax: the sharp fixed-stand attractors out-bid commanded
    # deviations (at Δz=−2cm/15° tilt they cost −0.83/−0.79/−1.9 per step).
    # Their bootstrap job is done by 3000, and body_pose_tracking at cmd=0
    # (30% of resamples) takes over the sharp-peak-at-nominal role. The broad
    # bootstrap layers are left untouched — recovery leans on them.
    cfg.curriculum["height_stand_sharp_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "height_stand_sharp", "weight_stages": [{"step": 0, "weight": 1.0}, {"step": 3000 * 24, "weight": 0.5}, {"step": 4000 * 24, "weight": 0.2}]})
    cfg.curriculum["upright_sharp_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "upright_sharp", "weight_stages": [{"step": 0, "weight": 1.5}, {"step": 3000 * 24, "weight": 1.0}, {"step": 4000 * 24, "weight": 0.5}]})
    cfg.curriculum["standing_composite_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "standing_composite", "weight_stages": [{"step": 0, "weight": 3.75}, {"step": 3000 * 24, "weight": 2.5}, {"step": 4000 * 24, "weight": 1.5}]})

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into the ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_stand",
    run_name="microduck_stand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
