"""Microduck *stand* task (v1.5) — specialized: sitting pose → standing.

Episodic policy that gently rises from the sitting keyframe to the standing
keyframe. Companion to the sit env — together they form a clean sit↔stand
pair, each policy doing one direction.

Reset:  sitting keyframe (trunk z ≈ 0.07, knees/ankles bent, head at HOME).
Target: standing keyframe (trunk z ≈ 0.12, HOME joints).
Reward design (mirror of sit env): a single fixed target is rewarded from
t=0 to end of episode; gentleness is enforced via |a_z| only; smoothness is
enforced by the usual sim2real regularisers. No trajectory waypoints, no
episode-progress gating — the policy is free to discover its own rise path.

Body control (reintroduced 2026-07-29): once standing, the policy tracks a
commanded trunk delta [z, roll, pitch] from the nominal stand (the real
body_pose command in the previously zero-padded 6D obs slot). Kicks in at
iter 2500 via the body-control curricula at the bottom of this file, after
the ground_state_mix recovery curriculum has finished ramping.
"""

import math
from copy import deepcopy

ENABLE_SYMMETRY = False

# Domain randomisation, matched to the velocity env for sim2real parity.
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003           # ramped to 0.015 via com_range curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003           # ramped to 0.01 via head_com_range curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# Final push range; the push curriculum ramps 0 → ±0.08 → this so the sit-rise
# bootstrap isn't shoved around from step 0.
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # deg; real IMU has ~5° systematic pitch error

EPISODE_LENGTH_S = 6.0

# Sitting source pose (servo index → rad): the sit policy's actual end-state (swept
# stable equilibrium). Neck/head omitted → stay at HOME.
SITTING_JOINT_OVERRIDES = {
    1:   0.0,      # left  hip_roll   (HOME -0.0873)
    2:  -0.4079,   # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3:   1.35,     # left  knee       (HOME -0.0049)
    4:   0.0,      # left  ankle      (HOME +0.4530)
    10:  0.0,      # right hip_roll   (HOME +0.0873)
    11:  0.4079,   # right hip_pitch  (HOME +0.4579)
    12: -1.35,     # right knee       (HOME +0.0049)
    13:  0.0,      # right ankle      (HOME -0.4530)
}

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk heights (m), both MEASURED in sim on this model revision. A STAND_Z 5 mm above
# what HOME can reach once forced a permanent back-lean compromise.
SIT_Z = 0.060
STAND_Z = 0.115

# Body pose command. OFF keeps the 61D obs (zero-padded body slot) but drops the command,
# tracking reward and body-control curricula.
ENABLE_BODY_CONTROL = True
# Only z/roll/pitch are tracked; x/y/yaw keep a tiny alive range so their input weights
# don't die. z is asymmetric (little leg extension above HOME). ±20° trained twitchy.
BODY_CMD_MAX_Z_DOWN  = 0.04             # m, crouch below STAND_Z
BODY_CMD_MAX_Z_UP    = 0.030             # m, extend above STAND_Z
BODY_CMD_MAX_ANGLE   = math.radians(15)  # rad, trunk pitch/roll
BODY_CMD_ALIVE_XY    = 0.005             # m, permanent x/y noise range
BODY_CMD_ALIVE_ANGLE = 0.05              # rad, stage-0 / permanent-yaw range
BODY_CMD_ZERO_PROB   = 0.3  # exact-zero command = deployment idle; uniform sampling never hits it

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
    BODY_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_standup_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck stand environment configuration (sit-keyframe start)."""

    site_names = ["left_foot", "right_foot"]

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Task rewards: one fixed target (HOME pose + STAND_Z) from t=0, no waypoints. Total
    # task mass is kept ≈ velocity's (~12) so the shared regularisers act at the same
    # relative strength; at 4× the mass the same weights let jitter go unpriced.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,   # HOME = standing
        },
    )

    # Neck/head are steered by the head_pose command; no other reward touches them.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # Head DC-droop: L1 on a 1 s EMA of head tracking error prices only sustained sag.
    # Upright-gated so the ground/rising phase accumulates nothing (an ungated head
    # penalty once froze the flip), and introduced at iter 3000 by curriculum.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by head_pose_bias_weight curriculum
        params={
            "command_name":       "head_pose",
            "tau_s":              1.0,
            "gate_height_low":    0.09,
            "gate_height_high":   0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": 45.0,
        },
    )

    # L1 bootstrap: constant gradient far from HOME; weight chosen so parking ~0.18 rad
    # off-HOME (bent knees) is not cheap enough to ignore.
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=1.25,
        params={
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Two-layer height Gaussian: wide std for the bootstrap pull from sit, narrow std for
    # the last cm (the wide layer alone saturated at z ≈ 0.109 with no gradient left).
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std":           0.04,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std":           0.015,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # Weight sized so "sit still" is net negative (it was a positive basin at 1/3 this).
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=7.5,
        params={
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Pays for the MOTION of rising (destination-only rewards made "sit upright" the
    # optimum). max_height just above STAND_Z so it stays active through the last cm
    # (a gate at 0.11 parked the policy at 0.108). No max_vz cap: capping rise speed
    # taxed noisy recovery attempts during discovery and prone recovery never trained.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.75,
        params={
            "asset_cfg":  SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": 0.125,
        },
    )

    # |a_z| penalty. POSITIVE weight: trunk_vertical_accel_penalty already returns -|a_z|
    # (a negative weight rewarded shocks). Ungated, so prone flips pay it in full; 0.005
    # is the ceiling — doubling it froze face-up recovery.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Arrival damper: trunk ω_xy² gated on height and tilt, against the real-robot
    # rise → overshoot → tip → retry loop. Weight 0 until iter 3000: any attempt-tax
    # active while prone recovery is being discovered (ground_state_mix ramps until 2500)
    # makes "do nothing" win — timing, not magnitude, decides this.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    0.09,
            "height_high":   0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Two-layer upright: cos(tilt) pulls from any orientation, the height-gated Gaussian
    # has gradient near vertical where cos runs out (a run converged at 37° back-lean).
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Height gate blocks the "crouch low and vertical" exploit; std 0.3 (≈17°) so the
    # lean basin still scores visibly (0.1 was flat there).
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=1.5,
        params={
            "std":         0.3,
            "height_low":  SIT_Z,
            "height_high": STAND_Z,
            "asset_cfg":   SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Multiplicative goal score with broad stds: tight stds scored ~5e-5 at the lean basin
    # (no gradient); these score ~0.2 there and ~1.0 at the goal.
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=3.75,
        params={
            "target_height":    STAND_Z,
            "height_std":       0.04,
            "upright_std":      0.40,
            "pose_std":         0.40,
            "joint_indices":    _LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg":        SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Locomotion variant (not _6d): x/y would reference the spawn origin the robot leaves
    # during flips. Weight ramps in from iter 2500 (after ground_state_mix). Tight stds:
    # at 1 cm z error, z_std 0.01 scores 0.37 (gradient), 0.02 scores 0.78 (none).
    if ENABLE_BODY_CONTROL:
        cfg.rewards["body_pose_tracking"] = RewardTermCfg(
            func=microduck_mdp.body_pose_tracking_locomotion,
            weight=0.0,
            params={
                "command_name": "body_pose",
                "nominal_height": STAND_Z,
                "z_std": 0.01,
                "angle_std": math.radians(5),
                "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
                "vel_gate_command_name": None,
            },
        )

    # Sim2real regularisers: velocity's set and weights. body_ang_vel / action_rate are
    # motion-blockers for the flip (-0.15 / -1.2 once killed back-recovery; if recovery
    # freezes, halve body_ang_vel first). joint_torque_rate_l2 starts at 0 and is
    # introduced at iter 3000 with arrival_damping (see there).
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Drop only the base "upright" Gaussian — standup uses its own
    # upright_linear/upright_sharp instead. (angular_momentum kept above to match
    # velocity; soft_landing/hip_yaw_roll_deviation dropped to match velocity.)
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # mjlab 1.3.0 base template adds sensor-based foot_height + height_scan obs.
    # Standup has no terrain-height sensor (and drops the walking foot rewards),
    # so remove these terms. foot_air_time/foot_contact(_forces) use the
    # feet_ground_contact sensor, which standup does define, so they stay.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # Sensor-derived critic obs get NaN-safe wrappers: a non-finite contact force slips past
    # robot_state_is_nan and kills the run via check_nan. Standup flips constantly.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # IMU obs delay ≤ 1 step: the real dxl IMU path is within ±20 ms.
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    # Per-env constant IMU misalignment on the actor obs only; critic sees truth.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Actor/critic terms share base-template objects; deepcopy so `biased` hits actor only.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Head pose command: deltas from HOME, ranges widened by the head_pose_range curriculum.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # Body pose command [x, y, z, roll, pitch, yaw]; z/roll/pitch widened by curriculum
    # once recovery skills exist.
    if ENABLE_BODY_CONTROL:
        cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
            resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
            zero_command_prob=BODY_CMD_ZERO_PROB,
            ranges=(
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),        # x (m)
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),        # y (m)
                (-0.005, 0.005),                                # z (m)
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # roll
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # pitch
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # yaw
            ),
        )

    # Command obs slots [twist(3), head_pose(4), body_pose(6)]; body slot zero-padded when
    # body control is off so the obs shape is identical.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        if ENABLE_BODY_CONTROL:
            cfg.observations[group].terms["body_command"] = ObservationTermCfg(
                func=mdp.generated_commands, params={"command_name": "body_pose"},
            )
        else:
            cfg.observations[group].terms["body_command"] = ObservationTermCfg(
                func=microduck_mdp.zero_command_padding, params={"dim": 6},
            )

    # Twist command: tiny non-zero range keeps the slot's input weights alive.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # Robot starts seated/prone — tilt-based fall termination doesn't apply.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )

    # BAM writes per-env dof_frictionloss/dof_damping; register the fields for expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Spawn mix (stage 0 of the ground_state_mix curriculum; face-up is introduced late).
    # Sit spawns carry joint/tilt noise because the real sit→stand hand-off never matches
    # the keyframe. Prone z is just above the measured 0.044 m rest height.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":            0.20,
            "face_up_prob":              0.00,
            "sitting_prob":              0.40,
            "standing_prob":             0.40,
            "prone_z_min":               0.05,
            "prone_z_max":               0.09,
            "face_up_roll_max":          math.radians(90),  # reverse curriculum, see mdp
            "sitting_joint_overrides":   SITTING_JOINT_OVERRIDES,
            "sitting_joint_noise_std":   0.12,
            "sitting_tilt_max":          math.radians(10),
            "sitting_z_min":             0.05,
            "sitting_z_max":             0.09,
            "standing_z_min":            0.11,
            "standing_z_max":            0.12,
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        # dr.body_ipos with operation="add" re-reads compile-time defaults: non-accumulating.
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # pseudo_inertia scales mass and inertia together by e^(2α), CoM untouched.
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # dof_frictionloss is zeroed under BAM; scale the actuator's friction budget instead.
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

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

    # Spawn-mix curriculum easy → hard: a flat 25/25/25/25 from step 0 let the policy
    # optimise the easy majority and freeze face-up into "do nothing".
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                # step,          standing, sitting, face_down(front), face_up(back)
                {"step": 0,          "params": {"standing_prob": 0.40, "sitting_prob": 0.40, "face_down_prob": 0.20, "face_up_prob": 0.00}},
                {"step": 600 * 24,   "params": {"standing_prob": 0.25, "sitting_prob": 0.30, "face_down_prob": 0.35, "face_up_prob": 0.10}},
                {"step": 1500 * 24,  "params": {"standing_prob": 0.20, "sitting_prob": 0.25, "face_down_prob": 0.30, "face_up_prob": 0.25}},
                {"step": 2500 * 24,  "params": {"standing_prob": 0.15, "sitting_prob": 0.20, "face_down_prob": 0.30, "face_up_prob": 0.35}},
            ],
        },
    )

    # Head pose range: 5% → 100% of each joint's reachable delta over ~2000 iters.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # Trunk CoM DR capped at ±15 mm: beyond that the CoM can leave the foot support
    # polygon and trains hyper-reactive correction.
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 500 * 24,  "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 1000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    # Rise is discovered under light smoothing, then damping tightens. -1.0 is the
    # ceiling: -1.2 once blocked back-recovery.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 500 * 24,   "weight": -0.2},
                {"step": 750 * 24,   "weight": -0.4},
                {"step": 1000 * 24,  "weight": -0.6},
                {"step": 1250 * 24,  "weight": -0.8},
                {"step": 1500 * 24,  "weight": -1.0},
            ],
        },
    )

    # Smoothness polish introduced at iter 3000, after ground_state_mix finishes (2500).
    # The same weights from step 0 stopped the flips being discovered at all. If recovery
    # degrades after 3000, soften the last stage; never move the introduction earlier.
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": -0.025},
                {"step": 4000 * 24,  "weight": -0.05},
            ],
        },
    )
    # Final 1.5 = velocity's 3.0 scaled to this task mass; 15° droop costs 0.39/step.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "head_pose_bias",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": 0.5},
                {"step": 4000 * 24,  "weight": 1.5},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": -1e-3},
            ],
        },
    )

    # Body-control curricula only below this early return.
    if not ENABLE_BODY_CONTROL:
        return cfg

    # Ramps in at 2500 when ground_state_mix reaches its final mix. Final 4.0 out-bids the
    # remaining fixed-stand opposition (~2/step after the relax stages below).
    cfg.curriculum["body_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "body_pose_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": 1.5},
                {"step": 3000 * 24,  "weight": 3.0},
                {"step": 4000 * 24,  "weight": 4.0},
            ],
        },
    )

    # Range widening synced to the weight ramp; only z/roll/pitch widen.
    _alive_xy  = (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY)
    _alive_ang = (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE)
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                # ranges = (x, y, z, roll, pitch, yaw)
                {"step": 0, "ranges": (
                    _alive_xy, _alive_xy, (-0.005, 0.005),
                    _alive_ang, _alive_ang, _alive_ang,
                )},
                {"step": 2500 * 24, "ranges": (
                    _alive_xy, _alive_xy, (-0.010, 0.005),
                    (-math.radians(8), math.radians(8)),
                    (-math.radians(8), math.radians(8)),
                    _alive_ang,
                )},
                {"step": 3000 * 24, "ranges": (
                    _alive_xy, _alive_xy, (-0.018, 0.008),
                    (-math.radians(12), math.radians(12)),
                    (-math.radians(12), math.radians(12)),
                    _alive_ang,
                )},
                {"step": 4000 * 24, "ranges": (
                    _alive_xy, _alive_xy,
                    (-BODY_CMD_MAX_Z_DOWN, BODY_CMD_MAX_Z_UP),
                    (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                    (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                    _alive_ang,
                )},
            ],
        },
    )

    # Relax the sharp fixed-stand attractors once their bootstrap job is done: at full
    # command they out-bid tracking by ~3.5/step. body_pose_tracking at cmd=0 takes over
    # the sharp-peak role; the broad bootstrap layers recovery leans on stay untouched.
    cfg.curriculum["height_stand_sharp_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "height_stand_sharp",
            "weight_stages": [
                {"step": 0,          "weight": 1.0},
                {"step": 3000 * 24,  "weight": 0.5},
                {"step": 4000 * 24,  "weight": 0.2},
            ],
        },
    )
    cfg.curriculum["upright_sharp_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "upright_sharp",
            "weight_stages": [
                {"step": 0,          "weight": 1.5},
                {"step": 3000 * 24,  "weight": 1.0},
                {"step": 4000 * 24,  "weight": 0.5},
            ],
        },
    )
    cfg.curriculum["standing_composite_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "standing_composite",
            "weight_stages": [
                {"step": 0,          "weight": 3.75},
                {"step": 3000 * 24,  "weight": 2.5},
                {"step": 4000 * 24,  "weight": 1.5},
            ],
        },
    )

    return cfg


MicroduckStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer must be baked into the ONNX (export.py)
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_stand",
    run_name="microduck_stand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
