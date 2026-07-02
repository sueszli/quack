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
"""

import math
from copy import deepcopy

# Symmetry
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to the velocity env for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True   # match velocity: randomize head-assembly CoM
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_ARMATURE_RANDOMIZATION        = True   # match velocity: reflected rotor inertia
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True

# ── Ranges (matched to the velocity env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003           # ramped to 0.02 via com_range curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003           # ramped to 0.01 via head_com_range curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)    # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)      # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# Softer than velocity's ±0.5: a ±0.5 m/s instant velocity shove is violent for a
# STATIC stander (velocity walks with momentum + a wider dynamic base), which drove
# frantic tiny-step recovery. ±0.25 gives more deliberate recovery. (push curriculum
# below ramps up to this final value.)
VELOCITY_PUSH_RANGE                 = (-0.25, 0.25)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 2.0

# Episode length: long enough for a gentle rise + brief stabilisation.
EPISODE_LENGTH_S = 6.0

# ── Sitting source pose (asset.data.joint_pos index → angle in rad) ───────────
# Must match the *actual end-state* of the sit policy (head at HOME, knees bent
# ~60°, ankles 0). Neck/head intentionally omitted → reset stays at HOME so the
# standup policy starts from exactly where the sit policy converges.
# Articulation joint indices under mjlab 1.3.0 + canonical BAM. The passive jaw
# joints are NO LONGER part of the articulation (excluded from qpos), so the
# layout is the clean 14-joint order: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
# (Previously passive_1/passive_2 sat at 9,10 and shifted the right leg to 11-15.)
SITTING_JOINT_OVERRIDES = {
    1:   0.0,      # left  hip_roll
    3:   1.0472,   # left  knee
    4:   0.0,      # left  ankle
    10:  0.0,      # right hip_roll
    12: -1.0472,   # right knee
    13:  0.0,      # right ankle
}

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk height targets (m).
SIT_Z = 0.07
# STAND_Z = empirically-measured trunk z at the natural standing equilibrium
# (HOME joint pose, vertical trunk). Previously was 0.120 — 5 mm above
# what's mechanically reachable at HOME — which forced the policy into a
# back-lean compromise to satisfy the impossible height target. Measured
# via the velocity policy holding the robot still at zero command: 115 mm.
STAND_Z = 0.115

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

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
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

    # ── Rewards: minimum-viable set for an organic standup policy ────────────
    # Single fixed target (STAND = HOME pose + STAND_Z), active from t=0. No
    # trajectory, no waypoints, no episode-progress gating. The policy is free
    # to discover any rise path that satisfies:
    #   (1) end-state matches the HOME pose + STAND_Z
    #   (2) rise is gentle (low |a_z| throughout)
    #   (3) trunk stays upright throughout (failure mode: tip backward while
    #       extending legs; no "low z is safe" regime as in sit)
    #   (4) joint/action motion stays smooth (sim2real regularisers)

    # Pose target — legs+hips+knees+ankles. target_overrides=None → HOME.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,   # HOME = standing
        },
    )

    # Head pose tracking (commandable head control, like the velocity env).
    # Replaces the old pose_stand_neck reward (which pinned the neck/head to HOME)
    # — the neck/head are now steered by the head_pose command instead. Removed
    # from pose_stand_l1 / standing_composite below for the same reason, so no
    # reward fights head_pose_tracking's gradient.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=3.0,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # L1 bootstrap — constant gradient even when far from HOME.
    # Bumped 2 → 5: at convergence the policy parks ~0.18 rad off-HOME (mostly
    # bent knees) costing only -0.35/step at weight 2 — cheap enough to ignore.
    # At weight 5 that error costs -0.9/step, forcing the policy to actually
    # close the gap on the remaining joints.
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=5.0,
        params={
            # Legs only — neck/head are steered by head_pose_tracking.
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Trunk height target — two-layer Gaussian to get both bootstrap reach
    # AND a sharp peak at STAND_Z.
    #  - ``height_stand``: wide std (0.04), for the bootstrap pull from sit.
    #  - ``height_stand_sharp``: narrow std (0.015), creates a strong gradient
    #    in the final cm. Earlier runs converged at z ≈ 0.109 because the
    #    wide-std Gaussian was already saturated (0.93/1.0) — no gradient to
    #    pull the last cm. The sharp layer adds 0.36→1.0 reward jump in that
    #    same range, ~3× the marginal pull.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std":           0.04,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std":           0.015,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # L1 bumped 10 → 30: previous run plateaued sitting still because the
    # static-sit basin (-0.5 reward from L1 + everything else positive) was
    # net positive. At weight 30, sitting still costs -1.5/step — net cost
    # of "stay sitting" forces exploration.
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=30.0,
        params={
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Reward upward CoM velocity below STAND_Z — pays for the *motion* of
    # rising, not just for the destination. Critical bootstrap: with only
    # destination rewards, "stay sitting upright collecting most-of-pose +
    # upright" was the dominant local optimum. Rewarding vz > 0 directly
    # makes any rise attempt immediately positive. Gates off above
    # max_height so the policy can't farm it by bobbing.
    # max_height set just above STAND_Z (0.12 → 0.125) so the reward stays
    # active through the final cm of rise. Earlier 0.11 caused the policy to
    # park at ~0.108 (gate-off altitude) and never finish the climb.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=3.0,
        params={
            "asset_cfg":  SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": 0.125,
        },
    )

    # Gentle rise — penalty on |a_z|. Compatible with com_upward_velocity:
    # constant positive vz collects upward-velocity reward AND has a_z = 0,
    # so the two pressures together select for smooth constant-velocity rise.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Upright — two-layer like the height reward.
    #  - ``upright_linear``: cos(tilt). Strong gradient at high tilt (e.g.,
    #    while inverted at the start of a recovery), weak near vertical.
    #    Provides bootstrap pull from any orientation.
    #  - ``upright_sharp``: exp(-tilt²/std²) with std ≈ 6°. Gradient is
    #    STRONGEST in the near-vertical regime where the linear version
    #    runs out of steam. Previous run converged at ~37° back-lean because
    #    the linear pull at small tilt becomes weak; this term punishes that
    #    exact regime.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Sharp Gaussian upright, gated by trunk z. Pays only when the robot is
    # actually at the standing height — prevents the "crouch low and vertical"
    # exploit. Broadened std 0.1 → 0.3 (≈17°): too sharp before, scored
    # near-zero at the lean basin (no gradient). With 0.3, the lean basin
    # at z=0.111 (smoothstep ~0.91) and tilt 37° (gaussian ~0.11) scores
    # ~0.1 = visible gradient that pulls toward vertical.
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=6.0,
        params={
            "std":         0.3,
            "height_low":  SIT_Z,
            "height_high": STAND_Z,
            "asset_cfg":   SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Smooth multiplicative goal-state score (broad stds).
    # The previous tight stds (height=0.015, upright=0.15, pose=0.20) had
    # the composite at ~5e-5 at the lean basin — invisible to the policy,
    # zero gradient. Broadening so the lean basin scores ~0.2 (visible
    # gradient) while the goal still scores ~1.0 (clear attractor).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height":    STAND_Z,
            "height_std":       0.04,    # 4cm — broad, covers the climb
            "upright_std":      0.40,    # ≈ 23° — lean basin scores ~0.3
            "pose_std":         0.40,    # joint-RMS, broad enough for partial pose
            "joint_indices":    _LEG_JOINTS,   # neck/head steered by head_pose_tracking
            "target_overrides": None,
            "asset_cfg":        SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # ── Sim2real regularisers (smoothness) ────────────────────────────────────
    # action_rate base weight + curriculum matched to the velocity env (see below)
    # Base weight matches velocity (-0.6); the action_rate_weight curriculum
    # below ramps it -0.4 → -0.8 → -1.0 (same stages as velocity).
    cfg.rewards["action_rate_l2"]        = RewardTermCfg(func=mdp.action_rate_l2,                 weight=-0.6)
    cfg.rewards["joint_torque_rate_l2"]  = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=-5e-4)
    cfg.rewards["joint_torques_l2"]      = RewardTermCfg(func=microduck_mdp.joint_torques_l2,     weight=-5e-3)

    # ── Stability ─────────────────────────────────────────────────────────────
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    # Bumped -0.3 → -0.6: at convergence the policy was shaking visibly
    # while holding the standing pose (body_ang_vel reward ≈ -0.6 means
    # |ω|² ≈ 1 → ω ≈ 1 rad/s constant wobble). Heavier damping makes
    # "settle and stop moving" cheaper than continuous oscillation.
    cfg.rewards["body_ang_vel"].weight = -0.6

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Hip yaw/roll drift penalty — keeps a narrow base while pushing up. Without
    # it the policy can splay legs sideways while extending knees, which gives
    # a low-cost "push trunk up via leg spread" exploit instead of a clean rise.
    cfg.rewards["hip_yaw_roll_deviation"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*"),
            ),
        },
    )

    # Drop velocity-env terms that are either subsumed (upright Gaussian,
    # angular_momentum, soft_landing) or irrelevant here.
    for name in ("upright", "angular_momentum", "soft_landing"):
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Observations (identical layout to walking / sit policies) ─────────────
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

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Obs noise matched to the velocity env.
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = passive_excluded
    cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)

    # ── Head pose command (commandable head control, like the velocity env) ───
    # 4D deltas-from-HOME on neck/head joints: [neck_pitch, head_pitch, head_yaw,
    # head_roll]. Tracked by head_pose_tracking below; ranges widened by the
    # head_pose_range curriculum. Same per-joint caps as the velocity env.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # Command obs slots. head_command is now the real head_pose command (was
    # zero-padding); body_command stays zero-padded (body control not used here).
    # Layout parity with velocity/velstand: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
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

    # ── Terminations ──────────────────────────────────────────────────────────
    # Robot starts seated — tilt-based fall termination doesn't apply here.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # match velocity

    # Start in the sitting keyframe with noise on joints + trunk tilt. Real
    # deployment hand-off from the sit policy won't reproduce the SIT
    # keyframe exactly — the standup policy must be robust to a band of
    # plausible "sit-ish" starts. Without noise the policy was overfitting
    # to the exact canonical SIT pose.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            # Initialize from any pose, 25% each: front (face-down), back
            # (face-up), sitting keyframe, and already-standing (so the policy
            # also learns to *hold* a stand, not only to rise).
            # Initial mix = curriculum stage 0 (easy); the ground_state_mix
            # curriculum ramps these easy→hard over training. Face-up (back) starts
            # at 0 and is introduced late (hardest recovery).
            "face_down_prob":            0.20,  # belly to floor (+90° pitch)
            "face_up_prob":              0.00,  # back to floor (-90° pitch) — introduced late
            "sitting_prob":              0.40,  # sit keyframe (deployment hand-off)
            "standing_prob":             0.40,  # already upright at standing height
            # Prone reset height: trunk rests at ~0.044 m face-down (measured), so
            # spawn just above the ground rather than the 0.20–0.25 default (which
            # would free-fall ~15 cm before landing).
            "prone_z_min":               0.05,
            "prone_z_max":               0.09,
            "sitting_joint_overrides":   SITTING_JOINT_OVERRIDES,
            "sitting_joint_noise_std":   0.12,           # ≈ 7° per joint
            "sitting_tilt_max":          math.radians(10),  # ±10° pitch/roll
            "sitting_z_min":             0.06,           # widen z range too
            "sitting_z_max":             0.10,
            # Standing init: trunk just above the measured equilibrium (STAND_Z=0.115).
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
        # mjlab 1.3.0: stock dr.body_ipos (operation="add") reads the compile-time
        # default each reset → non-accumulating natively.
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
        # Match velocity: randomize the CoM of the head-assembly bodies.
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
        # Match velocity: reflected rotor inertia (non-accumulating, affects BAM).
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
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=microduck_mdp.randomize_mass_and_inertia,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "scale_range": MASS_INERTIA_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_imu_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_imu_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE,
            },
        )

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

    # Init-pose curriculum: ramp the set_ground_state mix from EASY → HARD instead
    # of a flat 25/25/25/25 from step 0. With the flat split the policy optimized
    # the easy majority (hold-stand + sit-rise) and left the hard poses under-
    # trained — front only partially rose and face-up (back) froze into "do
    # nothing". This introduces standing/sitting first, then face-down, then
    # face-up last, and biases toward the hard poses late so they get the most
    # practice. (event_param_curriculum shallow-merges these keys into the live
    # set_ground_state event; the z-ranges / joint overrides are left untouched.)
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

    # Head pose command range curriculum — same per-joint widening as the velocity
    # env (5% → 100% of each joint's reachable delta from HOME over ~2000 iters).
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

    # NOTE: the earlier head_pose_std / head_pose_weight curricula (band-aids for
    # the head-droop) were removed — the droop was a backward-CoM balance crutch,
    # fixed at the source by the STAND2 forward-shifted standing pose. head_pose
    # tracking stays at its baseline (weight 3.0, std 0.5) + head_pose_range.

    # CoM-randomization range curricula — match velocity (ramp 0.003 → 0.02 trunk,
    # 0.003 → 0.01 head over the first ~2000 / ~1000 iters).
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
                    {"step": 2000 * 24, "range": 0.02},
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

    # action_rate curriculum — same stages as the velocity env: ramp the
    # action_rate_l2 penalty -0.4 → -0.8 → -1.0 over the first 500 iters.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,         "weight": -0.4},
                {"step": 250 * 24,  "weight": -0.8},
                {"step": 500 * 24,  "weight": -1.0},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
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
