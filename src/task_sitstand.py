"""Microduck sitstand task — commanded sit ↔ stand, GENTLY.

One policy, both directions, driven by a posture command:
    cmd (twist slot) = [sit_flag, 0, 0]   sit_flag in {0 = STAND, 1 = SIT}
"Stand" is the all-zero command — the same deployment idle as every other
policy. The command flips mid-episode after a dwell of a few seconds, so each
episode trains descents, seated rest, rises, standing rest, and "hold what
you're already doing" (reset state × command are independent).

Posture-conditioned single-target rewards (posture_*) select the target (SIT
keyframe + SIT_Z vs HOME + STAND_Z) per env from the live command. No
trajectory, no waypoints, no phase timing — the transition path and its length
are the policy's to discover (knee-down first, head assist, etc. are all
allowed: full-collision model, no head-ground penalty, no fall termination).

Keyframes (stability-verified, keep in sync with the standup env):
  SIT   = knee ±1.35, hip_pitch ∓0.4079, ankle/hip_roll 0, trunk z 0.060.
          The old keyframe tipped over — verify TILT in sim before changing it.
  STAND = HOME joints, trunk z 0.115 (measured standing equilibrium).

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
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
# Ramp DELAYED by push_magnitude: pushes mid-descent before the transitions
# have consolidated make the policy unlearn them and converge to "just stand
# doing nothing" (the sit env's lesson).
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Room for 2-3 posture segments, i.e. at least one full sit → rest → rise →
# rest cycle per episode.
EPISODE_LENGTH_S = 12.0
# Dwell before a resample may flip the posture. The lower bound must exceed a
# gentle transition (~1.5 s) plus some rest, so "arrive, then hold still" is
# always trained.
POSTURE_DWELL_S = (3.5, 6.5)
# 0.5 → all four (reset state × command) combinations get equal coverage.
SIT_PROB = 0.5

# ── SIT keyframe (joint_pos index → rad). Single fixed target. ───────────────
# STABILITY-VERIFIED: knee ±1.35, hip_pitch = HOME ∓ 0.05 lean, ankle 0,
# hip_roll 0 settles at 3-5° tilt for 95-100% of noisy resets. The old
# keyframe (knee ±1.0472, hip_pitch HOME) is NOT statically stable — it tips
# to ~88° in 1 s and silently drove the sit env's hop/back-flop/plank exploit
# chain. If the robot or keyframe changes, RE-RUN THE SWEEP and verify TILT,
# not z. Keep in sync with task_standup.SITTING_JOINT_OVERRIDES.
SITTING_TARGET_OVERRIDES = {
    1: 0.0,  # left  hip_roll   (HOME -0.0873)
    2: -0.4079,  # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3: 1.35,  # left  knee       (HOME -0.0049)
    4: 0.0,  # left  ankle      (HOME +0.4530)
    # neck/head intentionally omitted → steered by the head_pose command.
    10: 0.0,  # right hip_roll   (HOME +0.0873)
    11: 0.4079,  # right hip_pitch  (HOME +0.4579)
    12: -1.35,  # right knee       (HOME +0.0049)
    13: 0.0,  # right ankle      (HOME -0.4530)
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk heights (m), both MEASURED in sim — never carried across robot or
# keyframe changes.
STAND_Z = 0.115
SIT_Z = 0.060

# Gating window for upright_while_tall: full incentive above STAND_UPRIGHT_Z,
# fading to 0 at SIT_UPRIGHT_Z (committed to the sit). Blocks the "tip
# backward while still high" descent exploit; the always-on
# upright_linear floor covers the seated regime.
STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z = 0.075

# Target-ramp duration (s): the command term slews an internal STAND↔SIT
# target blend over this time and the posture rewards track the MOVING target.
# THE anti-crash mechanism. With a binary target, arriving early pays the full
# goal jackpot (~7/step) for every step saved while the linear speed caps
# integrate to a bounded cost (~50 total for an instant drop) — crashing won
# ~7×. With the ramp, being AHEAD of the setpoint zeroes the height/composite
# stack for the ramp remainder, so tracking the slow setpoint is the argmax.
POSTURE_RAMP_S = 2.0

# Vertical-speed caps (m/s) — BACKSTOPS for overshoot/bounce around the slewed
# target, not the primary gentleness mechanism. The rise cap is looser (rising
# against gravity needs momentum to get over the heels) and is introduced by
# curriculum only after the rise motion exists.
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED = 0.08

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
from .task_velocity import HEAD_BODY_NAMES, HEAD_POSE_CMD_RESAMPLE_S, MICRODUCK_ROUGH_TERRAINS_CFG


def make_microduck_sitstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck sitstand environment configuration."""

    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)

    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    # No head-ground contact penalty: using the head as a third support point
    # during transitions is explicitly allowed. The plank-as-terminal-rest
    # exploit is anti-selected by posture_composite + posture_stillness
    # instead (both ≈0 at plank tilt/height).

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Standup robot variant: full collision meshes — the body must physically
    # rest on the ground while seated, and knees/head may touch mid-transition.
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

    # ── Rewards: posture-conditioned single-target stack ──────────────────────
    # Every task term reads the commanded posture and selects its target per
    # env. Positive task mass ≈ velocity's, so the shared sim2real
    # regularisers act at the same RELATIVE strength (the standup lesson).

    # Legs only (the head is command-steered). Generous std keeps gradient
    # alive from either end (~1.35 rad knee delta).
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(func=microduck_mdp.posture_pose_match, weight=4.0, params={"command_name": "twist", "std": 0.5, "joint_indices": _LEG_JOINTS, "sit_overrides": SITTING_TARGET_OVERRIDES})

    # Active in BOTH postures. Weight kept light so a transient head-assist
    # during a transition only pays a small tracking cost.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(func=microduck_mdp.head_pose_tracking, weight=0.75, params={"command_name": "head_pose", "std": 0.5})

    # L1 bootstrap — constant gradient toward the commanded pose.
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(func=microduck_mdp.posture_pose_l1, weight=1.0, params={"command_name": "twist", "joint_indices": _LEG_JOINTS, "sit_overrides": SITTING_TARGET_OVERRIDES})

    # Two-layer: wide layer for the bootstrap pull across the 55 mm travel,
    # sharp layer so the final cm has gradient instead of a saturated plateau.
    cfg.rewards["posture_height"] = RewardTermCfg(func=microduck_mdp.posture_height_gaussian, weight=1.0, params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z, "std": 0.04})
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(func=microduck_mdp.posture_height_gaussian, weight=1.0, params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z, "std": 0.015})
    # Resting in the WRONG posture must be clearly net-negative both ways —
    # staying seated under a stand command was standup's stall mode at low L1.
    cfg.rewards["posture_height_l1"] = RewardTermCfg(func=microduck_mdp.posture_height_l1, weight=6.0, params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z})

    # Pays for upward motion itself under a STAND command (zero under SIT):
    # destination-only rewards have zero gradient at zero motion, and without
    # this the standup env parked seated. max_height sits just ABOVE the
    # target so the final cm still pays.
    cfg.rewards["rise_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={
            "command_name": "twist",
            "max_height": 0.125,
            "max_vz": MAX_RISE_SPEED,  # explosive launch can't out-earn a gentle rise
        },
    )

    # ── Gentleness (the point of this env) — three complementary signals ─────
    #  - descent_speed: per-step penalty on downward vz beyond the cap, so a
    #    fast drop pays on every step of the fall and can't be amortised. At
    #    magnitude 5 a crash-sit was still net-positive; 10 from step 0,
    #    tightened to 20 by curriculum.
    #  - rise_speed: the mirror cap, introduced at iter 750 — a motion-tax
    #    active while the skill is being DISCOVERED makes attempts
    #    net-negative and the skill is never found. The sit-keyframe start is
    #    easy (no prone flips), so 750 is late enough.
    #  - gentle_motion: |a_z| shock penalty, both directions, always on.
    #
    # ⚠️ POSITIVE weights: all three functions already return negative values
    # (-clamp(...), -|a_z|). At negative weights the double negative makes
    # them REWARDS for violence and trains a butt-hopping, crash-sitting
    # policy. After any reward change, check Episode_Reward/<penalty> ≤ 0.
    cfg.rewards["descent_speed"] = RewardTermCfg(func=microduck_mdp.trunk_downward_velocity_penalty, weight=10.0, params={"max_down_vel": MAX_DESCENT_SPEED, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["rise_speed"] = RewardTermCfg(func=microduck_mdp.trunk_upward_velocity_penalty, weight=0.0, params={"max_up_vel": MAX_RISE_SPEED, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["gentle_motion"] = RewardTermCfg(func=microduck_mdp.trunk_vertical_accel_penalty, weight=0.05, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Anti-flop calibration. The always-on linear floor holds the trunk
    # vertical at BOTH rests: at 2.5, "lie on your back" trails an upright rest
    # by ~4.5/step. The height-gated booster blocks the "tip backward while
    # tall" descent exploit and doubles as an arrival-uprightness pull.
    cfg.rewards["upright_linear"] = RewardTermCfg(func=microduck_mdp.body_upright_linear, weight=2.5, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["upright_while_tall"] = RewardTermCfg(func=microduck_mdp.upright_while_tall, weight=1.5, params={"height_low": SIT_UPRIGHT_Z, "height_high": STAND_UPRIGHT_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # "Arrive, then rest QUIETLY, UPRIGHT" as an explicit positive peak. The z
    # gate is a band around the commanded height (inactive during transitions);
    # the tilt gate pays nothing for a tilted rest, so back/face/side flops
    # earn zero.
    cfg.rewards["posture_stillness"] = RewardTermCfg(func=microduck_mdp.posture_stillness, weight=2.0, params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z, "band_full": 0.012, "band_zero": 0.03, "vel_std": 0.05, "tilt_full_deg": 25.0, "tilt_zero_deg": 60.0})

    # Multiplicative goal score vs the COMMANDED target — kills partial-sum
    # farming (plank, flop, lean, park-1cm-short). Broad stds keep gradient
    # visible far from the goal. head_std matters: without it the policy
    # rested with the head DANGLING to the floor (everything else on target,
    # and the hanging head adds passive stability). Transient head assist
    # mid-transition stays free (composite ≈0 there anyway).
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name": "twist",
            "sit_overrides": SITTING_TARGET_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "height_std": 0.03,
            "upright_std": 0.40,  # ≈ 23° effective — plank (~70°+) scores ~0
            "pose_std": 0.40,
            "head_std": 0.40,  # head fully dropped (~1.2 rad) → factor ~0.01
        },
    )

    # ── Sim2real regularisers — velocity's exact set and absolute weights ───
    # Plus joint_torque_rate_l2 (anti-jitter), phased in once the transition
    # motions exist. The caps + |a_z| already push toward slow careful motion;
    # these smoothness terms damp jitter WITHOUT blocking a slow big motion, so
    # heavier-than-velocity is defensible — tighten if the real robot shakes.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name})

    # Replaced by the two-layer upright above.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / sit / standup policies) ───
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    # mjlab 1.3.0 base template adds sensor-based foot_height + height_scan obs.
    # Sitstand has no terrain-height sensor (and drops the walking foot rewards),
    # so remove these terms. foot_air_time/foot_contact(_forces) use the
    # feet_ground_contact sensor, which sitstand does define, so they stay.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])

    # IMU obs delay: max_lag 1 — velocity's 2026-07 audit value (real dxl IMU
    # path is fast, ±20 ms envelope).
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

    # Per-env constant rotation of the IMU-derived actor obs; the critic keeps
    # the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel present_velocity is ~1 period old).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Deepcopy per group — they share base-template objects, so the
    # encoder-bias `biased` flag below must not leak into the critic.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Actor sees joint_pos + per-env bias; the critic keeps the true value.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Head pose command ─────────────────────────────────────────────────────
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # Obs layout parity: [twist(3), head_pose(4), body_pose(6)]. body_command
    # stays zero-padded (no body control here).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "head_pose"})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 6})

    # ── Command: sit/stand posture flag in the twist slot ────────────────────
    # The runtime drives this by writing 0/1 into the vx slot of the command
    # buffer. Internally the term slews a target blend over POSTURE_RAMP_S that
    # the posture rewards track; the OBS stays the raw binary flag.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(**{**vars(command), "sit_prob": SIT_PROB, "ramp_s": POSTURE_RAMP_S, "sit_z": SIT_Z, "stand_z": STAND_Z})

    # No fall termination: wobbles/tips during transitions must play out so the
    # policy experiences the impact/upright costs instead of a truncated episode.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    # ── Events ────────────────────────────────────────────────────────────────
    # BAM writes per-env dof_frictionloss/dof_damping every step; this no-op
    # event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(func=microduck_mdp.expand_bam_friction_fields, mode="startup")

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Base reset: standing, just above the measured equilibrium (STAND_Z=0.115).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.11, 0.12)

    # 50/50 standing / already-seated. Combined with the independent 50/50
    # posture command this trains all four cases (sit-from-stand,
    # rise-from-sit, hold-stand, hold-sit) and hands the policy both goal
    # states' values directly.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.5,
            "standing_prob": 0.5,
            "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std": 0.10,  # ≈ 6° per joint
            "sitting_tilt_max": math.radians(8),
            "sitting_z_min": 0.06,  # settles to the 0.060 rest
            "sitting_z_max": 0.075,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    # Contact-solver hardening. The standup XML has full collisions on every
    # body, and the seated pose puts trunk + folded legs + head all in close
    # ground/self contact. The defaults (nconmax=35, iters=10) overflow the
    # solver on sit attempts → NaN → nan_state terminations that punish the
    # descent itself ("learn then unlearn by iter 500").
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

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

    # 5% → 100% of each joint's reachable delta from HOME.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(func=microduck_mdp.pose_command_range_curriculum, params={"command_name": "head_pose", "range_stages": [{"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))}, {"step": 500 * 24, "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))}, {"step": 1000 * 24, "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))}, {"step": 1500 * 24, "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))}, {"step": 2000 * 24, "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))}]})

    # Trunk CoM is capped at ±15 mm: beyond that the randomized CoM can leave
    # the foot support polygon entirely, which trains hyper-reactive correction.
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]})

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    # Delayed: a push mid-transition tips the robot into configurations it
    # can't recover from before the motions have consolidated. Early pushes
    # made the sit policy unlearn sitting and just stand doing nothing.
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(func=microduck_mdp.push_curriculum, params={"event_name": "push_robot", "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}, {"step": 1000 * 24, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}}, {"step": 1500 * 24, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}}, {"step": 2000 * 24, "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}}, {"step": 2500 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}}]})

    # action_rate curriculum — velocity's exact ramp (-0.1 → -1.0 by iter 1500).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.1}, {"step": 500 * 24, "weight": -0.2}, {"step": 750 * 24, "weight": -0.4}, {"step": 1000 * 24, "weight": -0.6}, {"step": 1250 * 24, "weight": -0.8}, {"step": 1500 * 24, "weight": -1.0}]})

    # Discover the sit under magnitude 10 (crash-sit already net-negative),
    # then tighten. POSITIVE weights — the function is self-negating.
    cfg.curriculum["descent_speed_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "descent_speed", "weight_stages": [{"step": 0, "weight": 10.0}, {"step": 500 * 24, "weight": 20.0}]})

    # Introduced only AFTER the rise motion exists: any motion-tax during
    # discovery makes attempts net-negative and the skill is never found. Late
    # (1500/2500) because the rise needs a brief dynamic burst to rock over the
    # heels (vz > 0.08 for a few steps) — an earlier cap stalled the policy in a
    # head-down forward fold. If the rise degrades when this kicks in, soften
    # the final stage, never move it earlier.
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "rise_speed", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 1500 * 24, "weight": 5.0}, {"step": 2500 * 24, "weight": 10.0}]})

    # Torque-rate anti-jitter — phased in once both transition motions exist.
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "joint_torque_rate_l2", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 750 * 24, "weight": -5e-4}, {"step": 1250 * 24, "weight": -1e-3}]})

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckSitStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into the ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_sitstand",
    run_name="microduck_sitstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
