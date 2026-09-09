"""Microduck roller standup — get up on rollers, then HOLD the stance.

Episodic policy: the robot starts face down, face up, or already standing.
Port of the walking-duck `standup` recipe to the rollers model, derived from
`make_microduck_velocity_rollers_env_cfg` so the robot, sensors, DR and the 61D
observation are inherited as-is (runtime-interchangeable).

Two structural differences from `standup`:
  - the passive wheels INTERLEAVE in the joint order → remapped indices
    (_LEG_JOINTS below), locked by tests/test_roller_standup_cfg.py;
  - no head_pose command: the head/body slots stay zero-padded (roller family
    convention) and the head is held straight by neck_joint_pos_l2, by NAME.

The new piece is the INVERTED rolling-friction curriculum (wheels braked →
free): the wheels roll, so there is no grip to push off. Bootstrap with nearly
locked wheels, then ramp toward the true value. If `standing_composite`
collapses at a stage, the "grippy feet" gesture does not transfer and a skater
technique will have to be guided (knee support, one skate at a time).

Deployed in `--standing` opposite the roller policy in `--walking`, switched on
the magnitude of the velocity command; the twist slot is left at zero there.
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from . import task_mdp as microduck_mdp
from .task_symmetry import PpoWithSymmetryCfg
from .task_velocity_rollers import make_microduck_velocity_rollers_env_cfg

# ── Trunk heights (m) ─────────────────────────────────────────────────────────
# Measured by exact kinematics on scene_rollers.xml (minimum mesh vertex of the
# colliding geoms, STAND pose, trunk brought down to contact): standing 0.1407,
# resting face down 0.0752, face up 0.0475. The model WITHOUT wheels gives
# 0.1172 kinematically vs the 0.115 standup measured under load → ~2 mm of sag,
# applied here too.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S = 6.0  # rise + stabilize
NUM_STEPS_PER_ENV = 24

# ── Play override: force the proportion of FACE-UP starts ─────────────────────
# At play the env is rebuilt from scratch: common_step_counter restarts at 0, so
# ground_state_mix applies its stage 0 where face_up_prob = 0 — we would NEVER
# see a face-up start, which is exactly the case worth inspecting by eye.
#   STANDUP_PLAY_FACE_UP=1.0  -> 100 % face-up starts
#   STANDUP_PLAY_FACE_UP=0.4  -> the mix of the last curriculum stage
#   undefined / "none" / "random" -> default behavior (stage 0)
# Only has an effect with play=True.
PLAY_FACE_UP = None
# Face-down:standing ratio of the LAST curriculum stage. The remainder
# (1 - face_up) is split in this ratio, so 0.4 reproduces the end-of-training mix.
_PLAY_FACE_DOWN_SHARE = 2.0 / 3.0


def _resolve_play_face_up():
    """Proportion of face-up starts at play: STANDUP_PLAY_FACE_UP env var, otherwise the constant."""
    raw = os.environ.get("STANDUP_PLAY_FACE_UP")
    if raw is None:
        return PLAY_FACE_UP
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_standup] STANDUP_PLAY_FACE_UP='{raw}' invalid -> default {PLAY_FACE_UP}")
        return PLAY_FACE_UP


# ── Joint indices — the passive wheels are INTERLEAVED ────────────────────────
# Actual order of the rollers model (18 joints after the free-joint), verified in
# MuJoCo via get_walk_rollers_spec().compile():
#   0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
#   5-6   passive_LF_wheel, passive_LR_wheel
#   7-10  neck_pitch, head_pitch, head_yaw, head_roll
#   11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
#   16-17 passive_RF_wheel, passive_RR_wheel
# standup's [0-4, 9-13] / [5-8] are the indices of the model WITHOUT wheels and
# do NOT hold here. Locked by tests/test_roller_standup_cfg.py.
#
# Only _LEG_JOINTS is consumed (by the pose rewards). _NECK_JOINTS and
# _WHEEL_JOINTS document the layout and back the index test: the neck resolves
# by NAME (neck_joint_pos_l2) and the wheels by the ^passive_.* regex.
_LEG_JOINTS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS = [7, 8, 9, 10]
_WHEEL_JOINTS = [5, 6, 16, 17]

# Skating rewards of the roller env, meaningless on the ground. feet_flat: the
# blades are NOT flat during the rise, so it would fight the gesture.
# hip_roll_neutral: getting up requires spreading the legs. pose /
# com_height_target / upright: replaced by the standup targets below.
_SKATING_REWARDS = ("wheel_speed", "braking", "skating_air_time", "glide", "single_support", "gait_symmetry", "forward_lean", "heading_hold", "feet_flat", "hip_roll_neutral", "pose", "com_height_target", "upright")


def make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """ "Getting up on rollers" env: start on the ground, target = standing on wheels."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Skating rewards removed ───────────────────────────────────────────────
    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # ── Command: twist slot neutralized (≈ 0) ────────────────────────────────
    # The roller env installs a RelativeHeadingVelocityCommandCfg (cmd[2] = heading
    # error computed internally). Here nothing is steered: we go back to the
    # neutralized command-only, like standup. The head_pose (4) and
    # body_pose (6) slots stay zero-padded → 61D obs parity preserved.
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

    # ── Numerical robustness (same choice as roller_slope) ───────────────────
    # A rare contact (~1/25M steps) makes the free-joint diverge to NaN: we
    # sanitize the obs (→ 0) so as not to kill the training, the offending env resets
    # at the next step.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # ── Standup rewards — transplanted from standup, remapped ─────────────────
    # The weights come from the iterations documented in task_standup.py: only
    # touch them with a reason. Only the joint indices and the two heights
    # change here. NB: a FRESH SceneEntityCfg per term — mjlab resolves and
    # mutates them in place, and a shared object gives stale indices.

    # LEGS only: the neck and head are held by the inherited neck_joint_pos_l2,
    # which resolves by NAME.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(func=microduck_mdp.pose_target_match, weight=8.0, params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None})
    # L1 bootstrap: constant gradient even far from HOME (the Gaussian saturates).
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(func=microduck_mdp.pose_l1_penalty, weight=5.0, params={"joint_indices": _LEG_JOINTS, "target_overrides": None})

    # Three layers: a wide Gaussian pulls from the ground, a narrow one forces
    # the last cm where the wide one is saturated, and a strong L1 makes
    # "staying on the ground" net NEGATIVE — without it the policy settles for
    # the lazy optimum of lying motionless.
    cfg.rewards["height_stand"] = RewardTermCfg(func=microduck_mdp.height_target_gaussian, weight=4.0, params={"std": 0.04, "target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(func=microduck_mdp.height_target_gaussian, weight=4.0, params={"std": 0.015, "target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["height_stand_l1"] = RewardTermCfg(func=microduck_mdp.height_l1_penalty, weight=30.0, params={"target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Pays for the rising MOTION, not just the destination: without it, "stay
    # seated collecting the partial pose" dominates. The cutoff is 10 mm ABOVE
    # the target, or the policy parks at the cutoff altitude and never finishes.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(func=microduck_mdp.com_upward_velocity, weight=3.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "max_height": ROLLER_STAND_Z + 0.010})
    # Penalizes |a_z|. Pairs with com_upward_velocity: a constant vertical
    # velocity collects that reward AND has a_z = 0, so together they select a
    # smooth rise at constant velocity.
    #
    # ⚠️ POSITIVE WEIGHT, not a typo. trunk_vertical_accel_penalty already
    # returns -|a_z| (like height_l1_penalty and pose_l1_penalty, used here at
    # +30 and +5). The -0.02 inherited from standup was a double negative that
    # REWARDED vertical acceleration — measured at Episode_Reward/gentle_rise =
    # +0.0118, the only penalty term logged positive. That is the cause of the
    # "very violent" behaviour, and it explains the failed damping attempts
    # documented in standup: they were fighting a term pushing the other way.
    #
    # The magnitude stays DELIBERATELY small: |a_z| is necessarily high during a
    # roll-over from the back, so a large weight here is a motion blocker. The
    # actual damping is carried by joint_torque_rate_l2, which penalizes torque
    # VARIATION rather than the motion itself.
    cfg.rewards["gentle_rise"] = RewardTermCfg(func=microduck_mdp.trunk_vertical_accel_penalty, weight=+0.02, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Two layers: cos(tilt) has a strong gradient when lying down but runs out
    # of steam near vertical, so the tight height-gated Gaussian takes over and
    # kills the backward lean (standup's failure mode: tipping backwards while
    # extending the legs).
    cfg.rewards["upright_linear"] = RewardTermCfg(func=microduck_mdp.body_upright_linear, weight=6.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["upright_sharp"] = RewardTermCfg(func=microduck_mdp.upright_gaussian_at_height, weight=6.0, params={"std": 0.3, "height_low": ROLLER_PRONE_Z, "height_high": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # MULTIPLICATIVE height × verticality × pose: being good on 2 of 3 pays
    # nothing, which breaks the "leaning at the right height" compromises that
    # additive rewards let through. Stds deliberately WIDE to stay visible
    # during the rise — tight stds scored ~5e-5, i.e. zero gradient.
    cfg.rewards["standing_composite"] = RewardTermCfg(func=microduck_mdp.standing_composite_score, weight=15.0, params={"target_height": ROLLER_STAND_Z, "height_std": 0.04, "upright_std": 0.40, "pose_std": 0.40, "joint_indices": _LEG_JOINTS, "target_overrides": None, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Anti-jitter: penalizes torque VARIATION, not its amplitude nor the trunk
    # rotation, so it damps trembling without blocking the roll-over. standup
    # identified it as the only damper that does not kill the rise from the
    # back, so it is THE safe lever to raise.
    #
    # Calibration: |Δτ|² is ~0.1 at convergence, so the contribution is ≈ 0.1 ×
    # |weight|. standup's inherited -2e-3 contributed -0.0002/step against ~+41.6
    # of task reward — nothing at all. At -2.0 it measured -0.255/step. If it is
    # still violent, raise THIS term rather than body_ang_vel or action_rate,
    # which are motion blockers and froze the rise from the back.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=-0.2)

    # NO head impact penalty. Tried with the velstand values (body_impact_cost,
    # `neck` subtree, weight -1.0, threshold 2.0): the policy converged to lying
    # down INERT — head_impact_penalty measured -1.01/step, the largest negative
    # term, while standing_composite collapsed from +14.3 to +3.3.
    #
    # The error was believing a "targeted" penalty does not restrain the motion.
    # To get up from its back this robot PIVOTS on its head and shoulders: the
    # head is the fulcrum of the roll-over, not collateral damage, and
    # penalizing it blocks the only available mechanism. If head-slamming
    # returns once the sign bug above is fixed, the fix must be a HEIGHT-GATED
    # penalty (as upright_sharp is) so the ground phase is spared.
    #
    # ⚠️ Beware the lazy optimum that makes this freeze possible: pose_stand_legs
    # stayed at +7.72 out of 8 while the robot was lying down (legs at HOME in
    # lying position → reward collected almost for free). height_stand_l1
    # (weight +30) is what must make "staying on the ground" net negative.

    # ── Start ON THE GROUND: face down / face up / already standing ──────────
    # Added LAST in cfg.events: execution follows insertion order, and this term
    # must overwrite the pose set by reset_base / reset_robot_joints. The
    # "already standing" bucket is not decorative — without it the policy learns
    # to rise but not to HOLD, and falls back right after getting up. No
    # "sitting" bucket, so no sitting_joint_overrides to remap. The
    # probabilities are stage 0 of the ground_state_mix curriculum.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,  # face down (+90° of pitch)
            "face_up_prob": 0.00,  # face up — the hardest, introduced late
            "sitting_prob": 0.00,
            "standing_prob": 0.50,
            "sitting_joint_overrides": None,
            # Face-down and face-up share ONE z range but nothing in their
            # contacts: the belly lifts off at 0.0752, the back rests at 0.0475.
            # 0.076 eliminates belly-side interpenetration (at 0.05 the belly is
            # +25 mm into the ground) at the cost of a back starting 28–42 mm
            # above its rest — a gentler artifact than a contact pushout.
            "prone_z_min": 0.076,
            "prone_z_max": 0.09,
            # Standing on wheels: ROLLER_STAND_Z = 0.138 (vs 0.11–0.12 without wheels).
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # Pitch/roll noise at start. NOTE set_random_ground_state's
            # "standing" bucket reuses the "sitting" quaternion, so this noise
            # ALSO applies to standing starts — intended, it prevents
            # overfitting to a perfectly straight spawn.
            "sitting_tilt_max": math.radians(10),
        },
    )

    # The robot STARTS fallen, so a tilt termination would kill the episode at
    # the first step. The inherited nan_state stays.
    cfg.terminations.pop("fell_over", None)

    # Easy → hard. Under a flat mix the policy optimizes the easy majority and
    # leaves the back under-trained (standup froze into "do nothing" on that
    # pose), so face-up comes late and is biased heavily at the end.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(func=microduck_mdp.event_param_curriculum, params={"event_name": "set_ground_state", "param_stages": [{"step": 0, "params": {"standing_prob": 0.50, "sitting_prob": 0.00, "face_down_prob": 0.50, "face_up_prob": 0.00}}, {"step": 600 * NUM_STEPS_PER_ENV, "params": {"standing_prob": 0.35, "sitting_prob": 0.00, "face_down_prob": 0.45, "face_up_prob": 0.20}}, {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"standing_prob": 0.25, "sitting_prob": 0.00, "face_down_prob": 0.40, "face_up_prob": 0.35}}, {"step": 2500 * NUM_STEPS_PER_ENV, "params": {"standing_prob": 0.20, "sitting_prob": 0.00, "face_down_prob": 0.40, "face_up_prob": 0.40}}]})

    # Force face-up starts so they can be inspected. The curriculum must ALSO be
    # removed: event_param_curriculum runs BEFORE the reset events and would
    # rewrite these probabilities with its stage 0 at the first reset.
    if play:
        play_face_up = _resolve_play_face_up()
        if play_face_up is not None:
            remainder = 1.0 - play_face_up
            cfg.events["set_ground_state"].params.update({"face_up_prob": play_face_up, "face_down_prob": remainder * _PLAY_FACE_DOWN_SHARE, "standing_prob": remainder * (1.0 - _PLAY_FACE_DOWN_SHARE), "sitting_prob": 0.00})
            del cfg.curriculum["ground_state_mix"]

    # ── INVERTED rolling friction: braked → free ─────────────────────────────
    # The heart of the difficulty: the wheels roll, so there is NO longitudinal
    # grip to push off. The roller env RAISES this friction (0 → 0.0015); here
    # we LOWER it, bootstrapping the gesture on an easy problem (nearly locked
    # wheels ≈ feet) before imposing the real physics of rolling.
    #
    # DIAGNOSTIC: if Episode_Reward/standing_composite collapses at a stage, the
    # "grippy feet" gesture does not transfer to free wheels and a skater
    # technique will have to be guided. That is a usable result, not a failure.
    #
    # ⚠️ sim2real: only checkpoints from AFTER the last stage (iter 4000+) are
    # deployment candidates. Before that the policy relies on a rolling friction
    # that does not exist on the real robot.
    _WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)
    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(func=microduck_mdp.wheel_friction_curriculum, params={"event_name": "randomize_wheel_friction", "ranges_stages": [{"step": 0, "ranges": _WHEEL_FRICTION_STAGE0}, {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": (0.0200, 0.0200)}, {"step": 2000 * NUM_STEPS_PER_ENV, "ranges": (0.0080, 0.0080)}, {"step": 3000 * NUM_STEPS_PER_ENV, "ranges": (0.0030, 0.0030)}, {"step": 4000 * NUM_STEPS_PER_ENV, "ranges": (0.0015, 0.0015)}]})
    # Redundant in practice (the curriculum manager runs before the reset events
    # and defaults to stage 0), but keeps the event's DEFAULT consistent with
    # stage 0 if the curriculum is ever removed while the event stays.
    cfg.events["randomize_wheel_friction"].params["ranges"] = _WHEEL_FRICTION_STAGE0

    # ── action_rate: the standup ramp, not the roller one ────────────────────
    # The roller env goes to -2.0 for a calm gait, but that is a motion blocker:
    # it slows the fast action the rise from the back needs (too strong an
    # action_rate killed that recovery in standup). Smoothness is carried here
    # by joint_torque_rate_l2.
    cfg.rewards["action_rate_l2"].weight = -0.6
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.4}, {"step": 250 * NUM_STEPS_PER_ENV, "weight": -0.8}, {"step": 500 * NUM_STEPS_PER_ENV, "weight": -1.0}]})

    # ── Ramped pushes ───────────────────────────────────────────────────────
    # push_robot is inherited from the roller env without a curriculum, and a
    # shove from step 0 disrupts the bootstrap of the rise.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(func=microduck_mdp.push_curriculum, params={"event_name": "push_robot", "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}, {"step": 500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}}, {"step": 1000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}}]})

    return cfg


# ── RL runner config — identical to standup ───────────────────────────────────
MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # the normalizer MUST be baked into the ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
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
        # Symmetry OFF: SYMMETRY_CFG is wired for the old 51D obs layout.
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
