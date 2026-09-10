import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from . import task_mdp as microduck_mdp
from .task_symmetry import PpoWithSymmetryCfg
from .task_velocity import LOCAL_CHECKPOINTS_ONLY
from .task_velocity_rollers import make_microduck_velocity_rollers_env_cfg

# Trunk heights (m), measured by exact kinematics on scene_rollers.xml (STAND pose,
# trunk lowered to contact): standing 0.1407, face down 0.0752, face up 0.0475.
# STAND_Z drops ~2 mm off the kinematic value for sag under load.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S = 6.0
NUM_STEPS_PER_ENV = 24

# At play the step counter restarts at 0, so ground_state_mix applies stage 0 and
# face-up starts — the hardest case, the one worth watching — never appear.
# STANDUP_PLAY_FACE_UP overrides; play only.
PLAY_FACE_UP = None
# Splits the remainder 2:1 face-down:standing, the last curriculum stage's ratio.
_PLAY_FACE_DOWN_SHARE = 2.0 / 3.0


def _resolve_play_face_up():
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


# The passive wheels are INTERLEAVED in the rollers joint order, so standup's
# [0-4, 9-13] indices do NOT hold here. Locked by tests/test_roller_standup_cfg.py.
# Only _LEG_JOINTS is consumed; the other two exist for that test (the neck resolves
# by name, the wheels by the ^passive_.* regex).
_LEG_JOINTS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS = [7, 8, 9, 10]
_WHEEL_JOINTS = [5, 6, 16, 17]

# Skating rewards, meaningless on the ground and actively fighting the rise:
# the blades are not flat during it and getting up needs the legs spread.
_SKATING_REWARDS = ("wheel_speed", "braking", "skating_air_time", "glide", "single_support", "gait_symmetry", "forward_lean", "heading_hold", "feet_flat", "hip_roll_neutral", "pose", "com_height_target", "upright")


def make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S

    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # Nothing is steered here, but the slot keeps a tiny non-zero range so its input
    # neurons stay alive; head_pose and body_pose stay zero-padded (61D obs parity).
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

    # A rare contact (~1/25M steps) diverges the free joint to NaN; sanitizing the obs
    # keeps training alive and the offending env resets on the next step.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # A FRESH SceneEntityCfg per term: mjlab resolves and mutates them in place, so a
    # shared object hands out stale indices.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(func=microduck_mdp.pose_target_match, weight=8.0, params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None})
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(func=microduck_mdp.pose_l1_penalty, weight=5.0, params={"joint_indices": _LEG_JOINTS, "target_overrides": None})

    # Three layers: wide Gaussian pulls off the ground, narrow one forces the last cm,
    # and the L1 makes lying still net NEGATIVE — without it that is the lazy optimum.
    cfg.rewards["height_stand"] = RewardTermCfg(func=microduck_mdp.height_target_gaussian, weight=4.0, params={"std": 0.04, "target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(func=microduck_mdp.height_target_gaussian, weight=4.0, params={"std": 0.015, "target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["height_stand_l1"] = RewardTermCfg(func=microduck_mdp.height_l1_penalty, weight=30.0, params={"target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Cutoff sits 10 mm ABOVE the target: at the target the policy parks there instead
    # of finishing the rise.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(func=microduck_mdp.com_upward_velocity, weight=3.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "max_height": ROLLER_STAND_Z + 0.010})
    # ⚠️ POSITIVE weight, not a typo: trunk_vertical_accel_penalty self-negates
    # (returns -|a_z|), so a negative weight double-negates into a REWARD for
    # violence. Magnitude stays small on purpose — |a_z| is necessarily high in a
    # roll-over from the back, so a big weight here blocks the motion. Damping is
    # joint_torque_rate_l2's job.
    cfg.rewards["gentle_rise"] = RewardTermCfg(func=microduck_mdp.trunk_vertical_accel_penalty, weight=+0.02, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Two layers: cos(tilt) has gradient while lying down but flattens near vertical,
    # where the height-gated Gaussian takes over and kills the backward lean (standup
    # tipped over backwards while extending the legs).
    cfg.rewards["upright_linear"] = RewardTermCfg(func=microduck_mdp.body_upright_linear, weight=6.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["upright_sharp"] = RewardTermCfg(func=microduck_mdp.upright_gaussian_at_height, weight=6.0, params={"std": 0.3, "height_low": ROLLER_PRONE_Z, "height_high": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Stds deliberately WIDE: tight ones scored ~5e-5, i.e. no gradient at all.
    cfg.rewards["standing_composite"] = RewardTermCfg(func=microduck_mdp.standing_composite_score, weight=15.0, params={"target_height": ROLLER_STAND_Z, "height_std": 0.04, "upright_std": 0.40, "pose_std": 0.40, "joint_indices": _LEG_JOINTS, "target_overrides": None, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Penalizes torque VARIATION, not amplitude or rotation, so it damps trembling
    # without blocking the roll-over — the only damper that does not kill the rise
    # from the back, hence THE safe lever to raise. |Δτ|² is ~0.1 at convergence, so
    # the contribution is ~0.1 × |weight|. Prefer this over body_ang_vel or
    # action_rate, which are motion blockers.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=-0.2)

    # ⚠️ Deliberately NO head impact penalty: rising from the back PIVOTS on the head,
    # so the head is the fulcrum, not collateral damage. Tried at -1.0 / 2 N and the
    # policy converged to lying still. If head-slamming reappears, gate the penalty by
    # HEIGHT (as upright_sharp is) so the ground phase is spared.
    # ⚠️ pose_stand_legs pays +7.72/8 to a robot lying with its legs at HOME, so
    # height_stand_l1 is what has to make staying down net negative.

    # Inserted LAST: events run in insertion order and this must overwrite the pose
    # from reset_base / reset_robot_joints. The "already standing" bucket is load
    # bearing — without it the policy rises but never learns to HOLD.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,
            "face_up_prob": 0.00,
            "sitting_prob": 0.00,
            "standing_prob": 0.50,
            "sitting_joint_overrides": None,
            # Face-down and face-up share ONE z range but rest at different heights
            # (belly lifts at 0.0752, back at 0.0475). 0.076 avoids belly
            # interpenetration (0.05 sank 25 mm into the ground) at the cost of a back
            # starting 28-42 mm high — the gentler artifact.
            "prone_z_min": 0.076,
            "prone_z_max": 0.09,
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # set_random_ground_state reuses the sitting quaternion for the standing
            # bucket, so this noise applies to standing starts too. Intended.
            "sitting_tilt_max": math.radians(10),
        },
    )

    # The robot starts fallen, so a tilt termination would end every episode at step 1.
    cfg.terminations.pop("fell_over", None)

    # Easy → hard. Under a flat mix the policy optimizes the easy majority and leaves
    # face-up under-trained (standup froze into "do nothing" on it), so face-up arrives
    # late and ends up weighted heaviest.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(func=microduck_mdp.event_param_curriculum, params={"event_name": "set_ground_state", "param_stages": [{"step": 0, "params": {"standing_prob": 0.50, "sitting_prob": 0.00, "face_down_prob": 0.50, "face_up_prob": 0.00}}, {"step": 600 * NUM_STEPS_PER_ENV, "params": {"standing_prob": 0.35, "sitting_prob": 0.00, "face_down_prob": 0.45, "face_up_prob": 0.20}}, {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"standing_prob": 0.25, "sitting_prob": 0.00, "face_down_prob": 0.40, "face_up_prob": 0.35}}, {"step": 2500 * NUM_STEPS_PER_ENV, "params": {"standing_prob": 0.20, "sitting_prob": 0.00, "face_down_prob": 0.40, "face_up_prob": 0.40}}]})

    # The curriculum must go too: it runs BEFORE the reset events and would rewrite
    # these probabilities with its stage 0 on the first reset.
    if play:
        play_face_up = _resolve_play_face_up()
        if play_face_up is not None:
            remainder = 1.0 - play_face_up
            cfg.events["set_ground_state"].params.update({"face_up_prob": play_face_up, "face_down_prob": remainder * _PLAY_FACE_DOWN_SHARE, "standing_prob": remainder * (1.0 - _PLAY_FACE_DOWN_SHARE), "sitting_prob": 0.00})
            del cfg.curriculum["ground_state_mix"]

    # INVERTED curriculum, braked → free: the wheels roll, so there is no grip to push
    # off. Bootstraps the gesture on nearly locked wheels (≈ feet) before imposing real
    # rolling physics. If standing_composite collapses at a stage, the grippy-feet
    # gesture does not transfer and a skater technique has to be guided instead.
    # ⚠️ sim2real: only checkpoints after the LAST stage (iter 4000+) are deployable —
    # earlier ones lean on a rolling friction the real robot does not have.
    _WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)
    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(func=microduck_mdp.wheel_friction_curriculum, params={"event_name": "randomize_wheel_friction", "ranges_stages": [{"step": 0, "ranges": _WHEEL_FRICTION_STAGE0}, {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": (0.0200, 0.0200)}, {"step": 2000 * NUM_STEPS_PER_ENV, "ranges": (0.0080, 0.0080)}, {"step": 3000 * NUM_STEPS_PER_ENV, "ranges": (0.0030, 0.0030)}, {"step": 4000 * NUM_STEPS_PER_ENV, "ranges": (0.0015, 0.0015)}]})
    # Redundant in practice (the curriculum already defaults to stage 0); keeps the
    # event honest if the curriculum is ever removed.
    cfg.events["randomize_wheel_friction"].params["ranges"] = _WHEEL_FRICTION_STAGE0

    # The roller env's -2.0 is a motion blocker here: it slows the fast action the rise
    # from the back needs. Smoothness comes from joint_torque_rate_l2 instead.
    cfg.rewards["action_rate_l2"].weight = -0.6
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.4}, {"step": 250 * NUM_STEPS_PER_ENV, "weight": -0.8}, {"step": 500 * NUM_STEPS_PER_ENV, "weight": -1.0}]})

    # Inherited pushes have no curriculum, and a shove from step 0 wrecks the rise's
    # bootstrap.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(func=microduck_mdp.push_curriculum, params={"event_name": "push_robot", "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}, {"step": 500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}}, {"step": 1000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}}]})

    return cfg


MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # export.py MUST bake the normalizer into the ONNX
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
        # OFF: SYMMETRY_CFG is wired for the old 51D layout and breaks on 61D.
        symmetry_cfg=None,
    ),
    logger=LOCAL_CHECKPOINTS_ONLY,
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
