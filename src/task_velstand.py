# Curriculum phases: clean walking first (fell_over active), then falls become
# recovery opportunities (fell_over disabled, fallen_too_long recycles failures),
# then prone/crouch inits ramp in. Prone is capped at 45% so the walking data share
# stays above ~55% — an earlier design ran 2/3 prone resets and starved the walk to
# ~25% of experience.

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from . import task_mdp as microduck_mdp
from .robot import MICRODUCK_STANDUP_ROBOT_CFG
from .task_symmetry import PpoWithSymmetryCfg
from .task_velocity import LOCAL_CHECKPOINTS_ONLY, make_microduck_velocity_env_cfg

FELL_OVER_DISABLE_ITER = 500
NUM_STEPS_PER_ENV = 24

# Recovery REWARDS gate on TILT ONLY. Adding a low-height condition let SITTING
# (z≈0.07, trunk upright) open the gate, and the policy sat there farming
# upright_linear — gating a positive reward on a bad state rewards entering it.
# Tilt > 40° cannot be farmed from a comfortable pose.
REWARD_GATE_TILT_DEG = 40.0
# The TERMINATION keeps a z-gate so sitters and stuck-low envs get recycled instead
# of paid. 0.08, not 0.10: a normally wobbling upright robot dips to 0.084-0.096, so
# 0.10 sits inside the early-learning envelope and recycled crouch-walking explorers.
# 0.08 still catches sitting (≈0.07) and prone (≈0.05).
TERM_GATE_Z = 0.08
TERM_GATE_TILT_DEG = 40.0

# "Recovery complete", shared by the recovery_success bounty and the fallen_tax
# release. z must sit INSIDE the policy's real standing envelope (measured
# 0.084-0.096 wobbling, ≈0.117 at the full STAND keyframe): an earlier 0.105
# demanded standing taller than the policy ever is, so the bounty never fired and
# recoveries parked in a deep crouch just past the 40° gates, where every dense
# recovery term stops paying. 0.09 is reachable yet still 2 cm above sitting.
RECOVERED_UP_TILT_DEG = 25.0
RECOVERED_UP_Z = 0.09

# Tax and bounty stay off until here. Not about walk stability (the walk is stable by
# ~750): the gap after fell_over is disabled at 500 buys a TAX-FREE window where
# natural-fall get-up attempts cost nothing and the dense progress terms can teach
# them. Enabling the tax at 800 instead made hopeless prone episodes bleed -0.5/step
# for the whole timeout, and prone recovery collapsed to 0% while PPO chased the
# easier crouch-slice reward.
RECOVERY_ECON_KICKIN_ITER = 1200

# Continuously fallen this long → reset. At 5 s a face-down recovery spent most of
# its budget just reaching the deep crouch and got recycled right at the frontier,
# starving the crouch→stand last mile of on-policy data.
FALLEN_TIMEOUT_S = 8.0

# crouch_prob is a reverse-curriculum slice: envs reset directly into random
# mid-recovery crouches, so the last mile gets dense data instead of only being
# reached at the tail of rare good rollouts. It starts before the economics because
# near-upright states are harmless untaxed and double as stand-tall posture data.
# Prone starts only AFTER the tax-free window above.
PRONE_RAMP_STAGES = [{"step": 0, "params": {"prone_prob": 0.00, "face_down_prob": 1.0, "crouch_prob": 0.00}}, {"step": 800 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.00, "face_down_prob": 1.0, "crouch_prob": 0.15}}, {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.15, "face_down_prob": 0.80, "crouch_prob": 0.15}}, {"step": 2000 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.30, "face_down_prob": 0.65, "crouch_prob": 0.15}}, {"step": 2500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "crouch_prob": 0.15}}]


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # The curriculum does not run in play, so the disable below never fires.
    if play:
        cfg.terminations.pop("fell_over", None)

    # Trunk/head shells keep their contacts, so the robot can lie down and push off.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    # head_pose_bias arrives ungated, which is fine where fell_over ends fallen
    # episodes. Velstand episodes SURVIVE falls, so the ungated EMA would charge head
    # droop through the whole ground phase — a flat tax on being fallen that the
    # recovery economics never priced in. Gating it keeps the term pricing only
    # sustained droop while actually standing.
    cfg.rewards["head_pose_bias"].params.update({"gate_height_low": 0.09, "gate_height_high": 0.11, "gate_tilt_full_deg": 20.0, "gate_tilt_zero_deg": REWARD_GATE_TILT_DEG})

    # ANY positive reward for BEING in a fallen-ish state gets farmed from some
    # comfortable pose (observed: sitting, lying, a head-tripod at ~55°). So this is
    # POTENTIAL-BASED (Δcos tilt): rising pays, falling costs, holding pays zero.
    # Unfarmable and ungated, and it also rewards catching a stumble while walking.
    cfg.rewards["upright_progress"] = RewardTermCfg(func=microduck_mdp.upright_progress, weight=5.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    # z-axis companion: the crouch→stand last mile is mostly a HEIGHT change at modest
    # tilt, where Δcos(tilt) is tiny and the Gaussian upright/pose rewards are flat.
    # Same potential-based construction. A full prone→stand rise collects ≈+2, the
    # crouch→stand mile ≈+1.
    cfg.rewards["height_progress"] = RewardTermCfg(func=microduck_mdp.height_progress, weight=30.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "ceiling": 0.115})
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.0,  # ramped in at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            # Slightly above standing so the rising reward pays until fully up; the
            # fallen gate, not this ceiling, is what stops gait-bounce farming.
            "max_height": 0.125,
            # Never triggers: the gate is tilt-only, see REWARD_GATE_TILT_DEG.
            "gate_z_below": 0.0,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # ⚠️ Deliberately NO impact penalties: recovery pushes off with head and trunk, and
    # a head penalty (-1.0 @ 2 N) taxed exactly that, leaving falls cheaper than
    # getting up. This term covers landing harshness instead — it penalizes torque
    # CHANGE, not magnitude or rotation, so it never blocks the recovery flip.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=-2e-3)

    # Zeroed while fallen: a robot lying on its trunk can rhythmically tap its feet
    # through the swing window and farm this.
    at = cfg.rewards["air_time"]
    at_params = dict(at.params)
    cfg.rewards["air_time"] = RewardTermCfg(func=microduck_mdp.feet_air_time_upright, weight=at.weight, params={**at_params, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG})
    # Lying still must be strictly worse than trying: without this, waiting out the
    # fallen_too_long recycle was rational, since attempts cost action-rate and torque
    # penalties while waiting cost nothing.
    cfg.rewards["fallen_tax"] = RewardTermCfg(
        func=microduck_mdp.fallen_state_penalty,
        weight=0.0,  # ramped to -0.5 at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            # Hysteresis: recoveries parked in a deep crouch just under the 40° gate,
            # past every recovery term's gate but short of standing. Releasing only on
            # the recovery_success conditions keeps a fall taxed until the stand is
            # FINISHED. Arms only above 40°, so normal gait is never taxed.
            "release_tilt_below_deg": RECOVERED_UP_TILT_DEG,
            "release_z_above": RECOVERED_UP_Z,
        },
    )
    # One-shot bounty on a COMPLETED recovery, with hysteresis so gate-oscillation
    # pays nothing — the endpoint signal the dense gated terms lack.
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=microduck_mdp.recovery_success,
        weight=0.0,  # ramped to +10 at RECOVERY_ECON_KICKIN_ITER
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "fallen_tilt_deg": REWARD_GATE_TILT_DEG, "min_fallen_s": 0.5, "up_tilt_deg": RECOVERED_UP_TILT_DEG, "up_z": RECOVERED_UP_Z},
    )

    # Spawn just above the ground: the function defaults are 0.20-0.25 m, which opens
    # every prone episode with a 15-20 cm free fall (face-down trunk rests at ~0.044).
    cfg.events["random_prone_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.0,  # ramped by the prone_init_prob curriculum
            "face_down_prob": 1.0,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,  # ramped by the prone_init_prob curriculum
        },
    )

    cfg.terminations["fallen_too_long"] = TerminationTermCfg(func=microduck_mdp.fallen_too_long, time_out=False, params={"gate_z_below": TERM_GATE_Z, "gate_tilt_above_deg": TERM_GATE_TILT_DEG, "max_duration_s": FALLEN_TIMEOUT_S})

    if not play:
        cfg.curriculum["fell_over_disable"] = CurriculumTermCfg(func=microduck_mdp.termination_param_curriculum, params={"term_name": "fell_over", "param_stages": [{"step": 0, "params": {"limit_angle": math.radians(70.0)}}, {"step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV, "params": {"limit_angle": math.pi}}]})

    cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(func=microduck_mdp.event_param_curriculum, params={"event_name": "random_prone_init", "param_stages": PRONE_RAMP_STAGES})

    cfg.curriculum["fallen_tax_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "fallen_tax", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.5}]})
    cfg.curriculum["recovery_success_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "recovery_success", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 10.0}]})
    cfg.curriculum["com_upward_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "com_upward_velocity", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 2.0}]})

    return cfg


MicroduckVelStandRlCfg = RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}), critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True), algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None), logger=LOCAL_CHECKPOINTS_ONLY, experiment_name="velstand", run_name="velstand", save_interval=250, num_steps_per_env=24, max_iterations=20_000)
