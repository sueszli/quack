"""Microduck VelStand: walking + fall recovery in one policy.

Walk layer = make_microduck_velocity_env_cfg verbatim, on the all-collision standup XML.
Recovery is a small reward layer gated on being genuinely fallen (tilt > 40°): zero during
clean walking. Orientation/height progress are POTENTIAL-based (Δcos tilt, Δz) so no fallen
pose can be farmed; a fallen tax with hysteresis and a one-shot completion bounty supply the
economics; a crouch reverse-curriculum slice gives the crouch→stand last mile dense data.

Lesson arc: (1) gating positive rewards on low height let the policy sit and farm them —
gate on tilt only; (2) a tax from step 0 froze the walk in a crouch — recovery economics
start after a tax-free natural-fall window (fell_over off at 500, econ on at 1200); (3) the
completion bounty must be reachable inside the real standing envelope (z 0.084–0.096), not
the keyframe height; (4) prone recovery only bootstraps if prone spawns come AFTER econ,
which comes after the tax-free window — turning everything on at 800 collapsed it to 0%.

Phases: 0–500 fell_over active (clean walk first); 500+ fell_over off, fallen_too_long
recycles failed recoveries; 800+ crouch slice; 1200+ tax + bounty; 1500+ prone ramp
(face-down first, capped at 45% so the walking data share stays ≥ 55%).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

FELL_OVER_DISABLE_ITER = 500
NUM_STEPS_PER_ENV = 24

# Recovery REWARDS gate on tilt only (a z gate let sitting farm them). The TERMINATION
# keeps a z condition so sitters and stuck-low envs are recycled, not paid; 0.08 stays
# below the normal wobbling envelope (0.084–0.096) yet above sitting (≈0.07).
REWARD_GATE_TILT_DEG = 40.0
TERM_GATE_Z = 0.08
TERM_GATE_TILT_DEG = 40.0

# "Recovery complete", shared by the bounty and the fallen_tax release. z must be inside
# the real standing envelope: 0.105 was never reached and recoveries parked in a crouch.
RECOVERED_UP_TILT_DEG = 25.0
RECOVERED_UP_Z = 0.09

# Tax + bounty kick in after a tax-free natural-fall window (see module docstring).
RECOVERY_ECON_KICKIN_ITER = 1200

# Continuously fallen this long → reset. 5 s recycled face-down recoveries right at the
# crouch frontier, starving the last mile of data.
FALLEN_TIMEOUT_S = 8.0

# Prone capped at 45% (2/3 starved the walk); face-down first. crouch_prob spawns directly
# into mid-recovery crouches (reverse curriculum), tax-free until econ.
PRONE_RAMP_STAGES = [
    {"step": 0,                        "params": {"prone_prob": 0.00, "face_down_prob": 1.0,  "crouch_prob": 0.00}},
    {"step": 800 * NUM_STEPS_PER_ENV,  "params": {"prone_prob": 0.00, "face_down_prob": 1.0,  "crouch_prob": 0.15}},
    {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.15, "face_down_prob": 0.80, "crouch_prob": 0.15}},
    {"step": 2000 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.30, "face_down_prob": 0.65, "crouch_prob": 0.15}},
    {"step": 2500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "crouch_prob": 0.15}},
]


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # Curricula don't run in play, so the fell_over disable below never fires.
    if play:
        cfg.terminations.pop("fell_over", None)

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    # Velstand episodes survive falls, so gate head_pose_bias on upright or it becomes a
    # flat tax on being fallen.
    cfg.rewards["head_pose_bias"].params.update({
        "gate_height_low":    0.09,
        "gate_height_high":   0.11,
        "gate_tilt_full_deg": 20.0,
        "gate_tilt_zero_deg": REWARD_GATE_TILT_DEG,
    })

    # Potential-based Δcos(tilt): any positive reward for BEING fallen-ish got farmed
    # (sitting, lying, head-tripod); rising pays, holding pays zero.
    cfg.rewards["upright_progress"] = RewardTermCfg(
        func=microduck_mdp.upright_progress,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # Potential-based Δz: the crouch→stand last mile is a height change at modest tilt
    # where Δcos(tilt) is tiny. Full prone→stand collects ≈ +2.
    cfg.rewards["height_progress"] = RewardTermCfg(
        func=microduck_mdp.height_progress,
        weight=30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ceiling": 0.115,
        },
    )
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.0,  # ramped in at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": 0.125,
            "gate_z_below": 0.0,  # tilt-only gate
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # No impact penalties: recovery pushes off with head/trunk and a head penalty taxed
    # exactly that. Torque-rate covers landing harshness without blocking the flip.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-2e-3,
    )

    # air_time zeroed while fallen: a robot on its trunk tapped its feet through the window.
    at = cfg.rewards["air_time"]
    at_params = dict(at.params)
    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.feet_air_time_upright,
        weight=at.weight,
        params={**at_params, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )
    # Lying still must be strictly worse than trying (waiting for the recycle was rational).
    # Hysteresis: arms on tilt > 40°, releases only when the stand is FINISHED, so the
    # sub-40° crouch is no longer a free rest state.
    cfg.rewards["fallen_tax"] = RewardTermCfg(
        func=microduck_mdp.fallen_state_penalty,
        weight=0.0,  # ramped to -0.5 at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            "release_tilt_below_deg": RECOVERED_UP_TILT_DEG,
            "release_z_above": RECOVERED_UP_Z,
        },
    )
    # One-shot bounty on a completed recovery, with hysteresis so gate oscillation pays nothing.
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=microduck_mdp.recovery_success,
        weight=0.0,  # ramped to +10 at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "fallen_tilt_deg": REWARD_GATE_TILT_DEG,
            "min_fallen_s": 0.5,
            "up_tilt_deg": RECOVERED_UP_TILT_DEG,
            "up_z": RECOVERED_UP_Z,
        },
    )

    # Prone z just above the measured 0.044 m rest (the function defaults free-fall 15 cm).
    cfg.events["random_prone_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.0,        # ramped by prone_init_prob
            "face_down_prob": 1.0,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,       # ramped by prone_init_prob
        },
    )

    cfg.terminations["fallen_too_long"] = TerminationTermCfg(
        func=microduck_mdp.fallen_too_long,
        time_out=False,
        params={
            "gate_z_below": TERM_GATE_Z,
            "gate_tilt_above_deg": TERM_GATE_TILT_DEG,
            "max_duration_s": FALLEN_TIMEOUT_S,
        },
    )

    # Disable fell_over at iter 500 so falls become recovery training.
    if not play:
        cfg.curriculum["fell_over_disable"] = CurriculumTermCfg(
            func=microduck_mdp.termination_param_curriculum,
            params={
                "term_name": "fell_over",
                "param_stages": [
                    {"step": 0,
                     "params": {"limit_angle": math.radians(70.0)}},
                    {"step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV,
                     "params": {"limit_angle": math.pi}},
                ],
            },
        )

    cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "random_prone_init",
            "param_stages": PRONE_RAMP_STAGES,
        },
    )

    cfg.curriculum["fallen_tax_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "fallen_tax",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.5},
            ],
        },
    )
    cfg.curriculum["recovery_success_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "recovery_success",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 10.0},
            ],
        },
    )
    cfg.curriculum["com_upward_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "com_upward_velocity",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 2.0},
            ],
        },
    )

    return cfg


MicroduckVelStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
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
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velstand",
    run_name="velstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
