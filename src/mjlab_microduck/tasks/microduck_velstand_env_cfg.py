"""Microduck VelStand environment: walking + fall recovery + body pose control.

A copy of the vel env that uses the full-collision standup XML so the robot
can physically lie down (legs, trunk, head can touch the ground). Trained in
three phases:

  Phase 1 (0 → 500 iters)
    `fell_over` termination active (limit_angle = 70°).
    Same walking objectives as the vel env. Termination prevents the policy
    from exploiting body contacts as "free balance" — it has to learn clean
    walking first.

  Phase 2 (500 → 1500 iters)
    `fell_over` disabled (limit_angle ramped to π).
    Walking rewards still active. Recovery rewards (upright, com_upward_vel,
    com_height_target, impact penalties) provide gradient when fallen, so the
    policy learns to stand back up rather than ending the episode.

  Phase 3 (1500+ iters)
    Body-pose tracking weight + range curriculum kicks in. Same 6D command
    as standup env (x, y, z, roll, pitch, yaw).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


# Phase boundaries (in PPO iterations; env step counter scales by num_steps_per_env=24)
FELL_OVER_DISABLE_ITER = 500
BODY_POSE_KICKIN_ITER  = 1500
# Random prone-init: starts firing at iter PRONE_RAMP_START and reaches
# 2/3 override probability (→ 33% upright / 33% face-down / 33% face-up) at
# iter PRONE_RAMP_END. Combined with face_down_prob=0.5 under the override.
PRONE_RAMP_START       = 1500
PRONE_RAMP_END         = 3000
NUM_STEPS_PER_ENV      = 24

# Body pose final ranges (reached at end of curriculum)
BODY_CMD_MAX_XY        = 0.02                # ±20 mm lateral/forward
BODY_CMD_MAX_Z         = 0.03                # ±30 mm height
BODY_CMD_MAX_ANGLE     = math.radians(30)    # ±30° per Euler axis

# Body pose tracking nominal height — matches the lower bound of the vel-env
# reset_base z range (0.12–0.13), which is where the trunk sits during steady
# upright walking. At cmd_z=0 this gives z_err≈0 → reward saturates at 1, so
# the curriculum only kicks in when a non-zero z command is issued.
BODY_CMD_NOMINAL_HEIGHT = 0.12


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    # Build on top of the vel env so walk rewards / curricula stay in sync
    # automatically — we only add recovery + body-pose extensions here.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # In play mode the curriculum doesn't run (env starts at step 0), so the
    # fall-termination disable wouldn't take effect via the curriculum below.
    # Just delete the termination entirely — setting limit_angle=π would still
    # let bad_orientation fire if any cached-params path bypasses the update.
    if play:
        cfg.terminations.pop("fell_over", None)

    # Switch to the full-collision standup XML. The walk XML has stripped
    # contacts on the trunk/head shells to make falling cheap; the standup XML
    # keeps them, which is what we need for physical body-on-ground recovery.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    # Extra contact sensors for impact penalties during the recovery phase.
    trunk_impact_cfg = ContactSensorCfg(
        name="trunk_impact_contact",
        primary=ContactMatch(mode="body", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    head_impact_cfg = ContactSensorCfg(
        name="head_impact_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, trunk_impact_cfg, head_impact_cfg)

    # ── EVENTS: random prone init (ramped in by curriculum) ─────────────────
    # On reset, with probability `prone_prob`, override the upright orientation
    # set by reset_base with face-down or face-up (50/50). prone_prob starts at
    # 0 and the curriculum ramps it up over iters PRONE_RAMP_START → PRONE_RAMP_END,
    # ending at 2/3 → balanced 33/33/33 mixture of upright/face-down/face-up.
    cfg.events["random_prone_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={"prone_prob": 0.0, "face_down_prob": 0.5},
    )

    # ── REWARDS: fall recovery layer ─────────────────────────────────────────
    # These all sit at high reward when standing normally (free reward while
    # walking) and only meaningfully push the policy when it's fallen.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": 0.115,
        },
    )
    cfg.rewards["com_height_recovery"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=3.0,
        params={"target_height_min": 0.095, "target_height_max": 0.130},
    )
    cfg.rewards["trunk_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-0.1,
        params={"sensor_name": trunk_impact_cfg.name, "threshold": 5.0},
    )
    cfg.rewards["head_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-1.0,
        params={"sensor_name": head_impact_cfg.name, "threshold": 2.0},
    )

    # Body pose tracking: vel env defines this at weight=0.05 (kept-alive only).
    # Override params to match the wider standing ranges + bump nominal height,
    # then set weight to 0 — phase 3 curriculum ramps it up.
    # Body pose tracking — replace the vel-env's body_pose_tracking_6d (which
    # measures x/y/yaw against world spawn origin → kills gradient as soon as
    # the robot walks) with the locomotion-relative version: x/y in body frame
    # relative to feet centroid, yaw relative to feet-pointing direction.
    # Weight starts at 0; curriculum ramps it up at phase 3.
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_locomotion,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": BODY_CMD_NOMINAL_HEIGHT,
            # std ≈ max command on each axis → exp(-1) at full untracked miss.
            # Keeps gradient alive across the whole curriculum range (same
            # recipe that fixed head_pose_tracking).
            "xy_std":    BODY_CMD_MAX_XY,        # 0.02 m
            "z_std":     BODY_CMD_MAX_Z,         # 0.03 m
            "angle_std": BODY_CMD_MAX_ANGLE,     # 30°
            "feet_cfg":  SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        },
    )

    # ── CURRICULUM ───────────────────────────────────────────────────────────
    # Phase 1 → 2: disable fell_over termination at iter 500 by ramping its
    # limit_angle from 70° to 180° (so it never fires).
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

    # Phase 3: ramp body_pose_tracking weight starting at iter 1500.
    cfg.curriculum["body_pose_tracking_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "body_pose_tracking",
            "weight_stages": [
                {"step": 0,                                          "weight": 0.0},
                {"step": BODY_POSE_KICKIN_ITER * NUM_STEPS_PER_ENV,  "weight": 1.0},
                {"step": (BODY_POSE_KICKIN_ITER +  500) * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": (BODY_POSE_KICKIN_ITER + 1000) * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )

    # Random prone-init ramp: 0 until iter PRONE_RAMP_START, then climbs in
    # discrete stages of ~11% each up to 2/3 at iter PRONE_RAMP_END
    # (= 33% upright / 33% face-down / 33% face-up).
    # The intermediate stages make the rise visible in wandb instead of a
    # single jump halfway through the window.
    _prone_target = 2.0 / 3.0
    _prone_stages_n = 6  # number of equal increments inside the ramp window
    _prone_ramp_iters = PRONE_RAMP_END - PRONE_RAMP_START
    cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "random_prone_init",
            "param_stages": [
                {"step": 0, "params": {"prone_prob": 0.0, "face_down_prob": 0.5}},
                # Ramp stages: iter PRONE_RAMP_START + k*ramp/N → prob = k*target/N
                *(
                    {
                        "step": (PRONE_RAMP_START + (k * _prone_ramp_iters) // _prone_stages_n) * NUM_STEPS_PER_ENV,
                        "params": {"prone_prob": k * _prone_target / _prone_stages_n, "face_down_prob": 0.5},
                    }
                    for k in range(1, _prone_stages_n + 1)
                ),
            ],
        },
    )

    # Phase 3: widen body_pose ranges (overrides the vel env's "kept-alive"
    # curriculum, which only ever stays at ±5 mm / ±3°).
    cfg.curriculum["body_pose_range"].params["range_stages"] = [
        {"step": 0, "ranges": (
            (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005),
            (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05),
        )},
        {"step": BODY_POSE_KICKIN_ITER * NUM_STEPS_PER_ENV, "ranges": (
            (-0.010, 0.010), (-0.010, 0.010), (-0.015, 0.015),
            (-math.radians(15), math.radians(15)),
            (-math.radians(15), math.radians(15)),
            (-math.radians(15), math.radians(15)),
        )},
        {"step": (BODY_POSE_KICKIN_ITER + 1000) * NUM_STEPS_PER_ENV, "ranges": (
            (-BODY_CMD_MAX_XY, BODY_CMD_MAX_XY),
            (-BODY_CMD_MAX_XY, BODY_CMD_MAX_XY),
            (-BODY_CMD_MAX_Z,  BODY_CMD_MAX_Z),
            (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
            (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
            (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
        )},
    ]

    return cfg


MicroduckVelStandRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
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
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=20_000,
)
