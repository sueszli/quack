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
NUM_STEPS_PER_ENV      = 24

# Body pose final ranges (reached at end of curriculum)
BODY_CMD_MAX_XY        = 0.02                # ±20 mm lateral/forward
BODY_CMD_MAX_Z         = 0.03                # ±30 mm height
BODY_CMD_MAX_ANGLE     = math.radians(30)    # ±30° per Euler axis

# Body pose tracking nominal height — between vel-env walking height (~0.095)
# and standup-env standing height (0.115). Tracking error is dominated by xy
# and angle anyway, so the exact z reference matters less than the gradient.
BODY_CMD_NOMINAL_HEIGHT = 0.105


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    # Build on top of the vel env so walk rewards / curricula stay in sync
    # automatically — we only add recovery + body-pose extensions here.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

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
    cfg.rewards["body_pose_tracking"].weight = 0.0
    cfg.rewards["body_pose_tracking"].params.update({
        "nominal_height": BODY_CMD_NOMINAL_HEIGHT,
        "xy_std": 0.02,
        "z_std": 0.01,
        "angle_std": math.radians(10),
    })

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
