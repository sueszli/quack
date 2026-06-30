"""Microduck VelStand-TipToe environment: walk on tiptoes 🐤.

Same scene / curriculum as the VelStand env, plus a `feet_tiptoe_alignment`
reward that pushes each foot site's local x-axis to point downward while
the robot is commanded to move. Effect: heels stay up, weight on the toes
during locomotion.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
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

# Toggle body-pose tracking. When enabled, the reward is gated on linear
# velocity command magnitude (only active when the robot is supposed to be
# standing still), and curriculum ramps the weight up after the policy has
# learned the walking + recovery basics.
ENABLE_BODY_TRACKING   = False

# Toggle for random prone initialization (episodes start face-down/up with
# probability ramping from 0 at PRONE_RAMP_START → 2/3 at PRONE_RAMP_END,
# giving a 33/33/33 split of upright/face-down/face-up resets). Useful for
# bootstrapping fall recovery; disable to focus on walking first.
ENABLE_PRONE_INIT      = False
PRONE_RAMP_START       = 1500
PRONE_RAMP_END         = 3000

# Body pose final ranges (reached at end of curriculum)
BODY_CMD_MAX_XY        = 0.01                # ±10 mm lateral/forward (was 20 mm —
                                              # 20 mm body lean while walking is
                                              # mechanically too disruptive; the
                                              # policy converged to "ignore xy"
                                              # because the walking cost outweighed
                                              # the tracking gain)
BODY_CMD_MAX_Z         = 0.03                # ±30 mm height
BODY_CMD_MAX_ANGLE     = math.radians(30)    # ±30° per Euler axis

# Body pose tracking nominal height — matches the lower bound of the vel-env
# reset_base z range (0.12–0.13), which is where the trunk sits during steady
# upright walking. At cmd_z=0 this gives z_err≈0 → reward saturates at 1, so
# the curriculum only kicks in when a non-zero z command is issued.
BODY_CMD_NOMINAL_HEIGHT = 0.12


def make_microduck_velstand_tiptoe_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
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
    if ENABLE_PRONE_INIT:
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
    # Recovery-only height reward: rewards being "near standing height" regardless
    # of body_pose z command. Range covers the full body_pose z command sweep
    # (nominal_height ± BODY_CMD_MAX_Z ≈ 0.09–0.15) plus margin, so commanding
    # the body up or down doesn't fight this reward. Below ~0.08 means
    # genuinely fallen → 0 reward, which is what bootstraps fall recovery.
    cfg.rewards["com_height_recovery"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=3.0,
        params={"target_height_min": 0.08, "target_height_max": 0.16},
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

    # Tiptoe alignment — reward each foot's local x-axis pointing downward
    # while the robot is commanded to move. Sum over both feet ∈ [-2, 2];
    # weight 5.0 makes max contribution ≈ 10/step — dominant signal needed to
    # pull the policy out of the flat-foot walking basin (1.0 was tried first,
    # converged to ~3% alignment because flat-walk is locally much easier).
    cfg.rewards["feet_tiptoe"] = RewardTermCfg(
        func=microduck_mdp.feet_tiptoe_alignment,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )

    # Body pose tracking, GATED on linear velocity command magnitude.
    # When the robot is commanded to stand still (|vel_cmd_xy| ≈ 0) the gate
    # opens and the policy is rewarded for matching the body_pose command.
    # When commanded to walk the gate closes (≈0) and tracking has no effect.
    # This sidesteps the body-vs-walking gradient conflict that prevented
    # earlier runs from learning either well.
    #
    # xy tracking is masked out because trunk x/y lean is mechanically coupled
    # to pitch/roll on this robot (leaning forward shifts trunk AND pitches
    # it), so independent xy commands produce noise rather than useful gradient.
    if ENABLE_BODY_TRACKING:
        cfg.rewards["body_pose_tracking"] = RewardTermCfg(
            func=microduck_mdp.body_pose_tracking_locomotion,
            weight=0.0,  # curriculum ramps this up; tracking only active when standing
            params={
                "command_name": "body_pose",
                "nominal_height": BODY_CMD_NOMINAL_HEIGHT,
                "xy_std":    BODY_CMD_MAX_XY    / 2.0,
                "z_std":     BODY_CMD_MAX_Z     / 2.0,
                "angle_std": BODY_CMD_MAX_ANGLE / 2.0,
                "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 1.0),  # z, roll, pitch, yaw only
                "vel_gate_command_name": "twist",
                "vel_gate_std": 0.05,   # gate≈1 at |vel_cmd_xy|=0; ≈0 by |vel_cmd_xy|≥0.15 m/s
                "feet_cfg":  SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            },
        )
    else:
        cfg.rewards["body_pose_tracking"].weight = 0.0

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

    # body_pose_tracking weight curriculum REMOVED — body tracking disabled
    # for now. Keep the body_pose command term + obs slot for shape parity.

    # Push velocity stays at the vel env's inherited ±0.3 m/s — no curriculum.

    # Random prone-init ramp: 0 until iter PRONE_RAMP_START, then climbs in
    # discrete stages of ~11% each up to 2/3 at iter PRONE_RAMP_END
    # (= 33% upright / 33% face-down / 33% face-up).
    if ENABLE_PRONE_INIT:
        _prone_target = 2.0 / 3.0
        _prone_stages_n = 6
        _prone_ramp_iters = PRONE_RAMP_END - PRONE_RAMP_START
        cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "random_prone_init",
                "param_stages": [
                    {"step": 0, "params": {"prone_prob": 0.0, "face_down_prob": 0.5}},
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

    if ENABLE_BODY_TRACKING:
        # Weight ramp: 0 until BODY_POSE_KICKIN_ITER, then climbs hard. With
        # the gate, only ~25% of training samples (the standing envs) actually
        # contribute body-tracking gradient — to compensate, the weight is set
        # high enough that the gradient on those samples is strong.
        # NB: do NOT bump standing_envs ratio to compensate — prior runs showed
        # that >25% standing envs make the policy forget how to walk.
        cfg.curriculum["body_pose_tracking_weight"] = CurriculumTermCfg(
            func=velocity_mdp.reward_weight,
            params={
                "reward_name": "body_pose_tracking",
                "weight_stages": [
                    {"step": 0,                                                  "weight": 0.0},
                    {"step": BODY_POSE_KICKIN_ITER         * NUM_STEPS_PER_ENV,  "weight": 4.0},
                    {"step": (BODY_POSE_KICKIN_ITER +  500) * NUM_STEPS_PER_ENV, "weight": 6.0},
                    {"step": (BODY_POSE_KICKIN_ITER + 1000) * NUM_STEPS_PER_ENV, "weight": 8.0},
                ],
            },
        )

        # body_pose range widens at phase 3 (overrides vel env's kept-alive range).
        # xy ranges stay tiny since axis_weights mask them out anyway.
        cfg.curriculum["body_pose_range"].params["range_stages"] = [
            {"step": 0, "ranges": (
                (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005),
                (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05),
            )},
            {"step": BODY_POSE_KICKIN_ITER * NUM_STEPS_PER_ENV, "ranges": (
                (-0.005, 0.005), (-0.005, 0.005), (-0.015, 0.015),
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


MicroduckVelStandTipToeRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="velstand_tiptoe",
    run_name="velstand_tiptoe",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=20_000,
)
