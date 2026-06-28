"""Microduck Velocity2 environment — microban-recipe variant of the velocity task.

Built on top of `make_microduck_velocity_env_cfg` (so DR, observations, command
infra and curricula stay in sync) but with the reward/regularization recipe
swapped to match mjlab_microban's leaner, locomotion-focused setup:

  - tracking + posture/orientation: exactly microban (weights + std)
  - removed microduck-only posture extras: com_height_target, stillness_at_zero_command
  - gait/feet: exactly microban (air_time 3.0 @ 0.125–0.300 s, soft_landing deleted)
  - foot_slip: microban (-1.0)
  - added no_stepping penalty (microban) — ramped 0 → -1.0 by curriculum
  - feet_distance: NOT added yet (pinned — see commented block below)
  - action_rate: microban (-0.1), NO curriculum
  - removed microduck-only effort terms: neck_action_rate_l2, joint_torques_l2
    (the shared action_rate_l2 still smooths the neck — it sums over all action dims)
  - head_pose_tracking: kept ON (weight 1.0) so the neck keeps a position target
  - body_pose tracking: kept (infra intact) but DISABLED (weight 0)

See the 2026-06-28 microduck-vs-microban reward comparison for rationale.
"""

from dataclasses import replace
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)

NUM_STEPS_PER_ENV = 24

# Iteration at which the no_stepping penalty ramps to its final weight. microban
# defers this to ~iter 3000; microduck runs converge earlier (~2250) so we turn
# it on after basic walking is established. Tunable.
NO_STEPPING_KICKIN_ITER = 1000


def make_microduck_velocity2_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    r = cfg.rewards

    # ── Tracking + posture/orientation: match microban exactly ───────────────
    r["track_linear_velocity"].weight = 2.0
    r["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    r["track_angular_velocity"].weight = 3.0
    r["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    r["pose"].weight = 1.0
    r["upright"].weight = 1.0
    r["upright"].params["std"] = math.sqrt(0.1)
    # Drop microduck-only posture extras.
    r.pop("com_height_target", None)
    r.pop("stillness_at_zero_command", None)

    # ── Gait / feet: match microban exactly ──────────────────────────────────
    r["air_time"].weight = 3.0
    r["air_time"].params["command_threshold"] = 0.01
    r["air_time"].params["threshold_min"] = 0.125
    r["air_time"].params["threshold_max"] = 0.300
    r["foot_clearance"].params["target_height"] = 0.02
    r["foot_swing_height"].params["target_height"] = 0.02
    r.pop("soft_landing", None)  # microban deletes it

    # ── foot_slip: match microban ────────────────────────────────────────────
    r["foot_slip"].weight = -1.0

    # ── no_stepping (microban): penalize airborne feet at ~zero command ──────
    r["no_stepping"] = RewardTermCfg(
        func=microduck_mdp.no_stepping_penalty,
        weight=0.0,  # ramped to -1.0 by the curriculum below
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )

    # ── feet_distance: NOT added for now — PINNED for later. To enable, uncomment:
    # r["feet_distance"] = RewardTermCfg(
    #     func=microduck_mdp.feet_distance_penalty,
    #     weight=-1000.0,
    #     params={
    #         "min_dist": 0.08,  # microban; foot sites ~0.072 m apart standing straight
    #         "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    #     },
    # )

    # ── action_rate: microban value, NO curriculum ───────────────────────────
    r["action_rate_l2"].weight = -0.1
    # Drop microduck-only effort terms.
    r.pop("neck_action_rate_l2", None)
    r.pop("joint_torques_l2", None)

    # ── head/body pose tracking ──────────────────────────────────────────────
    # head_pose_tracking kept ON (weight 1.0): gives the 4 neck/head DOFs a
    # position objective. Combined with the shared action_rate_l2 (which sums
    # over ALL action dims, neck included), the neck is fully shaped even though
    # the neck-only neck_action_rate_l2 term was removed. body_pose tracking
    # stays disabled for now.
    if "head_pose_tracking" in r:
        r["head_pose_tracking"].weight = 2.0
    if "body_pose_tracking" in r:
        r["body_pose_tracking"].weight = 0.0

    # ── Curricula ────────────────────────────────────────────────────────────
    # Remove the action_rate weight ramp (we use a fixed -0.1 like microban).
    cfg.curriculum.pop("action_rate_weight", None)
    # Ramp no_stepping 0 → -1.0 once basic walking is established.
    cfg.curriculum["no_stepping_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "no_stepping",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": NO_STEPPING_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    return cfg


# Reuse the velocity runner config, but log under a separate experiment name so
# velocity2 runs don't collide with velocity runs.
MicroduckVelocity2RlCfg = replace(
    MicroduckRlCfg,
    experiment_name="velocity2",
    run_name="velocity2",
)
