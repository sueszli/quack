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
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)

NUM_STEPS_PER_ENV = 24

# ── Walk-improvement iteration (branch new_pre_alpha_improve_velocity) ─────────
# Action low-pass (EMA on joint targets) — matches microduck_runtime's
# --legs-low-pass / --head-low-pass (filtered = alpha*target + (1-alpha)*prev),
# applied once per control step (50 Hz). alpha: higher = less filtering; 0.6 ≈
# 12 Hz cutoff. CRITICAL: deploy with the runtime filter ON at the SAME alphas.
ENABLE_ACTION_LOW_PASS = True
ACTION_LOW_PASS_LEG_ALPHA = 0.6
ACTION_LOW_PASS_HEAD_ALPHA = 0.6

# Fraction of envs commanded to spin on the spot (lin=0, |ang| forced away from
# zero) — explicit turn-in-place practice. Kept at velocity2's ang ±1.0 range
# (the recipe deliberately caps turn range; the *missing practice* — not range —
# is why turning-on-the-spot was weak).
TURN_IN_PLACE_FRACTION = 0.15

# Iteration at which the no_stepping penalty ramps to its final weight. microban
# defers this to ~iter 3000; microduck runs converge earlier (~2250) so we turn
# it on after basic walking is established. Tunable.
NO_STEPPING_KICKIN_ITER = 10000


def make_microduck_velocity2_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    r = cfg.rewards

    # ── Action low-pass filter (EMA on joint targets) ────────────────────────
    # Swap the plain JointPositionActionCfg for the filtered variant (mirrors
    # microduck_runtime --legs-low-pass / --head-low-pass). MUST deploy with the
    # runtime filter ON at the same alphas.
    if ENABLE_ACTION_LOW_PASS:
        japa = cfg.actions["joint_pos"]
        assert isinstance(japa, JointPositionActionCfg)
        cfg.actions["joint_pos"] = microduck_mdp.FilteredJointPositionActionCfg(
            **vars(japa),
            leg_alpha=ACTION_LOW_PASS_LEG_ALPHA,
            head_alpha=ACTION_LOW_PASS_HEAD_ALPHA,
        )

    # ── Tracking + posture/orientation: match microban exactly ───────────────
    r["track_linear_velocity"].weight = 2.0
    r["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    r["track_angular_velocity"].weight = 2.0
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
    # microban values (proven to bootstrap walking). The earlier floor raise
    # (0.125→0.15, for longer/slower steps) was reverted: combined with the action
    # filter it blocked bootstrapping (air_time reward never rose). Re-add the
    # step-length tuning ONLY after confirming the filter + walking work.
    r["air_time"].params["threshold_min"] = 0.125
    r["air_time"].params["threshold_max"] = 0.300
    r["foot_clearance"].params["target_height"] = 0.02
    r["foot_swing_height"].params["target_height"] = 0.02
    r.pop("soft_landing", None)  # microban deletes it

    # ── foot_slip: match microban ────────────────────────────────────────────
    r["foot_slip"].weight = -0.1 # Was 1.0

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

    # ── action_rate: ramped via curriculum (-0.1 → -0.2 → -0.3, see below) ────
    r["action_rate_l2"].weight = -0.1  # stage-0 value; curriculum ramps it
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

    # ── Command ranges: match microban (the part of the recipe we hadn't ported) ─
    # microduck demanded ang ±1.5→2.0 (well beyond what the robot can turn) plus
    # symmetric lin ±0.2→0.4 widening that outpaced capability and tracked the
    # post-iter-1000 reward/episode-length decline. microban keeps modest, FIXED
    # ranges (it only widens at iter ~3000, beyond a typical run), so set fixed
    # ranges here and drop the widening curriculum below. ang ±0.75 is the big
    # change — it makes turning learnable.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.4, 0.4)
    twist.ranges.lin_vel_y = (-0.3, 0.3)
    twist.ranges.ang_vel_z = (-1.0, 1.0)
    # Turn-in-place practice: force lin=0, |ang| away from zero for a fraction of
    # envs. Keeps microban's ±1.0 turn range (deliberately capped — wider hurt);
    # the fix for weak spin-on-the-spot is missing PRACTICE, not more range.
    twist.rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION
    cfg.curriculum.pop("velocity_command_ranges", None)

    # ── Curricula ────────────────────────────────────────────────────────────
    # action_rate weight ramp: -0.1 (iter 0-500) → -0.2 (500-1000) → -0.3 (1000+).
    # Overwrites the inherited ramp (which went to -1.0).
    # End-weight capped at -0.4 (was -1.0): the EMA low-pass now carries the
    # high-frequency smoothing (incl. the neck, via head_alpha), so a heavy
    # action_rate on top would over-damp / fight the policy.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.4},
            ],
        },
    )

    # Action low-pass alpha curriculum: ramp the EMA IN over training. Start at
    # alpha=1.0 (NO filtering == unfiltered baseline) so walking bootstraps, then
    # ramp down to the deployment alpha (0.6) once the gait is established. A filter
    # from step 0 damped exploration and walking never bootstrapped (air_time flat).
    # Final stage MUST equal the runtime --legs/head-low-pass-alpha (0.6).
    if ENABLE_ACTION_LOW_PASS:
        cfg.curriculum["action_lowpass_alpha"] = CurriculumTermCfg(
            func=microduck_mdp.action_lowpass_alpha_curriculum,
            params={
                "alpha_stages": [
                    {"step": 0,                        "leg_alpha": 1.0,  "head_alpha": 1.0},
                    {"step": 500 * NUM_STEPS_PER_ENV,  "leg_alpha": 0.85, "head_alpha": 0.85},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "leg_alpha": 0.7,  "head_alpha": 0.7},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "leg_alpha": ACTION_LOW_PASS_LEG_ALPHA,
                                                       "head_alpha": ACTION_LOW_PASS_HEAD_ALPHA},
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
