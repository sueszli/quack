"""Microduck Velocity2 environment — microban-recipe variant of the velocity task.

Built on top of `make_microduck_velocity_env_cfg` (so DR, observations, command
infra and curricula stay in sync) but with the reward/regularization recipe
swapped to match mjlab_microban's leaner, locomotion-focused setup:

  - tracking + posture/orientation: exactly microban (weights + std)
  - removed microduck-only posture extras: com_height_target, stillness_at_zero_command
  - gait/feet: exactly microban (air_time 3.0 @ 0.125–0.300 s, soft_landing deleted)
  - foot_slip: -0.1 (deliberately 10× weaker than microban's -1.0)
  - no_stepping: REMOVED (not needed — the standing_envs curriculum ramping to 25%
    standing envs teaches stand-without-stepping by ~iter 2000 on its own)
  - feet_distance: NOT added yet (pinned — see commented block below)
  - action_rate: starts at microban's -0.1, ramped -0.1 → -1.0 by iter 1500 (curriculum)
  - removed microduck-only effort terms: neck_action_rate_l2, joint_torques_l2
    (the shared action_rate_l2 still smooths the neck — it sums over all action dims)
  - head_pose_tracking: kept ON (weight 2.0) so the neck keeps a position target
  - body_pose tracking: kept (infra intact) but DISABLED (weight 0)
  - turn-in-place: 15% of envs get lin=0 + |ang| ∈ [0.4, 1.0] (2026-07 audit:
    independent uniform sampling makes spin-on-the-spot ~2% of data → untrained)

See the 2026-06-28 microduck-vs-microban reward comparison for rationale.
"""

from dataclasses import replace
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)

NUM_STEPS_PER_ENV = 24

# Fraction of envs commanded to spin on the spot (lin=0, |ang| ∈ [0.4·max, max]).
# 2026-07 audit: with independent uniform command sampling, turn-in-place is ~2%
# of experience → real-robot spinning was slow/unstable. Explicit practice bucket.
TURN_IN_PLACE_FRACTION = 0.15


def make_microduck_velocity2_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    r = cfg.rewards

    # ── Tracking + posture/orientation: match microban exactly ───────────────
    r["track_linear_velocity"].weight = 2.0
    r["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    r["track_angular_velocity"].weight = 2.0
    r["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    r["pose"].weight = 1.0
    # upright: deliberately stronger than microban (was 1.0 / sqrt(0.1)).
    # 2026-07 pitch-vs-speed eval (claude_experiments/eval_velocity2_pitch_vs_speed.py):
    # the policy walks with a +2-4° steady forward lean (p90 ~6-8°) and ~2/3 of
    # push-induced falls at speed are FORWARD. At weight 1.0 / std²=0.1 a 4° lean
    # cost ~0.05/step — effectively free. At 2.0 / std²=0.05 it costs ~0.19/step:
    # enough gradient to hold the trunk level in steady gait while transient lean
    # (push recovery, accel) stays affordable.
    r["upright"].weight = 2.0
    r["upright"].params["std"] = math.sqrt(0.05)
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

    # ── foot_slip: kept at the base's -0.1 (NOT microban's -1.0 — deliberate:
    # -1.0 was too restrictive for this robot's pivot-heavy turning)
    r["foot_slip"].weight = -0.1

    # ── no_stepping: REMOVED (was added at weight 0.0 with a curriculum that never
    # existed — dead code, 2026-07 audit). Not needed: the standing_envs curriculum
    # (→25%) teaches stand-without-stepping by ~iter 2000 without an explicit penalty.

    # ── feet_distance: NOT added for now — PINNED for later. To enable, uncomment:
    # r["feet_distance"] = RewardTermCfg(
    #     func=microduck_mdp.feet_distance_penalty,
    #     weight=-1000.0,
    #     params={
    #         "min_dist": 0.08,  # microban; foot sites ~0.072 m apart standing straight
    #         "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    #     },
    # )

    # ── action_rate: starts at microban's -0.1; curriculum ramps -0.1 → -1.0 by iter 1500 ──
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
    # Explicit turn-in-place bucket (see TURN_IN_PLACE_FRACTION above).
    twist.rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION
    cfg.curriculum.pop("velocity_command_ranges", None)

    # ── Curricula ────────────────────────────────────────────────────────────
    # action_rate weight ramp: -0.1 (iter 0-500) → -0.2 (500-1000) → -0.3 (1000+).
    # Overwrites the inherited ramp (which went to -1.0).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
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
