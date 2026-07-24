"""Microduck roller SWIZZLE environment — clean classic swizzle.

A separate roller task producing a CLASSIC SWIZZLE: both blades stay on the ground,
the legs spread out and pull back in SYMMETRICALLY (hourglass pattern), propelling
the duck forward. Simpler / more stable alternative to the alternating stride
(`Mjlab-Velocity-Flat-MicroDuck-Rollers`), which does not transfer well to the real
robot. The stride env is left untouched.

Approach A (see docs/design-notes): the base
roller recipe NATURALLY converges to a swizzle, so we reuse the stride env wholesale
(robot, 61D obs, command, full DR, curricula, sim2real — deploys identically with
`--roller`) and only swap the reward recipe:
  - REMOVE the anti-swizzle / stride terms.
  - ADD leg_symmetry (legs mirror) + grounded (both blades down).
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    make_microduck_velocity_rollers_env_cfg,
)

# Stride / anti-swizzle rewards to drop for the swizzle task.
_ANTI_SWIZZLE = ("single_support", "glide", "skating_air_time", "gait_symmetry", "hip_roll_neutral")


def make_microduck_velocity_swizzle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Roller swizzle env: the stride env minus its anti-swizzle terms, plus symmetry
    and grounded rewards. Everything else (robot, obs, command, DR) is identical."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    for name in _ANTI_SWIZZLE:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Legs mirror each other (the swizzle's defining symmetry).
    cfg.rewards["leg_symmetry"] = RewardTermCfg(
        func=microduck_mdp.leg_symmetry_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Keep both blades on the ground (classic swizzle: no lifting).
    cfg.rewards["grounded"] = RewardTermCfg(
        func=microduck_mdp.grounded_reward,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )

    return cfg


# Same PPO hyperparameters as the stride roller task, new experiment/run name.
MicroduckSwizzleRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    experiment_name="velocity_swizzle",
    run_name="velocity_swizzle",
)
