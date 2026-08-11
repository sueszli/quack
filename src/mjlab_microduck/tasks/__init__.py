import os
import sys
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


# ---------------------------------------------------------------------------
# mjlab 1.3.0 migration — velocity-family-first scope.
# Only velocity / velocity2 / velstand / velstand_tiptoe / standup / sit are
# ported and verified under 1.3.0 + canonical BAM. The remaining env cfgs
# (rollers, pose, sitstand, testbench) are NOT yet migrated — re-enable each
# import + registration once it is ported and verified.
# ---------------------------------------------------------------------------
from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_velocity2_env_cfg import (
    make_microduck_velocity2_env_cfg,
    MicroduckVelocity2RlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_velstand_tiptoe_env_cfg import (
    make_microduck_velstand_tiptoe_env_cfg,
    MicroduckVelStandTipToeRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sit_env_cfg import (
    make_microduck_sit_env_cfg,
    MicroduckSitRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_shoot_env_cfg import (
    make_microduck_shoot_env_cfg,
    MicroduckShootRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Velocity task registered: Mjlab-Velocity-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Velocity2 — microban reward/regularization recipe on the velocity task.
register_mjlab_task(
    task_id="Mjlab-Velocity2-Flat-MicroDuck",
    env_cfg=make_microduck_velocity2_env_cfg(),
    play_env_cfg=make_microduck_velocity2_env_cfg(play=True),
    rl_cfg=MicroduckVelocity2RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity2-Rough-MicroDuck",
    env_cfg=make_microduck_velocity2_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity2_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelocity2RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ VelStand task registered: Mjlab-VelStand-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ VelStand task registered: Mjlab-VelStand-Rough-MicroDuck")

# VelStand-TipToe — same as VelStand but with a feet_tiptoe_alignment reward.
register_mjlab_task(
    task_id="Mjlab-VelStandTipToe-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_tiptoe_env_cfg(),
    play_env_cfg=make_microduck_velstand_tiptoe_env_cfg(play=True),
    rl_cfg=MicroduckVelStandTipToeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ VelStand-TipToe task registered: Mjlab-VelStandTipToe-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-VelStandTipToe-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_tiptoe_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_tiptoe_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandTipToeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ VelStand-TipToe task registered: Mjlab-VelStandTipToe-Rough-MicroDuck")

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Rough-MicroDuck")

# Sit task — standing → sitting keyframe, gently (companion to StandUp)
register_mjlab_task(
    task_id="Mjlab-Sit-Flat-MicroDuck",
    env_cfg=make_microduck_sit_env_cfg(),
    play_env_cfg=make_microduck_sit_env_cfg(play=True),
    rl_cfg=MicroduckSitRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Sit task registered: Mjlab-Sit-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-Sit-Rough-MicroDuck",
    env_cfg=make_microduck_sit_env_cfg(rough=True),
    play_env_cfg=make_microduck_sit_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Sit task registered: Mjlab-Sit-Rough-MicroDuck")

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ GroundPick task registered: Mjlab-GroundPick-Flat-MicroDuck")

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ BallKick task registered: Mjlab-BallKick-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ GroundPick task registered: Mjlab-GroundPick-Rough-MicroDuck")

# Shoot task — standing kick with the right leg while the left leg stays planted
register_mjlab_task(
    task_id="Mjlab-Shoot-Flat-MicroDuck",
    env_cfg=make_microduck_shoot_env_cfg(),
    play_env_cfg=make_microduck_shoot_env_cfg(play=True),
    rl_cfg=MicroduckShootRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Shoot task registered: Mjlab-Shoot-Flat-MicroDuck")

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Rollers task registered: Mjlab-Velocity-Flat-MicroDuck-Rollers")

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Swizzle task registered: Mjlab-Velocity-Swizzle-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerCrouch task registered: Mjlab-RollerCrouch-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerSlope task registered: Mjlab-RollerSlope-Flat-MicroDuck")

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerStandUp task registered: Mjlab-RollerStandUp-Flat-MicroDuck")

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Spin task registered: Mjlab-Spin-Flat-MicroDuck")
