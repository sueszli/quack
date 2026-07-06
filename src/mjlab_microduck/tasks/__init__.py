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
# Only velocity / velocity2 / velstand / velstand_tiptoe / standup are ported and
# verified under 1.3.0 + canonical BAM. The remaining env cfgs (rollers, pose,
# ground_pick, sit, sitstand, testbench) are NOT yet migrated — re-enable each
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

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ GroundPick task registered: Mjlab-GroundPick-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ GroundPick task registered: Mjlab-GroundPick-Rough-MicroDuck")
