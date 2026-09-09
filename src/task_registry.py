from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

import src.utils  # noqa: F401


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


from .task_backlash import make_backlash_variant
from .task_ball_kick import MicroduckBallKickRlCfg, make_microduck_ball_kick_env_cfg
from .task_ground_pick import MicroduckGroundPickRlCfg, make_microduck_ground_pick_env_cfg
from .task_roller_crouch import MicroduckRollerCrouchRlCfg, make_microduck_roller_crouch_env_cfg
from .task_roller_slope import MicroduckRollerSlopeRlCfg, make_microduck_roller_slope_env_cfg
from .task_roller_standup import MicroduckRollerStandUpRlCfg, make_microduck_roller_standup_env_cfg
from .task_roulade import MicroduckRouladeRlCfg, make_microduck_roulade_env_cfg
from .task_sitstand import MicroduckSitStandRlCfg, make_microduck_sitstand_env_cfg
from .task_spin import MicroduckSpinRlCfg, make_microduck_spin_env_cfg
from .task_standup import MicroduckStandUpRlCfg, make_microduck_standup_env_cfg
from .task_velocity import MicroduckRlCfg, make_microduck_velocity_env_cfg
from .task_velocity_rollers import MicroduckRollersRlCfg, make_microduck_velocity_rollers_env_cfg
from .task_velocity_swizzle import MicroduckSwizzleRlCfg, make_microduck_velocity_swizzle_env_cfg
from .task_velstand import MicroduckVelStandRlCfg, make_microduck_velstand_env_cfg

# Standard velocity task
register_mjlab_task(task_id="Mjlab-Velocity-Flat-MicroDuck", env_cfg=make_microduck_velocity_env_cfg(), play_env_cfg=make_microduck_velocity_env_cfg(play=True), rl_cfg=MicroduckRlCfg, runner_cls=MicroduckOnPolicyRunner)

register_mjlab_task(task_id="Mjlab-Velocity-Rough-MicroDuck", env_cfg=make_microduck_velocity_env_cfg(rough=True), play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True), rl_cfg=MicroduckRlCfg, runner_cls=MicroduckOnPolicyRunner)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(task_id="Mjlab-VelStand-Flat-MicroDuck", env_cfg=make_microduck_velstand_env_cfg(), play_env_cfg=make_microduck_velstand_env_cfg(play=True), rl_cfg=MicroduckVelStandRlCfg, runner_cls=MicroduckOnPolicyRunner)

register_mjlab_task(task_id="Mjlab-VelStand-Rough-MicroDuck", env_cfg=make_microduck_velstand_env_cfg(rough=True), play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True), rl_cfg=MicroduckVelStandRlCfg, runner_cls=MicroduckOnPolicyRunner)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(task_id="Mjlab-StandUp-Flat-MicroDuck", env_cfg=make_microduck_standup_env_cfg(), play_env_cfg=make_microduck_standup_env_cfg(play=True), rl_cfg=MicroduckStandUpRlCfg, runner_cls=MicroduckOnPolicyRunner)

register_mjlab_task(task_id="Mjlab-StandUp-Rough-MicroDuck", env_cfg=make_microduck_standup_env_cfg(rough=True), play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True), rl_cfg=MicroduckStandUpRlCfg, runner_cls=MicroduckOnPolicyRunner)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(task_id="Mjlab-SitStand-Flat-MicroDuck", env_cfg=make_microduck_sitstand_env_cfg(), play_env_cfg=make_microduck_sitstand_env_cfg(play=True), rl_cfg=MicroduckSitStandRlCfg, runner_cls=MicroduckOnPolicyRunner)

register_mjlab_task(task_id="Mjlab-SitStand-Rough-MicroDuck", env_cfg=make_microduck_sitstand_env_cfg(rough=True), play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True), rl_cfg=MicroduckSitStandRlCfg, runner_cls=MicroduckOnPolicyRunner)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(task_id="Mjlab-GroundPick-Flat-MicroDuck", env_cfg=make_microduck_ground_pick_env_cfg(), play_env_cfg=make_microduck_ground_pick_env_cfg(play=True), rl_cfg=MicroduckGroundPickRlCfg, runner_cls=MicroduckOnPolicyRunner)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(task_id="Mjlab-BallKick-Flat-MicroDuck", env_cfg=make_microduck_ball_kick_env_cfg(), play_env_cfg=make_microduck_ball_kick_env_cfg(play=True), rl_cfg=MicroduckBallKickRlCfg, runner_cls=MicroduckOnPolicyRunner)

register_mjlab_task(task_id="Mjlab-GroundPick-Rough-MicroDuck", env_cfg=make_microduck_ground_pick_env_cfg(rough=True), play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True), rl_cfg=MicroduckGroundPickRlCfg, runner_cls=MicroduckOnPolicyRunner)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers", env_cfg=make_microduck_velocity_rollers_env_cfg(), play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True), rl_cfg=MicroduckRollersRlCfg, runner_cls=MicroduckOnPolicyRunner)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(task_id="Mjlab-Velocity-Swizzle-MicroDuck", env_cfg=make_microduck_velocity_swizzle_env_cfg(), play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True), rl_cfg=MicroduckSwizzleRlCfg, runner_cls=MicroduckOnPolicyRunner)

register_mjlab_task(task_id="Mjlab-RollerCrouch-Flat-MicroDuck", env_cfg=make_microduck_roller_crouch_env_cfg(), play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True), rl_cfg=MicroduckRollerCrouchRlCfg, runner_cls=MicroduckOnPolicyRunner)

register_mjlab_task(task_id="Mjlab-RollerSlope-Flat-MicroDuck", env_cfg=make_microduck_roller_slope_env_cfg(), play_env_cfg=make_microduck_roller_slope_env_cfg(play=True), rl_cfg=MicroduckRollerSlopeRlCfg, runner_cls=MicroduckOnPolicyRunner)

# Roller STANDUP — getting up on rollers (dedicated policy, starts on the ground).
register_mjlab_task(task_id="Mjlab-RollerStandUp-Flat-MicroDuck", env_cfg=make_microduck_roller_standup_env_cfg(), play_env_cfg=make_microduck_roller_standup_env_cfg(play=True), rl_cfg=MicroduckRollerStandUpRlCfg, runner_cls=MicroduckOnPolicyRunner)

# Spin task — fast in-place rotation, on rollers (ground-pick slot).
register_mjlab_task(task_id="Mjlab-Spin-Flat-MicroDuck", env_cfg=make_microduck_spin_env_cfg(), play_env_cfg=make_microduck_spin_env_cfg(play=True), rl_cfg=MicroduckSpinRlCfg, runner_cls=MicroduckOnPolicyRunner)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(task_id="Mjlab-Roulade-Flat-MicroDuck", env_cfg=make_microduck_roulade_env_cfg(), play_env_cfg=make_microduck_roulade_env_cfg(play=True), rl_cfg=MicroduckRouladeRlCfg, runner_cls=MicroduckOnPolicyRunner)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see task_backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
import os
from collections.abc import Callable
from typing import Any, NamedTuple

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from .robot import MICRODUCK_BACKLASH_ROBOT_CFG, MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG, MICRODUCK_WALK_BACKLASH_ROBOT_CFG

_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG


class BacklashTask(NamedTuple):
    task_id: str
    make_cfg: Callable[..., ManagerBasedRlEnvCfg]
    make_kwargs: dict[str, Any]
    rl_cfg: Any
    robot_cfg: EntityCfg


# fmt: off
_BACKLASH_TASKS = (
    BacklashTask("Mjlab-Velocity-Flat-Backlash-MicroDuck",            make_microduck_velocity_env_cfg,         {},               MicroduckRlCfg,              _BL_WALK),
    BacklashTask("Mjlab-Velocity-Rough-Backlash-MicroDuck",           make_microduck_velocity_env_cfg,         {"rough": True},  MicroduckRlCfg,              _BL_WALK),
    BacklashTask("Mjlab-VelStand-Flat-Backlash-MicroDuck",            make_microduck_velstand_env_cfg,         {},               MicroduckVelStandRlCfg,      _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-VelStand-Rough-Backlash-MicroDuck",           make_microduck_velstand_env_cfg,         {"rough": True},  MicroduckVelStandRlCfg,      _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-StandUp-Flat-Backlash-MicroDuck",             make_microduck_standup_env_cfg,          {},               MicroduckStandUpRlCfg,       _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-StandUp-Rough-Backlash-MicroDuck",            make_microduck_standup_env_cfg,          {"rough": True},  MicroduckStandUpRlCfg,       _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-SitStand-Flat-Backlash-MicroDuck",            make_microduck_sitstand_env_cfg,         {},               MicroduckSitStandRlCfg,      _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-SitStand-Rough-Backlash-MicroDuck",           make_microduck_sitstand_env_cfg,         {"rough": True},  MicroduckSitStandRlCfg,      _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-GroundPick-Flat-Backlash-MicroDuck",          make_microduck_ground_pick_env_cfg,      {},               MicroduckGroundPickRlCfg,    _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-GroundPick-Rough-Backlash-MicroDuck",         make_microduck_ground_pick_env_cfg,      {"rough": True},  MicroduckGroundPickRlCfg,    _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-BallKick-Flat-Backlash-MicroDuck",            make_microduck_ball_kick_env_cfg,        {},               MicroduckBallKickRlCfg,      _BL_GROUNDCONTACT),
    BacklashTask("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers",    make_microduck_velocity_rollers_env_cfg, {},               MicroduckRollersRlCfg,       _BL_ROLLERS),
    BacklashTask("Mjlab-Velocity-Swizzle-Backlash-MicroDuck",         make_microduck_velocity_swizzle_env_cfg, {},               MicroduckSwizzleRlCfg,       _BL_ROLLERS),
    BacklashTask("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck",        make_microduck_roller_crouch_env_cfg,    {},               MicroduckRollerCrouchRlCfg,  _BL_ROLLERS),
    BacklashTask("Mjlab-RollerSlope-Flat-Backlash-MicroDuck",         make_microduck_roller_slope_env_cfg,     {},               MicroduckRollerSlopeRlCfg,   _BL_ROLLERS),
)
# fmt: on

_DEFAULT_BACKLASH_TASKS = ("Mjlab-Velocity-Flat-Backlash-MicroDuck", "Mjlab-Velocity-Rough-Backlash-MicroDuck", "Mjlab-VelStand-Flat-Backlash-MicroDuck", "Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers")

_BACKLASH_TASKS_BY_ID = {t.task_id: t for t in _BACKLASH_TASKS}
_requested = os.environ.get("MICRODUCK_BACKLASH_TASKS")
if _requested is None:
    _ENABLED_BACKLASH_TASKS = _DEFAULT_BACKLASH_TASKS
elif _requested.strip() == "all":
    _ENABLED_BACKLASH_TASKS = tuple(_BACKLASH_TASKS_BY_ID)
else:
    _ENABLED_BACKLASH_TASKS = tuple(t for t in (s.strip() for s in _requested.split(",")) if t)
    _unknown = [t for t in _ENABLED_BACKLASH_TASKS if t not in _BACKLASH_TASKS_BY_ID]
    if _unknown:
        raise ValueError(f"MICRODUCK_BACKLASH_TASKS names unknown backlash task(s): {_unknown}. Known: {sorted(_BACKLASH_TASKS_BY_ID)}")

for _task_id in _ENABLED_BACKLASH_TASKS:
    _t = _BACKLASH_TASKS_BY_ID[_task_id]
    register_mjlab_task(task_id=_t.task_id, env_cfg=make_backlash_variant(_t.make_cfg(**_t.make_kwargs), _t.robot_cfg), play_env_cfg=make_backlash_variant(_t.make_cfg(play=True, **_t.make_kwargs), _t.robot_cfg), rl_cfg=_t.rl_cfg, runner_cls=MicroduckOnPolicyRunner)
