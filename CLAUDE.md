# CLAUDE.md

mjlab (MuJoCo Warp) + PPO (rsl_rl) training envs for Microduck, a ~800 g, ~25 cm
biped with 14 Dynamixel XL330 servos. Policies train at 50 Hz, export to ONNX, and
are deployed by the runtime in `pollen-robotics/microduck`. Every convention below
exists because breaking it produced a policy that worked in the viewer and failed on
hardware.

## Commands

```bash
uv run list-envs
uv run train <TASK_ID> --env.scene.num-envs 4096
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # smoke test
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run export <TASK_ID> --wandb-run-path <...>
uv run publish --task <TASK_ID> --wandb-run-path <...> --checkpoint N \
    --repo <user>/microduck-<name> --kind episodic --duration-s 4.0
uv run infer --walking out.onnx   # CPU MuJoCo rehearsal; --no-bam = XML PD
uv run --with pytest pytest tests/
```

Never launch a long run without the 64-env / 5-iteration smoke test first: it catches
~95% of config errors for cents.

## Conventions

- Add every custom MDP function (rewards, events, observations, commands, curricula)
  to `src/task_mdp.py`, grouped by task; one cfg module per task family.
- `task_velocity.py` is the shared base (robot, DR, obs, commands) for other envs.
- Only publishable policies are constant-command episodic/perpetual ones; phase-driven
  and posture-flag policies are driven by the daemon and live in the official set.

## Invariants — do not break these

- **Obs layout is 61D (actor)** and shared across the policy family so the runtime can
  hot-swap policies: 48 proprioception + 13D command block
  `[twist(3), head_pose(4), body_pose(6)]`, in that order. An env that does not use a
  command slot ZERO-PADS it (keep the obs term, sample tiny ranges) — never drop a slot.
- **Joint layout** (14 servos, ctrl idx = joint idx on walk/groundcontact models):
  0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5–8 neck/head (neck_pitch,
  head_pitch, head_yaw, head_roll), 9–13 right leg. On roller/backlash models passive
  joints INTERLEAVE — never hardcode joint indices in mdp functions; use
  `_servo_joint_ids` / `_servo_joint_pos` (identity on plain models, correct elsewhere).
- **Unactuated joints are named `passive_*`** (wheels, backlash hinges) and every
  actuator/obs/reward selector uses `^(?!passive_).*`. New `passive_` regexes must not
  accidentally match backlash joints: `^passive_.*wheel`, not `^passive_.*`.
- **Actuators are BAM** (voltage-controlled XL330, friction computed by the actuator).
  A STANDALONE env cfg must register the `expand_bam_friction_fields` startup event, and
  joint-friction DR must scale the actuator's `friction_scale` — `dof_frictionloss` is
  zeroed under BAM, so randomizing it is a silent no-op.
- **Obs normalization is ON**, so the normalizer must be baked into the ONNX. Use
  `uv run export`; in-sim play applies the normalizer anyway and hides the bug, so never
  hand-convert a checkpoint.
- **Policies are UNFILTERED** (no action low-pass in training). Adding EMA filtering
  needs a matched runtime flag and a transfer test — a trained-with / deployed-without
  mismatch in either direction breaks transfer.
- **DR must not accumulate across resets.** mjlab 1.3.0's `dr.*` ops with
  `operation="add"/"scale"` re-read compile-time defaults and are safe; custom DR
  functions must restore-then-apply.
- If an obs is remapped to a sensor view (backlash encoder, bias), any tracking
  REWARD on that quantity must measure the same view, or the policy is punished for
  correcting what it sees.
- `-Backlash-` variants must mirror their base task's robot model (walk /
  groundcontact / rollers), otherwise backlash A/B comparisons are confounded.

## Building a new env

1. **Build on the closest template**, not from scratch: locomotion → velocity; episodic
   trick ending in a pose → standup; commanded two-state → sitstand; dynamic maneuver →
   roulade. `make_microduck_velocity*_env_cfg` keeps DR / obs / noise / delays in sync;
   standalone from mjlab's base template means porting the whole stack yourself (`_safe`
   critic obs terms, `nan_state` termination with sensor_names,
   `expand_bam_friction_fields`, encoder bias, IMU misalignment).
2. **Verify physics assumptions in sim BEFORE training.** A target/rest pose must be a
   stable equilibrium: hold its ctrl for 3 s from noisy inits and check TILT, not just
   height (a z-only settle test reports fallen states as resting). Measure target
   heights off the actual robot in sim; never carry them across model revisions (a
   5 mm-wrong STAND_Z made the goal unreachable for days).
3. **Cfg conventions**: `ENABLE_*` toggles + tuned constants at the top of the file;
   factory `make_..._env_cfg(play: bool, rough: bool)`; register in `task_registry.py`
   (+ `_BACKLASH_TASKS` if applicable); own `RslRl...RunnerCfg` with a distinct
   `experiment_name`. Symmetry mirror-loss (61D table in `task_symmetry.py`) is OFF by
   default — never enable it for asymmetric tasks.
4. **Write cfg tests** (`tests/test_*_cfg.py`): joint indices resolve on the real model,
   reward weights have the intended sign, gates open/closed where expected.
5. **Smoke test**: builds, steps NaN-free, obs is 61D, every reward term computes, ONNX
   exports.
6. Train and watch the log. Expect a few rounds of reward-hacking whack-a-mole.

## Reward design

- **Sign convention (bit four envs):** mjlab-base cost functions return ≥ 0 → negative
  weight. Self-negating microduck functions (`*_penalty`, `*_l1`, returning ≤ 0) →
  POSITIVE weight. A negative weight on a self-negating penalty double-negates into a
  reward for the violation and the policy farms it. **Check every run: every
  `Episode_Reward/<penalty>` in wandb must be ≤ 0.**
- **RL optimizes the letter of the reward.** Encode what counts as the maneuver in hard
  state-based gates (support contact, orientation-axis checks, latches), not in small
  penalty nudges — every under-specified degree of freedom gets exploited (ballistic
  whip instead of a roll, head-tripod instead of standing).
- **No jackpots:** rate-limit or slew any "reach X" reward. Arriving early at a goal
  state that then pays per-step buys arbitrary violence. For commanded transitions,
  track a slewed internal target (constant-rate blend) so being ahead of the ramp pays
  zero and slow IS the argmax; speed-cap penalties alone integrate to a bounded cost
  and lose.
- **Never gate a positive reward on being in a bad state** (fallen, low) — the policy
  parks in the cheapest qualifying pose. Use potential-based shaping (pay Δprogress,
  e.g. Δcos(tilt): rising pays, holding pays zero). For rest tasks, audit each positive
  term against every stable flop (back / face / side): if flopping keeps most of the
  stack, the policy will flop.
- **Episodic pose-landing tasks:** one fixed target from t=0 (Gaussian + L1 on joints
  and height, generous std) + |a_z| impact penalty + two-layer upright. Not
  keyframe/waypoint trajectories — the policy camps at waypoints, and the path is what
  RL should discover.
- **Two kinds of regularizer.** Motion-blockers (body_ang_vel, angular_momentum, pose
  std) penalize what a dynamic motion physically requires — keep them LOW for dynamic
  tasks. Smoothness (action_rate, joint_torque_rate) damps jitter without blocking slow
  big motions, but introduce it AFTER skill discovery (curriculum from ~0): any
  attempt-tax during exploration of a hard skill makes "do nothing" win. Slow careful
  tasks (reaching) want heavier smoothness than walking.
- **Compare reward mass, not weights,** when copying regularizers between envs: the same
  action_rate weight is 4× weaker under a 4×-larger positive task stack.
- **Tracking Gaussian std** ≈ the error you still care about, not the max error (too
  loose has no gradient at small errors). Before tightening, ask whether the error is
  escapable or inherent to the wanted behavior — a 38%-of-body-mass head MUST oscillate
  while walking, and a tight instantaneous head-tracking std taxed walking so hard the
  policy stood still. Price only the escapable part, e.g. L1 on a 1 s EMA charges DC
  bias and lets oscillation cancel.
- **Multiplicative composites beat additive sums at goal states:** an additive stack has
  a compromise basin (80% of every term via a lean), a product of Gaussians collapses on
  any single deficient factor. Pick stds wide enough that the CURRENT policy scores
  visibly, or the gradient is invisible.
- **Joints parking on hard limits:** add a qpos-side limit-proximity penalty on the
  offending joints. Stock `dof_pos_limits` only fires in the last ~7.5% of range, and
  command-side penalties do not work (the wide ctrlrange is intentional — low-kp servos
  need overshoot).

## Commands, observations, dead weights

- **A command input that is never non-zero has dead weights forever.** Every command
  slot keeps a small non-zero sampling range from step 0, even at reward weight 0.
- **Train zero-command behavior explicitly** (`zero_command_prob`-style exact-zero
  sampling): uniform sampling essentially never produces the all-zero command, which is
  the deployment idle state.
- Rare-but-important command regions need explicit buckets — with independent uniform
  sampling, turn-in-place (`rel_turn_in_place_envs`) was ~2% of experience and never
  trained.

## Curricula

- Steps are env steps: `iteration × 24` (`NUM_STEPS_PER_ENV = 24`).
- Use `microduck_mdp.reward_weight` for weight schedules and a dedicated
  params-curriculum for command/event ranges. `mdp.reward_weight` is a step function,
  not an interpolation — discretize ramps into stages.
- Mutate term cfgs through the managers (`env.event_manager.get_term_cfg(...)`), never
  `env.cfg.events[...]`: managers deepcopy their cfg at init, so writes to `env.cfg` are
  silent no-ops (this also bites eval scripts that force spawn states).
- **Phase-align every stage with what the policy has actually learned**: do not harden
  spawn mixes before the current slice consolidates, or add taxes before the skill
  exists. A wandb metric stepping DOWN exactly at a stage boundary means the pacing is
  wrong — stretch stages or move the introduction later, never earlier.
- Reverse-curriculum spawns (episodes starting partway through the maneuver, including
  nearly-done) fix "learns the start, never the last mile": the frontier otherwise gets
  no on-policy data.

## Training ops & reading a run

- wandb project `mjlab_microduck`; logs in `logs/<experiment_name>/`; resume with
  `--agent.load-checkpoint model_XXXX.pt --agent.resume True`.
- Watch per-iteration: mean reward rising, episode length behaving as the task demands,
  every penalty term ≤ 0, and the MAIN task term actually growing (total reward can rise
  purely on regularizers while the trick never happens). `Episode_Reward/<term>` logs
  the WEIGHTED value, so a term at weight 0 reads 0 regardless of behavior — interpret
  against the weight schedule.
- Budgets: simple episodic tricks ≈ 1000 iters at 4096 envs; gaits and curriculum-heavy
  recovery need 4000–6000.
- **Measure before theorizing.** When a run "fails", run a headless eval of the actual
  checkpoint (per-spawn-type batteries, end-state clusters, angular-rate profiles)
  before changing rewards: past "failures" were an early checkpoint, a success criterion
  splitting one behavior cluster in half, and a pay cap fighting measured physics. Sim
  metrics can pass while the video fails the human eye — watch the video and check which
  geom/axis touches.
- Report what rollouts actually show ("rolls but face-plants 1 in 3"), not "it works!".
  The user decides when it is good enough.

## Sim2real footguns

- A fresh `uv sync` is the ground truth: anything that works only via a manually
  installed local package dies on another machine. Keep `pyproject.toml` honest.
- **Wheels are per-architecture.** On linux-`aarch64` (DGX Spark / GB10) PyPI's torch
  wheel is CPU-ONLY (`2.9.1+cpu`, `torch.version.cuda is None`), so
  `torch.cuda.device_count() == 0` and mjlab's `select_gpus()` indexes an empty list →
  `IndexError` before iteration 0. `[tool.uv.sources]` routes torch to the cu129 index
  on `aarch64` only. Two silent break points, both locked by
  `tests/test_aarch64_cuda_torch.py`: torch must stay a DIRECT dependency (uv applies
  `[tool.uv.sources]` to direct deps only, so deleting the redundant-looking `torch==`
  pin makes the routing a no-op), and the pin must stay `==` (the CUDA index carries
  newer builds than PyPI; a `>=` silently dragged 2.9.1 → 2.13.0).
- A 25 cm robot tumbles at 3.5–5.5 rad/s NATURALLY. Do not impose human-scale speed
  intuitions via caps; put anti-violence pressure on impacts and thrash (|a_z|,
  action_rate, support gates), not on rotation speed.
- IMU DR is zero-centered: it trains tolerance to misalignment magnitude and CANNOT
  compensate a systematic mounting bias (that is a runtime calibration).
- Rehearse hot-swapping in `uv run infer` before touching the robot, with the correct
  command-slot writes — a posture flag lives in the twist vx slot, and feeding all-zeros
  means "stand", which looks like "policy ignores the button".
