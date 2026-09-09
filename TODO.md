# Embodied research: minimum experiment and curriculum

Status: implementation plan. No training experiments have been run for this plan.

## Start here

**One duck, one existing task, one Qwen researcher, and the existing PPO trainer.**

Qwen proposes experiments; PPO learns the robot's movement. The researcher's
first interface is text: task description, reward code, and measured results.
Use one Qwen endpoint throughout this pilot. Model comparisons, vision, and a
pretrained action model such as pi0.5 can be separate experiments later.

The eventual question is whether an agent can generate training environments
that produce policies capable of solving unfamiliar physical tasks.

## Experiment 0: establish the research loop

Reuse `Mjlab-BallKick-Flat-MicroDuck`: one duck kicks one ball forward while
remaining upright. Its existing motor actor cannot see the ball. This milestone
therefore validates automated experimentation, not puzzle-solving intelligence.

1. **Baseline.** Smoke-test the unchanged configuration, then train a baseline.
2. **Evaluate.** Measure ball displacement along the initial kick direction,
   lateral displacement, peak forward speed, falls, final trunk tilt, and invalid
   states. Save a few rollout videos. Score physical outcomes independently of
   the reward being tuned.
3. **Propose.** Give Qwen the task, reward code, and measurements. Allow changes
   to only two reward weights through validated JSON multipliers.
4. **Repeat.** Train and evaluate the proposal, return results, and request one
   more proposal. Compare with two random proposals under the same budget.

### First implementation ticket

- [ ] Add a headless checkpoint evaluator using the training environment and
  its observation normalizer. Output per-episode metrics and a summary JSON.
- [ ] Calibrate and freeze success thresholds in simulation: forward distance,
  lateral corridor, speed ceiling, and remaining upright through the episode.
  Invalid states count as failures. Check video against the numerical score.
- [ ] Save the resolved configuration, training seed, checkpoint, evaluation
  seeds, metrics, and sample video for each experiment.
- [ ] Wrap the existing trainer and evaluator in one repeatable command.
- [ ] Only then add a Qwen client and the two-weight proposal interface.

Mandatory smoke test before any longer run:

```bash
uv run train Mjlab-BallKick-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

Use a baseline calibration run to choose a training budget that shows learning
with room for improvement. Then freeze that budget for comparisons; do not
accidentally use the recipe's 10,000-iteration default. Smoke-test each candidate
before its longer run.

### Keep the search small

| Existing reward | Current weight | Allowed change |
|---|---:|---|
| `ball_forward_velocity` | `12.0` | Multiply by a finite number in `[0.5, 2.0]` |
| `ball_speed_overshoot` | `-4.0` | Multiply by a finite number in `[0.5, 2.0]` |

Preserve signs. Apply overrides to a fresh configuration before environment
construction. Keep physics, observations, termination, PPO settings, other
rewards, and curricula fixed. These two weights are not the action-rate weight
that the existing curriculum changes.

Use 20 fixed development episodes per candidate. Select using development
results only, then compare the selected Qwen candidate, selected random
candidate, and baseline on 100 separate evaluation seeds. Repeat across three
training seeds before claiming an improvement. Two proposals are a pipeline
pilot, not evidence that an LLM is better than ordinary parameter search.

**Milestone complete:** the propose–train–evaluate–revise loop runs reproducibly,
records failures, and produces inspectable behavior. Qwen does not have to win.

## Curriculum

Add one capability at a time and keep an independent evaluation set at each stage.

| Stage | Experiment | Evidence to advance |
|---|---|---|
| 0. Research loop | Tune the existing kick task | Reproducible automated experiments |
| 1. Physical problem solving | Reach a goal around a barrier using simulator state | Success on unseen barrier positions beyond a direct-to-goal baseline |
| 2. Vision | Solve the same task from camera observations | Closed-loop success on unseen layouts and appearances |
| 3. Environment generation | Researcher edits scene parameters, then reward code | Valid, solvable generated tasks improve performance on independently authored tasks |
| 4. Visual reconstruction | Build a scene from images/video of a known simulation | A policy trained in the reconstruction succeeds in the original simulation |
| 5. Open-ended curriculum | Archive tasks and policies; generate harder tasks | Broader task coverage while retaining earlier capabilities |
| 6. Real robot | Measure and correct simulation errors using real trials | Repeatable real-world performance |

Stage 1 introduces a task policy that issues velocity commands to the walking
controller. Give scene observations to this separate task policy; preserve the
motor controller's shared 61D observation contract. Stage 2 can compare a
pretrained VLM task policy with a learned visual task policy.

Introduce multiple ducks after single-duck task solving works. Start games with
a fixed opponent before self-play. Visual reconstruction needs validation of
dynamics as well as appearance. Simulation-only results cannot establish that
the real-world transfer gap has been closed.

## Repository anchors

- Task: `src/mjlab_microduck/microduck_ball_kick_env_cfg.py`
- Rewards: `src/mjlab_microduck/mdp.py`
- Registry: `src/mjlab_microduck/registry.py`
- Trainer entry point: `src/mjlab_microduck/train_cli.py`
- Export: `scripts/export.py` and `src/mjlab_microduck/export.py`

Preserve BAM physics, observation normalization, reward signs, and the 61D
motor interface. The current kick target constant is `1.0 m/s`; nearby comments
still mention an older `0.25 m/s` target. Resolve settings from executable
configuration. Run evaluation in a directly stepped training environment rather
than the wall-clock body-server loop.
