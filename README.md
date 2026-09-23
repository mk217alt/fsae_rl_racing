# FSAE RL Racing

## Overview

The objective of this project is to develop and compare reinforcement-learning approaches for autonomous racing of a simplified Formula Student-style vehicle, simulated end-to-end in NVIDIA Isaac Sim. A custom four-wheeled car (front-wheel drive, articulated front steering, top speed 6 m/s) is trained with PPO (Stable-Baselines3) to lap closed circuits. Starting from a single-car centerline-following baseline, the project branches into several lineages: a behavior-cloning-assisted variant, a classical (non-learned) pure-pursuit baseline, a curvature-aware parallel-training variant, a racing-line reward variant, and a two-car self-play formulation in which one shared policy drives both cars — trained for about 95 million steps on the original track and on a second, unseen "stadium" track.

Every lineage uses the same car model, action interface, and track construction. Because the lineages were trained on different reward functions, their training rewards are **not** comparable; every headline result was therefore re-measured with direct physical metrics (lap time, driving line, smoothness, crashes, overtakes). Several conclusions drawn earlier from training rewards did not survive this re-evaluation — see [Key findings](#key-findings) and §8.

This repository is developed as part of a thesis project. The full development record, including every mistake and correction, is in `rl_journal.html`.

## Key findings

Measured directly, not from training rewards:

- **Pure pursuit beats the learned single-car baseline** on the same track and reference line: flying laps of 51.4 s vs. 73.9 s (450k policy) and 60.6 s (700k smoothness policy), with no off-track runs and about one seventh of the action jitter of the smoother RL policy. The two were similar only in training reward.
- **Two record training rewards were simulator artifacts.** The curvature-lookahead (121.9) and racing-line (187.6) lineages trained on a stage whose ground was a finite box collider, on which the chassis collapses. A collapsed, stationary car holding full throttle still collects the throttle bonus. On the corrected ground, the racing-line policy is the fastest single-car controller (21.3 s laps, 97% of top speed), while the curvature-lookahead policy leaves the track within 3 s — it never learned to drive.
- **The behavior-cloning result is a data problem, not a verdict on BC.** All three demonstration datasets were recorded driving the track in the opposite direction, relative to a shifted reference line, with bang-bang keyboard input.
- **Zero-shot transfer was better than the reward suggested.** On the unseen stadium track, the original-track self-play policy lapped only 8% slower than after adaptation (23.9 s vs. 22.05 s); its reward drop (+141.7 → −155.5) is mostly the centering penalty for a wider line. Three rounds (3M steps) of adaptation closed the gap.
- **The self-play cars never learned to race.** Across 96 evaluation heats on the stadium track, the lead changed 11 times, 9 of them after an incident of the leader; the car that started on the inside led at the end of 85 heats. An explicit overtaking incentive made the follower follow closer, not pass.
- **Opponent awareness gave no advantage.** Racing the self-play policy against the (opponent-blind) racing-line policy on the corrected stage: a dead heat over both lane assignments, and 0.5% more distance for the racing-line policy in clean heats. An earlier "19% more distance" result had run on the defective ground and is withdrawn.

## Pipeline Overview

The project consists of the following stages:

1. [Simulation environment setup](#1-simulation-environment-setup)
2. [Single-car PPO baseline](#2-single-car-ppo-baseline)
3. [Behavior-cloning-assisted fine-tuning (negative result)](#3-behavior-cloning-assisted-fine-tuning-negative-result)
4. [Classical baseline: pure pursuit](#4-classical-baseline-pure-pursuit)
5. [Vectorized + curvature-lookahead single-car variant](#5-vectorized--curvature-lookahead-single-car-variant)
6. [Racing-line reward variant](#6-racing-line-reward-variant)
7. [Two-car self-play racing](#7-two-car-self-play-racing)
8. [Comparative evaluation](#8-comparative-evaluation)
9. [Verification & testing](#9-verification--testing)

---

## 1. Simulation Environment Setup

The base car model and track used by the early lineages were built directly in Isaac Sim via USD authoring.

Implemented in:

`Simulation/Simulation_Environment_Setup/create_simple_car.py`, `create_track.py`, `add_car_body.py`, `add_second_car.py`, `remove_car_body.py`, `set_front_wheel_drive.py`, `drive_car.py`

### Workflow
```
Car chassis + wheel articulation
        ↓
Front-wheel drive + steering joints
        ↓
Track centerline generation (straights + corner arcs)
        ↓
Road surface + collidable curb walls
        ↓
simple_car.usd (car + track combined stage)
```

### Processing Steps
- Build the chassis as an Xform link with a child Cube collision shape (not a scaled Cube-as-rigid-body).
- Attach four wheels; front wheels get a knuckle + revolute steering joint and a drive joint (front-wheel drive only — rear wheels spin freely, no DriveAPI).
- Generate a closed-loop track centerline (four straights and four quarter-circle corners, each arc approximated by 10 chords), 9 m wide, with collidable curb-wall boxes on both sides.
- Duplicate the car into a second lane for two-car scenarios (`add_second_car.py`), remapping all internal joint/material relationship targets.

### Output
`simple_car.usd` — the shared base stage used by the single-car baseline and BC/pure-pursuit lineages. Its ground is Isaac Sim's analytic ground plane.

### Notes
An early version gave the car body a decorative body-kit mesh as a scaled-Cube rigid body, which caused a duplicate-rendering Fabric/USD bug — the body kit was removed entirely rather than debugged further. Track construction also surfaced a subtle USD bug: **transform ops compose in the reverse of the order they're added** (the last-added op acts on the geometry first), so adding Scale→Rotate→Translate actually executes as Translate→Rotate→Scale and corrupts positions for anything with rotation. Fixed by always adding ops in Translate→Rotate→Scale order, and verified by reading back world transforms before trusting new track geometry.

**Ground collider:** later stages (§5–§7) were first authored with a large box as the ground. Box-vs-wheel contact turned out to be unreliable in PhysX — the wheels sink into the box and the chassis rests on it at 0.2 m (the "chassis collapse", §7). Every stage should use the analytic plane (`PhysicsSchemaTools.addGroundPlane`), as the two-car stages now do.

---

## 2. Single-Car PPO Baseline

Pure reinforcement learning, no imitation data, trained to follow a reference line on the original track.

Implemented in:

`Baselines/Single_Car_Baseline/car_track_env.py`, `train_car2.py`, `run_dual_car.py`, `compare_smoothness.py`

### Workflow
```
simple_car.usd
        ↓
Gymnasium env (5-dim obs: lateral_offset, sin/cos heading_error,
                forward_speed, yaw_rate)
        ↓
PPO (Stable-Baselines3, CPU device)
        ↓
car2_ppo_model.zip
```

### Processing Steps
Trained across 7 incremental rounds, progressively raising the centering/off-track reward weights as consistency improved, then extended with a smoothness term:

| Round | Cumulative steps | Weights (lateral / off-track / throttle / smoothness) | Result |
|---|---|---|---|
| 1 | 50k | 0.15 / −10 / — / — | about half of the episodes ended off-track |
| 2–3 | 150k | 0.25 / −20 / 0.05 → 0.40 / −20 / 0.15 | converged, last 25+ episodes full-length |
| 5 | 250k | 0.60 / −30 / 0.25 / — | nearly all episodes full-length, reward 89–104 |
| 6–7 | 450k | (unchanged) | reward 105–113, ~99% full-length — the "450k policy" |
| smoothness | 700k | + smoothness 0.08 | action jitter −18.6%, still full-length — the "700k policy" |

(The logs show rounds 2–3 as a single process from 50k to 150k steps; the split into two weight sets follows the project journal.)

### Output
`car2_ppo_model.zip` — the **700k smoothness policy**. The 450k policy is kept locally in `checkpoint_backups/` (gitignored). `run_dual_car.py` deploys it live: car 1 keyboard-driven, car 2 policy-driven, with an auto-recovery net that teleports the car back onto the reference line if it drifts off-track.

### Notes
- The reference line of this stage is the road centerline shifted by +3 m in y (the stage was derived from a two-lane scene), and the car drives the original vertex order, which is **clockwise seen from above**.
- Measured on physical metrics (§8), the 450k policy laps in 73.9 s and the 700k policy in 60.6 s — the smoothness term also made the car 18% faster, presumably by wasting less wheel speed on steering corrections. Both left the track in some long runs (7/20 and 2/20), always after more than 800 control steps, i.e. far beyond the 500-step (33 s) training episode.
- The training script resumes from an existing checkpoint but only saves at the end of each invocation — killing a run loses that run's progress. Checkpoint versioning was added later in the two-car lineage (§7).

---

## 3. Behavior-Cloning-Assisted Fine-Tuning (Negative Result)

An attempt to warm-start the single-car policy from human driving demonstrations before PPO fine-tuning — kept as a documented negative result.

Implemented in:

`Baselines/BC_Finetuning_Experiment/record_driving.py`, `pretrain_bc.py`, `warmup_critic.py`, `finetune_bc.py` (`_v2` through `_v5`)

### Workflow
```
Human keyboard driving demonstrations
        ↓
Behavior cloning (fit action mean to demonstrations)
        ↓
Critic warm-start (regress on Monte-Carlo returns
                    from real rollout episodes)
        ↓
PPO fine-tuning from the BC-pretrained policy
```

### Processing Steps
Run across three rounds, each with a different demonstration dataset:

1. **Round 1** (2,608 samples, natural driving style): fit BC action mean, then fixed two real problems — `log_std` left at SB3's default (std = 1.0, far too wide for a [−1, 1] action range; tightened to 0.15), and an untrained critic (warm-started on the returns of 40 rollout episodes). Even with both fixes and a reduced learning rate, fine-tuning plateaued at a best episode reward of −36.
2. **Round 2** (1,841 samples, deliberately centered driving, all fixes applied from the start): same ceiling, best reward −37.
3. **Round 3** (1,422 samples, deliberately varied speed/line): same ceiling again, best reward −28.2.

### Output
All intermediate checkpoints and datasets were kept for citing specific numbers: `car2_bc_pretrained{,_v1,_v2}.zip`, `car2_bc_warmed{,_v1,_v2}.zip`, `car2_bc_finetuned{,_v2,_v3,_v4,_v5}.zip`, `human_driving_data{,_v1,_v2}.npz`, and their monitor CSVs.

### Notes
The consistent failure was first attributed to the technique itself. **An analysis of the recorded data explains it differently:** in all three datasets, every sample has a heading error of more than 90° (mean cos −0.91 to −0.98) — car 1 faces +x in the scene, which on its start straight is against the RL task's direction of travel, so the track was driven the other way round. The observations were also computed relative to car 2's +3 m shifted reference line (median offset 2.3–3.0 m; 31% of dataset 1 beyond the 4.2 m off-track margin), and the keyboard produces only full-or-nothing commands. The three datasets therefore shared one systematic mismatch with the task. The experiment shows the effect of distribution shift between demonstrations and task; whether BC helps with matched demonstrations (same direction, same reference line, continuous input) remains open.

---

## 4. Classical Baseline: Pure Pursuit

A non-learned control baseline, to measure how much a learned policy actually gains over a hand-set geometric controller on this task.

Implemented in:

`Baselines/Classical_Baseline/pure_pursuit_baseline.py`

### Workflow
```
Reference line (same +3 m shifted line as the baseline)
        ↓
Bicycle-model pure-pursuit steering (4 m look-ahead)
        ↓
Curvature-aware speed target (≤ 5 m/s)
        ↓
Evaluation on the same env/track as the PPO baseline
```

### Output
With the baseline's training reward: 40/40 full-length episodes, average reward 108.70 (range 98.47–114.97). On physical metrics (§8): 20/20 runs completed three laps, flying laps of 51.43 ± 0.09 s, no off-track runs, mean action delta 0.117.

### Notes
Similar in training reward to the RL baseline (105–113), but **clearly better on physical metrics** — faster than both RL baseline policies (73.9 s / 60.6 s), about seven times smoother, and robust over long runs. Its speed law is conservative (target ≤ 5 m/s, slowing in corners); a pure-pursuit controller at the car's limit was not tested and would very likely be faster still.

---

## 5. Vectorized + Curvature-Lookahead Single-Car Variant

An experimental lineage combining two changes at once: parallel training for speed, and a longer-horizon observation.

Implemented in:

`Racing/Vec_Curvature_Lineage/build_vec_track_env.py`, `build_vec_deploy_stage.py`, `car_track_vec_env.py`, `train_car2_vec.py`, `run_dual_car_vec.py`

### Workflow
```
N car+track clones in one USD stage (vec_train_car2.usd)
        ↓
Single batched VecEnv (one world.step() advances all clones)
        ↓
7-dim observation (5-dim baseline + sin/cos of curvature 5m ahead)
        ↓
PPO
        ↓
car2_vec_ppo_model.zip
```

### Output
The parallel architecture works: ~2.3× training throughput vs. the single-env baseline (186 vs. 81 timesteps/sec with 10 clones), and the same design later carried the two-car lineage. The **policy** does not: at 472k steps its training reward peaked at 121.9 with 50–60% full-length episodes, but on the corrected ground it leaves the track in all 20 evaluation runs within 43–53 control steps (~3 s).

### Notes
- **The reward was a simulator artifact.** `vec_train_car2.usd` still has the box ground collider, and replayed training episodes show the chassis collapsed in 8 of 10 clones. A stationary car holding full throttle collects the throttle bonus (0.25 per step, 125 per episode); clones that moved 0.02–0.21 m in a whole episode scored 112.7–121.1. The lineage never learned to drive. This stage was never corrected.
- Reversing the track direction after training broke the policy (5/5 trials failed within ~50–60 steps) because it flips every corner's curvature. The reversal was left in `car_track_vec_env.py`; the evaluation scripts restore this lineage's training direction (original vertex order = **clockwise seen from above**). The racing-line and two-car lineages were trained fresh on the reversed order, i.e. counterclockwise seen from above, whatever the code comments call it.

---

## 6. Racing-Line Variant

A reward-shaping experiment: loosen the centerline-adherence penalty and let the physical curbs, rather than a tight reward term, bound how the car uses the track.

Implemented in:

`Racing/Racing_Line_Variant/car_track_racingline_vec_env.py`, `train_car2_racingline.py`

### Workflow
```
Same env/observation as the vec-curvature variant
        ↓
Lateral penalty loosened 0.6 → 0.1
        ↓
Throttle bonus raised 0.25 → 0.4
        ↓
PPO (321k steps)
        ↓
car2_racingline_ppo_model.zip
```

### Output
Training statistics by quarter of the final round:

| Quartile | Avg reward | Full-length rate |
|---|---|---|
| Q1 | 73.6 | 50% |
| Q2 | 133.2 | 92% |
| Q3 | 178.9 | 96% |
| Q4 | 187.6 | 96% |

On the corrected ground (§8): 20/20 runs completed three laps, flying laps of **21.27 ± 0.03 s** (5.82 m/s, 97% of top speed), never off-track, action delta 0.073 — the fastest single-car controller of the project.

### Notes
- **The 187.6 reward was not earned by driving.** This lineage trained on the same box-ground stage as §5. Replayed on it, the clones spend 68–79% of all steps collapsed and still score 190–206 per episode, because a stationary car at full throttle collects 0.4 × 500 = 200. The reward is therefore not evidence that the racing line is better than the baseline.
- The network nevertheless drives fast on correct physics, because it learned to steer from the part of each episode before the chassis collapsed (23–25 m on average). Despite its name, it does not use the track width: its mean offset from the centerline is only 0.16 m. The corners of the original track can be taken at full speed, so full throttle on the centerline is close to optimal there.

---

## 7. Two-Car Self-Play Racing

Two cars share one track and are driven by a single shared policy network (self-play), each observing the other's relative state. Trained for 104 rounds and 95.2 million steps: 49 rounds on the original track, then 55 rounds on a second, unseen "stadium" track.

Implemented in:

`Racing/Two_Car_Self_Play_Racing/build_race_track_env.py`, `build_race_deploy_stage.py`, `car_race_vec_env.py`, `train_race_vec.py`, `run_race_demo.py`, `run_race_solo_demo.py`, `run_race_playable_demo.py`, `diag_race_solver_fix.py`, `run_accel_demo.py`

Stadium track: `build_unknown_track_deploy.py`, `build_unknown_track_train.py`, `car_unknown_track_env.py`, `train_unknown_track.py`, `run_unknown_track_demo.py`, `test_baseline_track_eval.py`, `test_unknown_track_generalization.py`, `lap_time_benchmark.py`, `compare_two_car_smoothness.py`

### Workflow
```
8 racing-pair clones (2 cars sharing 1 track each, 16 cars)
        ↓
Single SHARED PPO policy controls every car slot
        ↓
10-dim observation (7-dim curvature-aware + opponent-relative:
                     progress gap, lateral gap, relative speed)
        ↓
Reward: progress − centering + throttle bonus − off-track,
        − proximity penalty (< 1.2 m), later + overtaking and − smoothness terms
        ↓
car_race_ppo_model.zip
```

### Processing Steps
| Phase | Rounds | Steps | Change | Outcome |
|---|---|---|---|---|
| A | original 1–14 | 0–5.5M | proximity penalty 5.0/1.8 m → 0.4/1.2 m; centering, off-track, proximity weights raised | full-length 42–49%, later 61–81%; box ground |
| B | original 15–18 | 5.5–9.0M | ground fixed + centering raised in one round → regression; centering restored in round 18 | 2–8%, recovery begins |
| C | original 19–22 | 9.0–13.1M | proximity penalty 0.8 → 3.0, then 1.5 | up to 53%, decline to 24%, recovery |
| D | original 23–49 | 13.1–40.1M | none | from round 29 mostly 70–93% |
| E | stadium 1–12 | 40.1–52.1M | new track, same reward | 80–100%; laps 7–16% slower by round 12 |
| F | overtaking 1–2 | 52.1–54.1M | throttle bonus 0.15 → 0.30; overtaking incentive 0.3 | 94–99% |
| G | smoothness 1–35 | 54.1–89.2M | smoothness term 0.08 | 90–100% |
| H | competition 1–6 | 89.2–95.2M | overtaking incentive 0.3 → 0.5 | 92–100% |

Key episodes along the way:

1. **Collision (proximity) penalty:** 5.0 per step within 1.8 m (normal overtaking distance on a 9 m track) was 15–50× larger than every other per-step term; the full-length rate fell 79% → 44% within round 2. Fixed to 0.4 within 1.2 m from round 3. A later value of 3.0 reproduced the same pathology more slowly and was backed off to 1.5.
2. **Chassis-collapse bug:** under throttle, a car's chassis settled at exactly 0.2 m, half its box height. Five car-side fixes and a ground physics material changed nothing; the cause was the ground collider's **shape** — a finite box instead of an analytic plane. Switching to `PhysicsSchemaTools.addGroundPlane` removed it (stress-test height error 0.000 m over 400 steps; 2.35% residual recovery rate with the trained policy).
3. **Regression after the fix:** correcting the training ground in the same round as a centering-weight change crashed the full-length rate to 4–6%. Reverting only the reward change started the recovery — hence the rule "one change per round" and the checkpoint versioning (`checkpoint_backups/`, from round 29).
4. **Stadium track:** 40 m / 8 m straights and 6 m corners (vs. 16 m straights and 10 m corners), same width. Zero-shot, the run-49 policy reached the time limit as often as on its own track (93.3%), but its reward fell from +141.7 to −155.5 (see §8 for what that meant physically). Adaptation training continued the same network on an 8-pair stadium stage.
5. **Reward redesign:** a lap-time benchmark revealed a slowdown by round 12 (22.1 s → 25.7 s flying laps in the benchmark) and demos showed one car following the other; throttle bonus and an overtaking incentive were added, then a smoothness term after the driving looked jerky, then a larger overtaking incentive.
6. **Spawn-position bias:** the stadium deploy stage spawns both cars at fixed positions; the car in position B (the inside of the first corner) led at every lap line in 6 of 8 benchmarks. `run_unknown_track_demo.py` and `lap_time_benchmark.py` now swap the starting positions at random.

### Output
`car_race_ppo_model.zip` — the final checkpoint, competition round 6 (95,172,864 steps). The run-49 original-track checkpoint and every later round are kept locally in `checkpoint_backups/` (gitignored).

### Notes
Measured with the race evaluation (§8), the final policy laps the stadium track in 22.63 s and the original track in 21.43 s without incident — but it does not race: the car starting on the inside leads after the first corner and keeps the lead. Compared with stadium round 3 (22.05 s, action delta 0.042), the final policy is slightly slower and about three times less smooth (0.120): the redesign reversed the round-12 slowdown but did not improve on round 3. The jerky driving had already built up during phase E, before the overtaking terms existed, and the smoothness term's effect faded after smoothness round 20.

---

## 8. Comparative Evaluation

Direct physical measurements of every lineage, written to check headline results that rested only on training rewards or demos. Write-ups: `COMMON_METRIC_EVALUATION.md`, `HEAD_TO_HEAD_EVALUATION.md`.

Implemented in:

`Evaluation/Comparative_Evaluation/common_metric_eval.py`, `collapse_reward_diagnostic.py`, `head_to_head_eval.py`, `two_car_race_eval.py` (earlier, superseded: `race_mixed_policy_eval.py`, `run_mixed_race_demo.py`)

### Workflow
```
common_metric_eval.py       single-car controllers: 20 seeded random starts, up to 3 laps /
                            2,500 control steps, lap time and speed along the road centerline,
                            offset, off-track and collapse events, action delta
                            (box-ground policies also re-run on the analytic plane; TRACE=1 records paths)
collapse_reward_diagnostic.py
                            replays 500-step training episodes with the training reward and
                            records how much of it was collected with a collapsed chassis
head_to_head_eval.py        racing-line policy vs. self-play policy (run 49) on the corrected stage:
                            5,000-step heats with lanes swapped, with/without the safety net,
                            plus 20 random-start heats of 1,500 steps
two_car_race_eval.py        one self-play checkpoint drives both cars: 12 heats of 2,000 steps from
                            random side-by-side starts, no safety net; lap times, offsets, incidents,
                            gap between the cars, lead changes (TRACE=1 records paths)
```

### Output
**Single-car controllers, original track** (`common_metric_eval.py`):

| Controller | Ground | 3 laps done | Off-track runs | Runs with collapse | Flying lap [s] | Speed [m/s] | Action delta |
|---|---|---|---|---|---|---|---|
| RL baseline, 450k | plane | 0/20 | 7 | 0 | 73.87 ± 0.70 | 1.37 | 1.030 |
| RL baseline + smoothness, 700k | plane | 0/20 | 2 | 0 | 60.62 ± 0.95 | 1.87 | 0.847 |
| Pure pursuit | plane | 20/20 | 0 | 0 | 51.43 ± 0.09 | 2.46 | 0.117 |
| Curvature lookahead | box (as trained) | 0/20 | 3 | 17 | – | 0.20 | 0.014 |
| Racing-line | box (as trained) | 0/20 | 0 | 20 | – | 0.13 | 0.003 |
| Curvature lookahead | analytic plane | 0/20 | 20 | 0 | – | 1.47 | 0.065 |
| Racing-line | analytic plane | 20/20 | 0 | 0 | 21.27 ± 0.03 | 5.82 | 0.073 |

**Head-to-head, racing-line vs. self-play run 49** (`head_to_head_eval.py`, corrected stage): summed over both lane assignments 3,644 m vs. 3,655 m with the safety net and 3,275 m vs. 3,273 m without — a dead heat; both lap at ~21 s (top speed). In 20 random-start heats the racing-line car finished ahead in 17; in the 16 clean heats by 0.5% of the distance. Each fixed heat was decided by what happened after the start (in one, the cars ran side by side for 4,941 of 5,000 steps), not by pace.

**Self-play race evaluation** (`two_car_race_eval.py`; lap times, offsets, action delta and gap over heats without incident; positive corner offset = inside):

| Checkpoint | Track | Flying lap [s] | Offset [m] | Corner offset [m] | Action delta | Heats with incident | Lead changes | Mean gap [m] |
|---|---|---|---|---|---|---|---|---|
| run 49, zero-shot | stadium | 23.87 ± 0.12 | 0.89 | −0.79 | 0.067 | 2 | 3 | 11.4 |
| stadium round 3 | stadium | 22.05 ± 0.05 | 0.16 | +0.28 | 0.042 | 0 | 0 | 7.1 |
| stadium round 12 | stadium | 23.52 ± 0.40 | 0.17 | +0.09 | 0.141 | 2 | 0 | 10.5 |
| overtaking round 2 | stadium | 23.21 ± 0.37 | 0.18 | +0.14 | 0.124 | 4 | 2 | 3.6 |
| smoothness round 1 | stadium | 22.67 ± 0.38 | 0.18 | +0.16 | 0.087 | 1 | 2 | 4.3 |
| smoothness round 20 | stadium | 23.24 ± 1.31 | 0.15 | +0.02 | 0.064 | 2 | 0 | 6.8 |
| smoothness round 35 | stadium | 22.58 ± 0.36 | 0.25 | −0.03 | 0.103 | 5 | 4 | 17.0 |
| competition round 6 (final) | stadium | 22.63 ± 0.12 | 0.16 | +0.08 | 0.120 | 0 | 0 | 10.3 |
| run 49 | original | 21.03 ± 0.08 | 0.07 | +0.01 | 0.203 | 6 | 0 | 6.5 |
| competition round 6 (final) | original | 21.43 ± 0.08 | 0.17 | +0.01 | 0.220 | 0 | 0 | 12.7 |

### Notes
- **Training rewards measured the wrong thing in several places**: collapsed cars holding throttle (§5, §6), a slowdown hidden by a saturated full-length rate (phase E), and a zero-shot reward drop caused mainly by the centering penalty (run 49 drives the stadium track 0.89 m off the centerline, on its own track 0.07 m; with a weight of 0.9 that difference alone is worth ~350 reward per episode).
- **Why the self-play cars do not race:** both cars are identical, driven by the same deterministic network, and reach the same top speed on every straight, so a follower can only gain through a shorter line the leader already occupies, or a mistake of the leader. At the final weight, gaining one car length is worth 0.5 × 2 = 1.0 — less than one step of the proximity penalty (1.5) — and over an episode the overtaking term telescopes to at most 0.5 × 30 = 15, against 150 for leaving the track. Contested racing would need opponents of different strength (e.g. earlier checkpoints), starts behind a slower car, and a reward for completed passes.
- The first head-to-head (`race_mixed_policy_eval.py`, "racing-line covers 19% more distance") ran on the box-ground stage against an early ~5M-step checkpoint and was dominated by collapse recoveries; it is withdrawn.
- Evaluation code needed the same scrutiny as training code: the curvature-lookahead policy was first evaluated in the wrong direction, the demo's recovery counters mix per-car and per-step counts (the "45% of decisions" recovery rate is 22.6% per car decision), and the fixed start of the lap benchmark decided the race.

---

## 9. Verification & Testing

Utility and diagnostic scripts used during development to validate track geometry, coordinate-frame conventions, and environment correctness, rather than part of the main training pipeline.

Implemented in:

`Evaluation/Verification_and_Testing/test_car_track_env.py`, `test_xform_order.py`, `validate_clockwise.py`, `verify_car.py`, `verify_deploy_fixes.py`, `verify_track.py`, `verify_vec_track_env.py`, `smoke_test_race_env.py`, `smoke_test_vec_env.py`, `diagnose_stage.py`

---

## Limitations

- Simulation only (Isaac Sim 5.1); no sim-to-real transfer was attempted. The car is simplified: no tire model, suspension, or powertrain dynamics, rigid-body contact with friction ≈ 0.5, top speed 6 m/s.
- The policies observe a privileged, noise-free track-relative state; no perception, localization error, latency, or domain randomization.
- One algorithm (PPO, SB3 defaults, 2×64 networks), no hyperparameter search; every lineage was trained once from a single unseeded run, so the variation between runs is unknown and differences are reported descriptively.
- Two tracks from one family (four straights, four circular corners, same width); training episodes last 33 s, and some failures only appear in longer runs.
- The curvature-lookahead and racing-line lineages were never retrained on corrected ground.
- The self-play setup uses two identical cars and one shared network; it did not produce contested racing.
- Monitor CSVs are overwritten by every round, so per-quarter statistics of early rounds survive only in the records written at the time (`rl_journal.html`).

## Requirements

- NVIDIA Isaac Sim 5.1.0 (standalone)
- Stable-Baselines3 2.9, Gymnasium 1.2, NumPy 1.26, PyTorch 2.7

All scripts are run via Isaac Sim's bundled Python interpreter, e.g.:

```
python.bat Racing/Two_Car_Self_Play_Racing/train_race_vec.py 1000000
python.bat Evaluation/Comparative_Evaluation/two_car_race_eval.py stadium 12 2000 newtrack_compete_run6
```

Checkpoint labels in the evaluation scripts (e.g. `run49`, `newtrack_compete_run6`) refer to the versioned backups in `checkpoint_backups/`, which are kept locally and not included in the repository.
