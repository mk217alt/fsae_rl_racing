# FSAE RL Racing

## Overview

The objective of this project is to develop and compare reinforcement-learning approaches for autonomous racing of a simplified Formula Student-style vehicle, simulated end-to-end in NVIDIA Isaac Sim. A custom four-wheeled car (front-wheel drive, articulated front steering, top speed 6 m/s) is trained with PPO (Stable-Baselines3) to lap closed circuits. Starting from a single-car centerline-following baseline, the project branches into several lineages: a behavior-cloning-assisted variant, a classical (non-learned) pure-pursuit baseline, a curvature-aware parallel-training variant, a racing-line reward variant, and a two-car self-play formulation in which one shared policy drives both cars — trained for 100 million steps on the original track and on a second, unseen "stadium" track.

Every lineage uses the same car model, action interface, and track construction. Because the lineages were trained on different reward functions, their training rewards are **not** comparable; every headline result was therefore re-measured with direct physical metrics (lap time, driving line, smoothness, crashes, overtakes). Several conclusions drawn earlier from training rewards did not survive this re-evaluation — see [Key findings](#key-findings) and §8.

This repository contains the code, models, and evaluation results of the master's thesis *Reinforcement Learning for Autonomous Racing of a Formula Student Vehicle in NVIDIA Isaac Sim* — see [Thesis status](#thesis-status). The full development record, including every mistake and correction, is in `7_journal/rl_journal.html`.

## Key findings

Measured directly, not from training rewards:

- **Pure pursuit beats the learned single-car baseline** on the same track and reference line: flying laps of 51.4 s vs. 73.9 s (450k policy) and 60.6 s (700k smoothness policy), with no off-track runs and about one seventh of the action jitter of the smoother RL policy. The two were similar only in training reward.
- **Two record training rewards were simulator artifacts.** The curvature-lookahead (121.9) and racing-line (187.6) lineages trained on a stage whose ground was a finite box collider, on which the chassis collapses. A collapsed, stationary car holding full throttle still collects the throttle bonus. On the corrected ground, the racing-line policy is the fastest single-car controller (21.3 s laps, 97% of top speed), while the curvature-lookahead policy leaves the track within 3 s — it never learned to drive.
- **The behavior-cloning result is a data problem, not a verdict on BC.** All three demonstration datasets were recorded driving the track in the opposite direction, relative to a shifted reference line, with bang-bang keyboard input.
- **Zero-shot transfer was better than the reward suggested.** On the unseen stadium track, the original-track self-play policy lapped only 8% slower than after adaptation (23.9 s vs. 22.05 s); its reward drop (+141.7 → −155.5) is mostly the centering penalty for a wider line. Three rounds (3M steps) of adaptation closed the gap.
- **The self-play cars never learned to race.** Across 108 evaluation heats on the stadium track, the lead changed 11 times, 9 of them after an incident of the leader; the car that started on the inside led at the end of 96 heats. An explicit overtaking incentive made the follower follow closer, not pass, and training on to exactly 100 million steps (five more rounds, the best training returns of the project) did not change how the cars race. Lap telemetry shows how the follower gives way: the leader drives every lap at full throttle, while the follower lifts before every corner until it is about 3 s behind, and then laps as fast as the leader without attacking.
- **A lock-in decides each race, not a stronger car.** The car that is ahead after the first corner stays ahead and the follower gives way: from a mid-straight start, whichever car started a few tenths of a meter further forward led at the end in 75–83% of heats. A mirrored test (every physical start played once by each car) gave the same start the win in 11 of 12 seeds, so there is no difference between Car A and Car B and no spawn-label effect. An earlier "Car B is the stronger car" reading came from an artifact of the test (shared random offsets) and is withdrawn.
- **Training from staggered starts made things worse.** With 75% of training resets placing the trailing car 1.5–8 m behind the leader (3 rounds, 3M steps), passes from behind fell from 8 of 36 heats to 1 of 36, only 4 of 12 race-evaluation heats were free of incidents (12 of 12 before) and 7 vehicles left the track (none before). Nine of the eleven first incidents happened after step 500, beyond the 500-step training episodes, so the training statistics did not show them. The experiment was discarded and the 100M-step policy restored.
- **Opponent awareness gave no advantage.** Racing the self-play policy against the (opponent-blind) racing-line policy on the corrected stage: a dead heat over both lane assignments, and 0.5% more distance for the racing-line policy in clean heats. An earlier "19% more distance" result had run on the defective ground and is withdrawn.

## Thesis status

The thesis is complete: seven chapters, 114 pages including 121 references and an appendix with the reward and training history of every training round. The manuscript itself is not part of this repository; every number it reports comes from the logs, models, and evaluation results of this project.

| Chapter | Content | Repository |
|---|---|---|
| 1 Introduction | motivation, research questions RQ1–RQ4, contributions | – |
| 2 Background and Related Work | RL and PPO, reward design, imitation learning, racing control, self-play, generalization, simulators | – |
| 3 System Design and Methodology | custom vehicle, tracks, and environments; reward terms; PPO setup; parallel architecture; self-play; evaluation protocol | `1_simulation_scenes/`; environment files in `2_python_scripts/` (`b_single_car/`, `e_curvature_lookahead/`, `f_racing_line/`, `g_two_car_racing/`) |
| 4 Implementation and Development Process | building the assets, the chassis-collapse defect, iterative reward design, evaluation tooling | `2_python_scripts/a_simulation_setup/`, `2_python_scripts/i_verification/` |
| 5 Single-Vehicle Results | baseline, smoothness term, pure pursuit, behavioral cloning, curvature look-ahead, racing line (RQ1, RQ2) | §2–§6, §8 |
| 6 Two-Vehicle Self-Play Results | 109 training rounds, zero-shot transfer and adaptation, reward redesign, race evaluation, the 100-million-step milestone, what decides a race (lock-in), lap telemetry with sector times, staggered-start training, head-to-head (RQ3, RQ4) | §7, §8 |
| 7 Discussion and Conclusion | answers to the research questions, challenges and limitations, future work | – |
| Appendix A | reward configuration and end-of-round statistics of every round of every lineage | `6_training_logs/` (full training logs kept locally), `7_journal/rl_journal.html` |

Answers to the research questions:

- **RQ1** — PPO learns a stable single-car controller from reward shaping alone, but a hand-set pure-pursuit controller is faster and smoother on the same track.
- **RQ2** — Behavioral cloning did not help here; the demonstrations did not match the task, so whether matched demonstrations would help remains open.
- **RQ3** — Two cars under one shared self-play policy drive close to top speed, but neither reward changes nor 100 million steps nor staggered training starts produced contested racing: the car ahead after the first corner keeps the lead.
- **RQ4** — A policy trained on one track drove an unseen track within 8% of the adapted lap time; three million steps of adaptation closed the gap.

## Folder layout

The files are sorted by type. Every script finds its files relative to its own location, so the folder can be placed anywhere.

| Folder | Contents |
|---|---|
| `1_simulation_scenes/` | Isaac Sim scenes (`.usd`): the car and the tracks |
| `2_python_scripts/` | all code (`.py`), one subfolder per part of the project (below), plus `check_results.py` |
| `3_trained_models/` | trained policies (`.zip`); `checkpoints/` holds one per training round |
| `4_results/` | evaluation results (`.json`, `.csv`) and the two evaluation write-ups (`.md`) |
| `5_data/` | recorded human driving (`.npz`) |
| `6_training_logs/` | training monitors, one line per training episode (`.csv`) |
| `7_journal/` | `rl_journal.html`, the full development record |
| `run_output/` | status, log and telemetry files the scripts write while running (not tracked) |
| `run_test.bat` | repeats the thesis measurements in Isaac Sim: double-click it, or run `.\run_test.bat 3` in PowerShell (see [Requirements](#requirements)) |

| Scene in `1_simulation_scenes/` | Contents | Used for |
|---|---|---|
| `simple_car.usd` | the car and the original rounded-square track | single-car baseline, behavioral cloning, pure pursuit (§1–§4) |
| `vec_train_car2.usd` | many copies of car and track side by side, one car each | parallel single-car training, collapse diagnostic (§5, §6) |
| `vec_deploy_car2.usd` | one original track with the curvature-aware car | demos of the curvature-lookahead car (§5) |
| `race_train.usd` | several copies of the original track with a pair of cars each | two-car self-play training (§7) |
| `race_deploy.usd` | one pair of cars on the original track | demos and race evaluation on the original track (§7, §8) |
| `unknown_track_train.usd` | several copies of the stadium track with a pair of cars each | adaptation training on the stadium track (§7) |
| `unknown_track.usd` | one pair of cars on the stadium track | demos, lap benchmark and race evaluation on the stadium track (§7, §8) |

The scenes are binary USD files; open them in Isaac Sim (*File → Open*), not in a text editor.

| Subfolder of `2_python_scripts/` | Part of the project |
|---|---|
| `a_simulation_setup/` | building the car and the original track (§1) |
| `b_single_car/` | single-car PPO baseline (§2) |
| `c_behavioral_cloning/` | behavior-cloning experiment (§3) |
| `d_pure_pursuit/` | classical pure-pursuit baseline (§4) |
| `e_curvature_lookahead/` | parallel training + curvature look-ahead (§5) |
| `f_racing_line/` | racing-line reward variant (§6) |
| `g_two_car_racing/` | two-car self-play racing (§7) |
| `h_evaluation/` | comparative evaluation (§8) |
| `i_verification/` | verification and testing (§9) |

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

`2_python_scripts/a_simulation_setup/create_simple_car.py`, `create_track.py`, `add_car_body.py`, `add_second_car.py`, `remove_car_body.py`, `set_front_wheel_drive.py`, `drive_car.py`

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

`2_python_scripts/b_single_car/car_track_env.py`, `train_car2.py`, `run_dual_car.py`, `compare_smoothness.py`

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
`car2_ppo_model.zip` — the **700k smoothness policy**. The 450k policy is `3_trained_models/checkpoints/car2_ppo_model_pre_smoothness.zip`. `run_dual_car.py` deploys it live: car 1 keyboard-driven, car 2 policy-driven, with an auto-recovery net that teleports the car back onto the reference line if it drifts off-track.

### Notes
- The reference line of this stage is the road centerline shifted by +3 m in y (the stage was derived from a two-lane scene), and the car drives the original vertex order, which is **clockwise seen from above**.
- Measured on physical metrics (§8), the 450k policy laps in 73.9 s and the 700k policy in 60.6 s — the smoothness term also made the car 18% faster, presumably by wasting less wheel speed on steering corrections. Both left the track in some long runs (7/20 and 2/20), always after more than 800 control steps, i.e. far beyond the 500-step (33 s) training episode.
- The training script resumes from an existing checkpoint but only saves at the end of each invocation — killing a run loses that run's progress. Checkpoint versioning was added later in the two-car lineage (§7).

---

## 3. Behavior-Cloning-Assisted Fine-Tuning (Negative Result)

An attempt to warm-start the single-car policy from human driving demonstrations before PPO fine-tuning — kept as a documented negative result.

Implemented in:

`2_python_scripts/c_behavioral_cloning/record_driving.py`, `pretrain_bc.py`, `warmup_critic.py`, `finetune_bc.py` (`_v2` through `_v5`)

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

`2_python_scripts/d_pure_pursuit/pure_pursuit_baseline.py`

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

`2_python_scripts/e_curvature_lookahead/build_vec_track_env.py`, `build_vec_deploy_stage.py`, `car_track_vec_env.py`, `train_car2_vec.py`, `run_dual_car_vec.py`

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

`2_python_scripts/f_racing_line/car_track_racingline_vec_env.py`, `train_car2_racingline.py`

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

Two cars share one track and are driven by a single shared policy network (self-play), each observing the other's relative state. Trained for 109 rounds and exactly 100 million steps: 49 rounds on the original track, then 60 rounds on a second, unseen "stadium" track.

Implemented in:

`2_python_scripts/g_two_car_racing/build_race_track_env.py`, `build_race_deploy_stage.py`, `car_race_vec_env.py`, `train_race_vec.py`, `run_race_demo.py`, `run_race_solo_demo.py`, `run_race_playable_demo.py`, `diag_race_solver_fix.py`, `run_accel_demo.py`

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
| I | competition 7–11 | 95.2–100.0M | none (milestone: last round 821,248 steps, ends at exactly 100,000,000) | 99–100% in every quarter, best episode return 269.9 |
| – | staggered 1–3 (discarded) | 100.0–103.0M | 75% of resets start the trailing car 1.5–8 m behind | 95–100%, but driving degraded (see below); rolled back to 100.0M |

Key episodes along the way:

1. **Collision (proximity) penalty:** 5.0 per step within 1.8 m (normal overtaking distance on a 9 m track) was 15–50× larger than every other per-step term; the full-length rate fell 79% → 44% within round 2. Fixed to 0.4 within 1.2 m from round 3. A later value of 3.0 reproduced the same pathology more slowly and was backed off to 1.5.
2. **Chassis-collapse bug:** under throttle, a car's chassis settled at exactly 0.2 m, half its box height. Five car-side fixes and a ground physics material changed nothing; the cause was the ground collider's **shape** — a finite box instead of an analytic plane. Switching to `PhysicsSchemaTools.addGroundPlane` removed it (stress-test height error 0.000 m over 400 steps; 2.35% residual recovery rate with the trained policy).
3. **Regression after the fix:** correcting the training ground in the same round as a centering-weight change crashed the full-length rate to 4–6%. Reverting only the reward change started the recovery — hence the rule "one change per round" and the checkpoint versioning (`3_trained_models/checkpoints/`, from round 29).
4. **Stadium track:** 40 m / 8 m straights and 6 m corners (vs. 16 m straights and 10 m corners), same width. Zero-shot, the run-49 policy reached the time limit as often as on its own track (93.3%), but its reward fell from +141.7 to −155.5 (see §8 for what that meant physically). Adaptation training continued the same network on an 8-pair stadium stage.
5. **Reward redesign:** a lap-time benchmark revealed a slowdown by round 12 (22.1 s → 25.7 s flying laps in the benchmark) and demos showed one car following the other; throttle bonus and an overtaking incentive were added, then a smoothness term after the driving looked jerky, then a larger overtaking incentive.
6. **Spawn-position bias:** the stadium deploy stage spawns both cars at fixed positions; the car in position B (the inside of the first corner) led at every lap line in 6 of 8 benchmarks. `run_unknown_track_demo.py` and `lap_time_benchmark.py` now swap the starting positions at random (the demo swaps the authored USD translates before `World.reset()`; a runtime teleport before `world.play()` froze the rendered demo after 90 control actions). The swap alone did not remove the effect, because the cause is not the spawn spot — see 7.
7. **Lock-in, not a stronger car (2026-09-24):** with the final checkpoint car B led at all ten lap lines from both spawn spots. Comparing every authored attribute of the two cars found no physical difference; a per-step trace shows a mirror-image start, with the outcome decided in the first corner (20–37 m in), where the early leader either runs wide and yields (~2 m/s) or keeps its line. Whichever car starts even a few tenths of a meter further forward leads at the end in 75–83% of 24-heat tests; the follower then stays ~13 m behind. A mirrored design (every physical start played by both cars) gave the same start the win in 11 of 12 seeds; the benchmark's fixed start is a near tie decided by numerical noise. An earlier reading of the first test (car B led 18 of 24) was an artifact of shared random offsets and is withdrawn.
8. **Staggered-start training (discarded):** `train_unknown_track.py` accepts a third argument `stagger_prob` (for the run above `0.75`), which makes `car_unknown_track_env.py` start the trailing car 1.5–8 m behind the leader on a random side. Three rounds (3M steps) with the reward unchanged kept the training statistics healthy (95–100% full-length) but degraded the driving: overtaking from behind 8/36 → 1/36 heats, clean race-evaluation heats 12/12 → 4/12, 7 vehicles off the track (0 before), flying laps 22.78 → 24.04 s (mean over all heats). Nine of eleven first incidents came after step 500, beyond the 500-step training episodes — evaluation must use the full 2,000-step horizon. The checkpoint was rolled back to the 100M policy (`3_trained_models/checkpoints/`).

### Output
`car_race_ppo_model.zip` — the final checkpoint, competition round 11 (100,000,000 steps; the 100M milestone; the staggered-start experiment was rolled back to it). Every round from original-track round 29 onward (plus round 27), including run 49 and competition round 6 (95,172,864 steps, the previous final checkpoint), is in `3_trained_models/checkpoints/` (see [Checkpoints](#checkpoints)).

### Notes
Measured with the race evaluation (§8), the final policy laps the stadium track in 22.78 s (12 of 12 heats free of incident) and the original track in 21.58 s (10 of 12 heats free of incident; brief chassis-height events in two, no vehicle left the track) — but it does not race: the car ahead after the first corner leads to the end (0 lead changes in 12 heats per track; on the stadium track the inside starter led in 11 of 12). Compared with stadium round 3 (22.05 s, action delta 0.042), the final policy is 3.3% slower and about three times less smooth (0.127): the redesign reversed the round-12 slowdown but did not improve on round 3. Compared with competition round 6, the five further rounds (4.8M steps) raised the training returns by roughly a tenth but left the driving marginally slower (+0.15 s), wider (0.29 m vs. 0.16 m mean offset) and no more competitive; with one run per configuration and 12 heats per checkpoint these differences are descriptive. The jerky driving had already built up during phase E, before the overtaking terms existed, and the smoothness term's effect faded after smoothness round 20.

---

## 8. Comparative Evaluation

Direct physical measurements of every lineage, written to check headline results that rested only on training rewards or demos. Write-ups: `COMMON_METRIC_EVALUATION.md`, `HEAD_TO_HEAD_EVALUATION.md`.

Implemented in:

`2_python_scripts/h_evaluation/common_metric_eval.py`, `collapse_reward_diagnostic.py`, `head_to_head_eval.py`, `two_car_race_eval.py`, `start_condition_eval.py`, `start_trace.py`, `stagger_eval.py`, `lap_telemetry.py`, `plot_lap_telemetry.py` (earlier, superseded: `race_mixed_policy_eval.py`, `run_mixed_race_demo.py`)

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
start_condition_eval.py     many short heats from the benchmark's mid-straight start with tiny random
                            start offsets, both spot assignments; optional mirrored mode (each physical
                            start played by both cars) separates spot, label, and initial-condition effects
start_trace.py              per-control-step trace of both cars from the fixed benchmark start (spawn swap
                            forced on/off) plus a comparison of the authored attributes of the two cars
stagger_eval.py             passing test: the trailing car starts 2/4/6 m behind the leader, both cars and
                            both sides in the trailing role, 500 steps; does it lead at the end?
lap_telemetry.py            Formula 1-style telemetry of the final policy: 10 laps from the fixed stadium
                            start (optionally swapped, or Car A placed ahead), every control step logs
                            speed, throttle, longitudinal acceleration, steering command and measured
                            wheel angle; lap and sector times (3 equal sectors, timing line at the grid)
plot_lap_telemetry.py       graphs of the telemetry: track map with sectors and a timing strip, sector times
                            lap by lap, traces along the lap, acceleration and steering lap by lap (runs
                            with Isaac Sim's Python)
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
| competition round 6 | stadium | 22.63 ± 0.12 | 0.16 | +0.08 | 0.120 | 0 | 0 | 10.3 |
| competition round 11 (final, 100M) | stadium | 22.78 ± 0.22 | 0.29 | −0.09 | 0.127 | 0 | 0 | 16.2 |
| run 49 | original | 21.03 ± 0.08 | 0.07 | +0.01 | 0.203 | 6 | 0 | 6.5 |
| competition round 6 | original | 21.43 ± 0.08 | 0.17 | +0.01 | 0.220 | 0 | 0 | 12.7 |
| competition round 11 (final, 100M) | original | 21.58 ± 0.19 | 0.29 | −0.05 | 0.228 | 2 | 0 | 16.2 |

**Start conditions and staggered training** (`start_condition_eval.py`, `stagger_eval.py`, single-pair stage, deterministic policy; results in `start_condition_results/`; the 100M checkpoint before and after three staggered-start rounds, race evaluation on the stadium track):

| Measure | Competition round 11 (100M) | After staggered-start training |
|---|---|---|
| Car that started further forward leads at the end (24 heats, small random offsets) | 20 of 24 (round 4: 19, round 6: 18) | – |
| Same physical start wins with both cars, mirrored design (12 seeds) | 11 of 12 (Car B leads 13 of 24) | – |
| Trailing car leads at the end, passing test (36 heats) | 8 (22%) | 1 (3%) |
| Race evaluation, heats free of incident | 12 of 12 | 4 of 12 |
| Vehicles that left the track | 0 | 7 |
| Flying lap, all heats [s] | 22.78 ± 0.22 | 24.04 ± 3.47 |
| Lateral offset [m] / corner offset [m] | 0.29 / −0.09 | 0.58 / −0.60 |
| Lead changes (all after an incident of the leader) | 0 | 7 |
| Ten-lap benchmark, mean flying lap [s] | 22.8 | 25.7 |

**Lap telemetry of the final policy** (`lap_telemetry.py`, stadium track, 10 laps, three runs: authored start, positions exchanged, Car A 2 m ahead; graphs `4_results/lap_telemetry_100M*_{track,sectors,traces,laps}.png`: track map with a timing strip, sector times lap by lap, traces along the lap, acceleration and steering lap by lap). Flying laps 2–10, ranges over the three runs:

| Measure | Leader | Follower |
|---|---|---|
| Flying lap [s] | 22.70 | 22.80–22.92 |
| Sector 1 / 2 / 3 [s] (44.6 m each) | 7.56–7.62 / 7.50–7.52 / 7.58–7.63 | 7.61–7.72 / 7.52–7.53 / 7.66–7.75 |
| Control steps at full throttle | 99.97–100% | 94.1–95.7% |
| Strongest deceleration [m/s²] | 1.10–1.24 | 2.72–3.36 |
| Lowest speed [m/s] | 5.78–5.81 | 5.11–5.38 |
| Mean absolute steering angle [°] | 10.1–10.3 | 10.5–10.6 |
| Steering reversals per lap (mean) | 165–170 | 62–69 |
| Mean distance from the centerline [m] | 0.41–0.47 | 0.17 |

The order is settled in sector 1 of the first lap; the interval then grows to 3.0–3.4 s within one to three laps and stays there. The back straight (sector 2) is identical for both roles (speed limit); the follower loses its time in the corner sectors. The leader never lifts and steers with rapid small corrections; the follower lifts before every corner and steers more calmly. With Car A placed ahead, the signatures swap with the roles — they belong to the role, not to the car.

### Notes
- **Training rewards measured the wrong thing in several places**: collapsed cars holding throttle (§5, §6), a slowdown hidden by a saturated full-length rate (phase E), and a zero-shot reward drop caused mainly by the centering penalty (run 49 drives the stadium track 0.89 m off the centerline, on its own track 0.07 m; with a weight of 0.9 that difference alone is worth ~350 reward per episode).
- **Why the self-play cars do not race:** both cars are identical, driven by the same deterministic network, and reach the same top speed on every straight, so a follower can only gain through a shorter line the leader already occupies, or a mistake of the leader. At the final weight, gaining one car length is worth 0.5 × 2 = 1.0 — less than one step of the proximity penalty (1.5) — and over an episode the overtaking term telescopes to at most 0.5 × 30 = 15, against 150 for leaving the track; giving way costs nothing. The start-condition tests show the result: an initial lead of a few tenths of a meter decides ~80% of heats (lock-in), independent of the car or the spawn spot. Changing only the start distribution (staggered training starts) did not fix this and degraded the driving. Untried: training and evaluation on longer episodes (the failures appeared after step 500 of the 2,000-step evaluation heats), a milder stagger, a reward paid for a completed pass, a lower collision penalty or trigger distance, opponents of different strength (e.g. earlier checkpoints), and asymmetric-role self-play.
- **Training statistics do not see failures beyond the training horizon:** the staggered-start policy had 95–100% full-length 500-step training episodes, yet 9 of 11 first incidents in 2,000-step evaluation heats occurred after step 500 (median step 1,509).
- The first head-to-head (`race_mixed_policy_eval.py`, "racing-line covers 19% more distance") ran on the box-ground stage against an early ~5M-step checkpoint and was dominated by collapse recoveries; it is withdrawn.
- Evaluation code needed the same scrutiny as training code: the curvature-lookahead policy was first evaluated in the wrong direction, the demo's recovery counters mix per-car and per-step counts (the "45% of decisions" recovery rate is 22.6% per car decision), the fixed start of the lap benchmark is a near tie decided by numerical noise (the same car won from both spawn spots), and a first test of start offsets suggested a Car B label effect that came from offsets shared between configurations — only a mirrored design (each physical start played by both cars) removed it.

---

## 9. Verification & Testing

Utility and diagnostic scripts used during development to validate track geometry, coordinate-frame conventions, and environment correctness, rather than part of the main training pipeline.

Implemented in:

`2_python_scripts/i_verification/test_car_track_env.py`, `test_xform_order.py`, `validate_clockwise.py`, `verify_car.py`, `verify_deploy_fixes.py`, `verify_track.py`, `verify_vec_track_env.py`, `smoke_test_race_env.py`, `smoke_test_vec_env.py`, `diagnose_stage.py`

---

## Limitations

- Simulation only (Isaac Sim 5.1); no sim-to-real transfer was attempted. The car is simplified: no tire model, suspension, or powertrain dynamics, rigid-body contact with friction ≈ 0.5, top speed 6 m/s.
- The policies observe a privileged, noise-free track-relative state; no perception, localization error, latency, or domain randomization.
- One algorithm (PPO, SB3 defaults, 2×64 networks), no hyperparameter search; every lineage was trained once from a single unseeded run, so the variation between runs is unknown and differences are reported descriptively.
- Two tracks from one family (four straights, four circular corners, same width); training episodes last 33 s (500 control steps) while race evaluations run 2,000 steps, and some failures only appear in the longer runs (the staggered-start experiment showed this).
- The curvature-lookahead and racing-line lineages were never retrained on corrected ground.
- The self-play setup uses two identical cars and one shared network; it did not produce contested racing, neither after reward changes, nor after 100 million steps, nor with staggered training starts. Evaluation samples are small (12 heats per checkpoint and track, 24 heats per start-condition test, one training run per configuration), so differences between checkpoints are descriptive.
- Monitor CSVs are overwritten by every round, so per-quarter statistics of early rounds survive only in the records written at the time (`7_journal/rl_journal.html`).

## Requirements

- NVIDIA Isaac Sim 5.1.0 (standalone)
- Stable-Baselines3 2.9, Gymnasium 1.2, NumPy 1.26, PyTorch 2.7

All scripts are run via Isaac Sim's bundled Python interpreter, e.g.:

```
python.bat 2_python_scripts/g_two_car_racing/train_race_vec.py 1000000
python.bat 2_python_scripts/g_two_car_racing/train_unknown_track.py 1000000 <run_label> 0.75   # stadium track, optional staggered starts
python.bat 2_python_scripts/h_evaluation/two_car_race_eval.py stadium 12 2000 100M_milestone
```

Every script finds the scenes, models and results relative to its own location, so the project folder can be placed anywhere and the scripts can be started from any working directory.

`run_test.bat` in the project folder repeats the measurements reported in the thesis without typing any paths. It finds Isaac Sim by itself (`ISAACSIM_PATH`, `C:\isaacsim`, next to the project folder, the Desktop or Downloads; otherwise it asks once and remembers the answer in `isaacsim_path.txt`), installs Stable-Baselines3 2.9.0 and Gymnasium 1.2.1 into Isaac Sim on first use, and prints the exact command before running it:

```
.\run_test.bat        # PowerShell: shows the list of tests (the same happens on a double-click)
.\run_test.bat 3      # runs test 3: the 100M-step policy on the stadium track
```

## Checking the results without the simulator

`2_python_scripts/check_results.py` (standard-library Python, so it runs on Windows, macOS and Linux; NumPy only for its last part) recomputes the thesis numbers from the committed result files: the race-evaluation lap statistics over incident-free heats, the totals over the 108 stadium heats, the lock-in, mirrored and passing tests, the training steps stored in each model file, the lap and sector times and the throttle and steering of leader and follower in the lap telemetry, and the direction of travel in the behavioral-cloning datasets.

```
python 2_python_scripts/check_results.py      # macOS / Linux: python3
```

## Checkpoints

`3_trained_models/checkpoints/` holds one Stable-Baselines3 checkpoint per training round (87 files, 14 MB); the evaluation scripts load them by label, `car_race_ppo_model_<label>.zip`.

| Label | Thesis name | Steps |
|---|---|---|
| `car2_ppo_model_pre_smoothness` | single-car baseline, 450k policy | 450k |
| `run27`, `run29` … `run49` | original track, rounds 27 and 29–49 (versioning began at round 29) | 18.1–40.1M |
| `newtrack_run1` … `newtrack_run12` | stadium rounds 1–12 (phase E) | 41.1–52.1M |
| `newtrack_overtake_run1`, `_run2` | overtaking rounds 1–2 (phase F) | 53.1–54.1M |
| `newtrack_smooth_run1` … `_run35` | smoothness rounds 1–35 (phase G) | 55.1–89.2M |
| `newtrack_compete_run1` … `_run11` | competition rounds 1–11 (phases H and I) | 90.2–100.0M |
| `100M_milestone` | competition round 11, identical to `newtrack_compete_run11` and to `car_race_ppo_model.zip` | 100,000,000 |
| `newtrack_stagger_run1` … `_run3` | discarded staggered-start rounds | 101.0–103.0M |
