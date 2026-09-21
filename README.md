# FSAE RL Racing

## Overview

The objective of this project is to develop and compare reinforcement-learning approaches for autonomous racing of a Formula Student vehicle, simulated end-to-end in NVIDIA Isaac Sim. A custom four-wheeled car (front-wheel drive, articulated steering) is trained to lap a closed circuit using PPO (Stable-Baselines3), starting from a single-car centerline-following baseline and expanding into several parallel lineages: a behavior-cloning-assisted variant, a classical (non-learned) control baseline for comparison, a curvature-aware parallel-training variant, a racing-line reward variant, and a two-car self-play racing formulation where a single shared policy controls both cars. Every lineage is trained and evaluated in the same simulator, sharing a common car model and track geometry, so results are directly comparable across approaches.

This repository is developed as part of a thesis project.

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
- Generate a closed-loop track centerline (straights and quarter-circle corners), 9m wide, with collidable curb-wall boxes on both sides.
- Duplicate the car into a second lane for two-car scenarios (`add_second_car.py`), remapping all internal joint/material relationship targets.

### Output
`simple_car.usd` — the shared base stage used by the single-car baseline and BC/pure-pursuit lineages.

### Notes
An early version gave the car body a decorative body-kit mesh as a scaled-Cube rigid body, which caused a duplicate-rendering Fabric/USD bug — the body kit was removed entirely rather than debugged further. Track construction also surfaced a subtle USD bug: **transform ops compose in the reverse of the order they're added** (the last-added op acts on the geometry first), so adding Scale→Rotate→Translate actually executes as Translate→Rotate→Scale and corrupts positions for anything with rotation. Fixed by always adding ops in Translate→Rotate→Scale order, and verified by reading back world transforms before trusting new track geometry.

---

## 2. Single-Car PPO Baseline

The accepted, final single-car policy: pure reinforcement learning, no imitation data, trained to follow the track centerline.

Implemented in:

`Baselines/Single_Car_Baseline/car_track_env.py`, `train_car2.py`, `run_dual_car.py`

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
Trained across 7 incremental rounds, progressively raising the centering/off-track reward weights as consistency improved:

| Round | Cumulative steps | Weights (lateral / off-track / throttle) | Result |
|---|---|---|---|
| 1-2 | 50k → 100k | 0.15/-10/— → 0.25/-20/0.05 | inconsistent → improved |
| 3 | 150k | 0.40/-20/0.15 | converged, last 25+ episodes full-length |
| 5 | 250k | 0.60/-30/0.25 | nearly all episodes full-length, reward 89-104 |
| 6 | 350k | (unchanged) | reward climbed to 108-113 |
| 7 | 450k | (unchanged) | held at 105-113.5, ~99% full-length — converged |

### Output
`car2_ppo_model.zip` — 450k steps, final. `run_dual_car.py` deploys it live: car 1 keyboard-driven, car 2 policy-driven, with an auto-recovery net that teleports the car back onto the centerline if it drifts off-track.

### Notes
The training script resumes automatically from an existing checkpoint but only saves once, at the very end of each invocation — killing a run mid-flight loses that run's progress entirely. This gap was later addressed with proper checkpoint versioning in the two-car self-play lineage (see §7).

---

## 3. Behavior-Cloning-Assisted Fine-Tuning (Negative Result)

An attempt to warm-start the single-car policy from human driving demonstrations before PPO fine-tuning — included here as a documented negative result, not a dead end that was silently dropped.

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
Run across three independent rounds, each with a differently sized and styled demonstration dataset, to rule out dataset-specific causes:

1. **Round 1** (2608 samples, natural driving style): fit BC action mean, then hit and fixed two real bugs — `log_std` left at SB3's default (std=1.0, far too wide for a [-1,1] action range; fixed by tightening to 0.15), and an untrained critic (fixed by warm-starting it on 40 rollout episodes' returns). Even with both fixes, fine-tuning at a reduced learning rate plateaued at max reward -36.
2. **Round 2** (1841 samples, deliberately centered driving, all round-1 fixes applied from the start): landed on the same ceiling, max reward -37 — ruled out driving style as the cause.
3. **Round 3** (1422 samples, deliberately varied speed/line): landed on the same ceiling again, max reward -28.2 — ruled out dataset size as the cause too.

### Output
All intermediate checkpoints and datasets were kept rather than deleted, for citing specific numbers: `car2_bc_pretrained{,_v1,_v2}.zip`, `car2_bc_warmed{,_v1,_v2}.zip`, `car2_bc_finetuned{,_v2,_v3,_v4,_v5}.zip`, `human_driving_data{,_v1,_v2}.npz`, and their associated monitor CSVs.

### Notes
Three independent datasets of three different sizes and driving styles, with 5+ real bugs fixed along the way, all converged to essentially the same ceiling — well below the pure-RL baseline's 105-113 reward, with full-length episode share falling from an initially promising start down to 3-8%. Since fixing any single dataset property should have moved the number if it were the real constraint, the most likely explanation is the technique itself: BC pretraining gives PPO a confident but narrow starting basin fit to the small demonstration set, which is harder to escape than exploring broadly from scratch. Pure RL was kept as the accepted baseline.

---

## 4. Classical Baseline: Pure Pursuit

A non-learned control baseline, to measure how much a learned policy actually gains over a well-tuned hand-engineered controller on this task.

Implemented in:

`Baselines/Classical_Baseline/pure_pursuit_baseline.py`

### Workflow
```
Track centerline
        ↓
Bicycle-model pure-pursuit steering
        ↓
Curvature-aware speed target
        ↓
Evaluation on the same env/track as the PPO baseline
```

### Output
40/40 full-length episodes, average reward 108.70 (range 98.47-114.97).

### Notes
This matches the pure-RL baseline's performance (105-113.5) almost exactly. It's a useful thesis data point: for this track and reward function, a hand-engineered controller is competitive with a learned policy — the value of RL here is more about generality (see the racing-line and self-play variants below) than raw single-line lap performance.

---

## 5. Vectorized + Curvature-Lookahead Single-Car Variant

An experimental lineage combining two changes at once: parallel training for speed, and a longer-horizon observation for quality.

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
~2x training throughput vs. the single-env baseline (152-196 vs. ~80 timesteps/sec). At 470k cumulative steps: peak reward 121.9 (exceeds the baseline's 113.5), but consistency is lower (~50-60% full-length episodes vs. the baseline's ~99%).

### Notes
Not adopted as a final policy — a different, not-fully-converged result rather than a strict improvement. The track direction was also reversed to clockwise in this lineage; a trained policy did **not** generalize to the mirrored curvature pattern (5/5 test trials failed within ~50-60 steps), confirming this needed retraining from scratch rather than a code-only fix. The clockwise convention was kept forward into the two-car self-play lineage, which trained fresh on it from the start.

---

## 6. Racing-Line Variant

A reward-shaping experiment: loosen the centerline-adherence penalty and let the physical track boundaries, rather than a tight reward term, bound how the car uses the track.

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
PPO
        ↓
car2_racingline_ppo_model.zip
```

### Output
Trained for 300k steps; a clean, converged improvement curve with no crashes at any point:

| Quartile | Avg reward | Full-length rate |
|---|---|---|
| Q1 | 73.6 | 50% |
| Q2 | 133.2 | 92% |
| Q3 | 178.9 | 96% |
| Q4 | 187.6 | 96% |

### Notes
This **beats the accepted pure-RL baseline** (187.6 vs. 105-113 reward) while matching its consistency — a genuinely better result on this metric, not just a different one, achieved purely through reward shaping on identical architecture and observations.

---

## 7. Two-Car Self-Play Racing

The main active lineage: two cars sharing one track, driven by a single shared policy network (self-play), each observing the other's relative state.

Implemented in:

`Racing/Two_Car_Self_Play_Racing/build_race_track_env.py`, `build_race_deploy_stage.py`, `car_race_vec_env.py`, `train_race_vec.py`, `run_race_demo.py`, `run_race_solo_demo.py`, `run_race_playable_demo.py`, `diag_race_solver_fix.py`, `run_accel_demo.py`

### Workflow
```
N racing-pair clones (2 cars sharing 1 track each)
        ↓
Single SHARED PPO policy controls every car slot
        ↓
10-dim observation (7-dim curvature-aware + opponent-relative:
                     progress gap, lateral gap, relative speed)
        ↓
car_race_ppo_model.zip
```

### Processing Steps
Two structural bugs were found and fixed over the course of training:

1. **Collision-penalty bug**: an oversized per-step collision penalty (5.0, at a 1.8m trigger distance comparable to normal overtaking distance) caused a declining full-length rate and spiraling worst-case reward. Fixed by reducing to 0.4 at a tighter 1.2m distance.
2. **Chassis-collapse physics bug**: under sustained throttle, a car's chassis would reliably settle to exactly half its own collision-box height. Root-caused after ruling out 5 car-side parameters to the ground collider's shape — the training/deploy stages used a finite box collider instead of an analytic ground plane (the convention the single-car stages used via Isaac Sim's own `add_default_ground_plane` helper). Switching to `PhysicsSchemaTools.addGroundPlane` dropped the bug's occurrence from ~100% to ~2.35% in live testing.

A subsequent reward-shaping change (raising the centering penalty alongside the physics fix in the same round) caused a real training regression, which took several isolated recovery rounds to work back from — reverting the reward change while keeping the physics fix, then reintroducing a severe (but calibrated) collision/off-track penalty, recovered full-length rate from single digits back into the 74-89% range with a new peak reward.

### Output
`car_race_ppo_model.zip`, actively training. Checkpoint versioning (`checkpoint_backups/`, gitignored) was added after this regression, since the training script otherwise unconditionally overwrites its one checkpoint every run with no way to roll back.

### Notes
This lineage is the most actively developed and, as of this writing, still short of full convergence — later commits and the accompanying training logs reflect an in-progress recovery, not a final result.

---

## 8. Comparative Evaluation

A head-to-head test between the racing-line single-car policy (no opponent awareness) and the two-car self-play policy, to check whether opponent-awareness alone translates into a competitive advantage.

Implemented in:

`Evaluation/Comparative_Evaluation/race_mixed_policy_eval.py`, `run_mixed_race_demo.py`

### Workflow
```
Racing-line policy (Car A or B, no opponent awareness)
        ↓
vs. Self-play policy (the other car, opponent-aware)
        ↓
Two heats, with lane (Car A / Car B) assignments swapped
        ↓
Distance-covered comparison, summed across both lane assignments
```

### Output
| Heat | Lane A | Lane B | Winner | Margin |
|---|---|---|---|---|
| 1 (original) | racing-line: 372.75m | self-play: 382.21m | self-play, narrowly | +9.46m (2.5%) |
| 2 (swapped) | self-play: 212.89m | racing-line: 336.03m | racing-line, decisively | +123.14m (58%) |

Summed across both lane assignments: racing-line 708.78m total vs. self-play 595.10m total.

### Notes
Swapping lanes flipped the outcome entirely — a single head-to-head heat was not a reliable measurement here, since lane/interaction effects (which car ends up blocking, physics-bug recovery cascades) dominated over any inherent policy-quality difference. The lane-controlled, multi-heat comparison shows the racing-line policy covering ~19% more ground overall, consistent with its much higher solo reward — opponent-awareness did not clearly translate into a competitive advantage on this fixed track with no dynamic obstacles beyond the one opponent. This is worth citing as a methodology point: naive single-heat agent-vs-agent evaluation can be dominated by confounds unrelated to policy quality.

---

## 9. Verification & Testing

Utility and diagnostic scripts used during development to validate track geometry, coordinate-frame conventions, and environment correctness, rather than part of the main training pipeline.

Implemented in:

`Evaluation/Verification_and_Testing/test_car_track_env.py`, `test_xform_order.py`, `validate_clockwise.py`, `verify_car.py`, `verify_deploy_fixes.py`, `verify_track.py`, `verify_vec_track_env.py`, `smoke_test_race_env.py`, `smoke_test_vec_env.py`, `diagnose_stage.py`

---

## Limitations

- All training and evaluation is done entirely in simulation (Isaac Sim); no sim-to-real transfer has been attempted.
- The BC+RL negative result (§3) is specific to this reward/termination setup and to small (1,000-3,000 sample) human demonstration datasets — it is not necessarily generalizable to all imitation-learning approaches.
- The two-car self-play lineage (§7) has a documented training regression and recovery arc; the checkpoint in this repository reflects the most recent training round at the time of the last push, not a guaranteed fully-converged final policy.

## Requirements

- NVIDIA Isaac Sim 5.1.0 (standalone)
- Stable-Baselines3, Gymnasium, NumPy, PyTorch

All scripts are run via Isaac Sim's bundled Python interpreter, e.g.:

```
python.bat Racing/Two_Car_Self_Play_Racing/train_race_vec.py 1000000
```
