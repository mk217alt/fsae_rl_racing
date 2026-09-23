# Common-metric evaluation of the single-vehicle controllers (2026-09-23)

The single-vehicle lineages were trained under different reward functions, so their episode rewards
cannot be compared. `common_metric_eval.py` measures every controller with the same physical
quantities, deterministically, from 20 seeded random start segments, for up to 3 laps or 2500
control steps (166.7 s):

* lap times, from cumulative progress along the **physical** road centerline (126.8 m)
* mean speed along the road centerline
* lateral position relative to the road centerline
* off-track terminations and chassis-collapse steps (chassis height < 0.30 m)
* mean consecutive-action change |dthrottle| + |dsteer| (smoothness)

Deterministic re-runs reproduce identical numbers (checked for `baseline450`).

## Results

| Controller | Stage ground | Runs finishing 3 laps | Off-track runs | Runs with collapse | Flying lap [s] | Mean speed [m/s] | Mean abs. offset [m] | Mean action delta |
|---|---|---|---|---|---|---|---|---|
| Pure-RL baseline, 450k (`checkpoint_backups/car2_ppo_model_pre_smoothness.zip`) | analytic plane | 0/20 | 7 | 0 | 73.87 +/- 0.70 | 1.37 | 1.86 | 1.030 |
| Baseline + smoothness, 700k (`car2_ppo_model.zip`) | analytic plane | 0/20 | 2 | 0 | 60.62 +/- 0.95 | 1.87 | 1.73 | 0.847 |
| Pure pursuit (no learning) | analytic plane | 20/20 | 0 | 0 | 51.43 +/- 0.09 | 2.46 | 1.78 | 0.117 |
| Vec + curvature (`car2_vec_ppo_model.zip`) | box (as trained) | 0/20 | 3 | 17 | - | 0.20 | 0.61 | 0.014 |
| Racing-line (`car2_racingline_ppo_model.zip`) | box (as trained) | 0/20 | 0 | 20 | - | 0.13 | 0.35 | 0.003 |
| Vec + curvature | analytic plane (swapped) | 0/20 | 20 (after 43-53 steps) | 0 | - | 1.47 | 1.07 | 0.065 |
| Racing-line | analytic plane (swapped) | **20/20** | 0 | 0 | **21.27 +/- 0.03** | **5.82** | 0.16 | 0.073 |

Notes:
* **Direction of travel (corrected 2026-09-23).** The vec lineage was trained on the ORIGINAL vertex order
  of `build_centerline()`; `points = list(reversed(points))` was added to `car_track_vec_env.py` only
  afterwards (2026-09-17, "make it clockwise" demo request, 5/5 reversed trials failed within 50-63 steps)
  and the lineage was never retrained. The first run of this evaluation used the current (reversed) file
  and therefore drove the vec policy against its training direction. Both scripts now restore each
  lineage's training direction (`TRAINED_REVERSED`); the vec rows above are the corrected runs. The
  reversed-direction runs are kept as `*_reversed_direction.*` (a control: box 8 off-track / 15 collapse;
  plane 20/20 off-track after 47-62 steps). The racing-line lineage was trained after the change, on the
  reversed order, so its rows are unaffected. Conclusion unchanged: in its own direction, on the analytic
  plane, the vec policy also leaves the track in every run within 43-53 control steps.
* Baseline / pure pursuit offsets are measured from the road centerline; their own reference line is the
  road centerline translated +3 m in y (inherited from the two-car shared stage), hence ~1.8 m mean offset.
* Baseline runs never finish 3 laps within the 2500-step cap (laps take 60-74 s); off-track events all
  occur after 800+ control steps, i.e. beyond the 500-step training episodes.

## Why the vec / racing-line rewards were misleading (`collapse_reward_diagnostic.py`)

`vec_train_car2.usd` (training stage of both lineages) still uses the finite **box** ground collider that
causes the chassis-collapse bug under sustained throttle. One 500-step training episode per clone (10 clones):

| Policy / mode / ground | Mean episode reward (training reward fn) | Mean distance [m] | Mean throttle | Collapsed share of steps |
|---|---|---|---|---|
| racing-line, deterministic, box | 206.2 | 22.9 | 0.997 | 0.79 |
| racing-line, stochastic, box | 190.4 | 25.0 | 0.984 | 0.68 |
| racing-line, deterministic, plane | 375.6 | 189.2 | 0.970 | 0.00 |
| racing-line, stochastic, plane | 329.0 | 159.1 | 0.885 | 0.00 |
| vec, deterministic, box | -216.3 | 3.1 | 0.964 | 0.63 |
| vec, stochastic, box | -105.4 | 3.9 | 0.903 | 0.50 |
| vec, deterministic, plane | -47.5 (off-track at step 45-53, 10/10) | 4.4 | 0.969 | 0.00 |
| vec, stochastic, plane | -44.1 (off-track at step 52-79, 10/10) | 5.8 | 0.853 | 0.00 |

(vec rows: training direction, see the note above; the diagnostic now also disables the environment's own
500-step limit so that only off-track terminations count as early ends.)

A collapsed, stationary car holding full throttle is paid the throttle bonus every step and never leaves
the track: one racing-line clone moved 0.02 m and scored 198.4; vec clones that moved -0.02 m / 0.14 m /
0.21 m scored 116.7 / 112.7 / 121.1. The large negative vec means come from clones that collapsed away from
the centerline (-0.6 x |d| every step, down to -759). These match the lineages' reported training figures
(racing-line average 187.6; vec "peak" 121.9).
