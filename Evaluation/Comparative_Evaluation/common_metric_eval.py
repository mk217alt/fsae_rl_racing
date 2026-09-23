"""Common-metric evaluation of the single-vehicle controllers.

The single-vehicle lineages were trained under DIFFERENT reward functions
(centering weight 0.6 vs 0.1, throttle bonus 0.25 vs 0.4, with/without a
smoothness term), so their episode rewards are not comparable. This script
measures every controller with the same physical quantities instead:

  * lap times (flying laps, timed from cumulative progress along the PHYSICAL
    road centerline, independent of each policy's own reference line),
  * mean speed along the road centerline,
  * lateral position relative to the road centerline (how much of the
    9 m track width the controller uses),
  * off-track terminations, and chassis-collapse steps (height < 0.30 m),
  * mean consecutive-action change |d throttle| + |d steer| (smoothness).

Each controller is evaluated deterministically in the exact environment it
was trained in (stage, reference line, direction of travel, physics), from
N_RUNS seeded random start segments, for up to TARGET_LAPS laps.

Direction of travel: the vec lineage was trained on the ORIGINAL point order of
build_centerline(); the `reversed(points)` line in car_track_vec_env.py was added
afterwards (2026-09-17 "make it clockwise" request, 5/5 reversed trials failed
within ~50-60 steps) and never retrained. The racing-line and two-car lineages
were trained after that change, on the reversed order. TRAINED_REVERSED below
restores each lineage's training direction (results obtained before this fix
are kept as *_reversed_direction.* - a reversed-direction control).

Usage:  python.bat common_metric_eval.py <controller> [plane]
        controller in: baseline450, baseline700, purepursuit, vec, racingline
        plane (vec / racingline only): replace the stage's finite box ground collider by the analytic
        ground plane used by all corrected stages (same physics material), to separate the policy's
        driving from the chassis-collapse defect of the box collider.
Results: common_metric_results_<controller>[_plane].json (+ .txt summary)
"""
import json
import math
import os
import sys

import numpy as np

THESIS = "C:/Users/sanja/Desktop/thesis"
OUT_DIR = f"{THESIS}/Evaluation/Comparative_Evaluation"
CONTROLLER = sys.argv[1]
GROUND = sys.argv[2] if len(sys.argv) > 2 else "asbuilt"
TAG = CONTROLLER + ("_plane" if GROUND == "plane" else "")
TRACE = os.environ.get("TRACE") == "1"   # record the path of the first run only (thesis figure)
N_RUNS = 1 if TRACE else 20
TARGET_LAPS = 3
STEP_CAP = 2500                   # control steps (= 166.7 s of simulated time)
COLLAPSE_Z = 0.30                 # nominal chassis height is 0.45 m, collapsed state rests at 0.20 m
PHYSICS_DT = 1.0 / 60.0
ACTION_REPEAT = 4
CONTROL_DT = PHYSICS_DT * ACTION_REPEAT

MODELS = {
    "baseline450": f"{THESIS}/checkpoint_backups/car2_ppo_model_pre_smoothness.zip",
    "baseline700": f"{THESIS}/Baselines/Single_Car_Baseline/car2_ppo_model.zip",
    "vec": f"{THESIS}/Racing/Vec_Curvature_Lineage/car2_vec_ppo_model.zip",
    "racingline": f"{THESIS}/Racing/Racing_Line_Variant/car2_racingline_ppo_model.zip",
}
TRAINED_REVERSED = {"vec": False, "racingline": True}   # point order each vec-stage policy was trained on


def use_training_direction(mod, reversed_order):
    """Make mod.build_centerline() return the point order the policy was trained on."""
    orig = mod.build_centerline

    def build():
        pts, cum = orig()                             # current file: reversed order
        if reversed_order:
            return pts, cum
        pts = list(reversed(pts))
        cum = [0.0]
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
        return pts, cum

    mod.build_centerline = build


# ---------------------------------------------------------------------------
# physical road centerline (identical construction to create_track.py /
# build_vec_track_env.py: rounded square, half-straight 8 m, corner radius 10 m)
# ---------------------------------------------------------------------------
class RoadCenterline:
    def __init__(self, reverse):
        a, r, n = 8.0, 10.0, 10
        corners = [((a, a), 90, 0), ((a, -a), 0, -90), ((-a, -a), -90, -180), ((-a, a), -180, -270)]
        pts = []
        for c, s0, s1 in corners:
            for k in range(n + 1):
                t = math.radians(s0 + (s1 - s0) * k / n)
                pts.append((c[0] + r * math.cos(t), c[1] + r * math.sin(t)))
        pts = [(x, y + a + r) for x, y in pts]
        if reverse:
            pts = list(reversed(pts))
        self.p = pts
        self.n = len(pts)
        self.cum = [0.0]
        for i in range(self.n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % self.n]
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.length = self.cum[-1]

    def project(self, x, y):
        best = (float("inf"), 0, 0.0)
        for i in range(self.n):
            x0, y0 = self.p[i]
            x1, y1 = self.p[(i + 1) % self.n]
            dx, dy = x1 - x0, y1 - y0
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)))
            px, py = x0 + t * dx, y0 + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best[0]:
                best = (d2, i, t)
        d2, i, t = best
        x0, y0 = self.p[i]
        x1, y1 = self.p[(i + 1) % self.n]
        dx, dy = x1 - x0, y1 - y0
        px, py = x0 + t * dx, y0 + t * dy
        side = math.copysign(1.0, dx * (y - py) - dy * (x - px))
        return self.cum[i] + t * math.hypot(dx, dy), side * math.sqrt(d2)


class RunTracker:
    """Accumulates the physical metrics of one run."""

    def __init__(self, road, x, y):
        self.road = road
        self.last_s, _ = road.project(x, y)
        self.total = 0.0
        self.steps = 0
        self.lap_marks = []           # step index at which each lap completed
        self.offsets = []
        self.collapse_steps = 0
        self.action_deltas = []
        self.last_action = None
        self.off_track = False
        self.finished = False
        self.path = []

    def update(self, x, y, z, action):
        s, d = self.road.project(x, y)
        if TRACE:
            self.path.append((round(float(x), 3), round(float(y), 3)))
        ds = s - self.last_s
        if ds > self.road.length / 2:
            ds -= self.road.length
        elif ds < -self.road.length / 2:
            ds += self.road.length
        self.last_s = s
        self.total += ds
        self.steps += 1
        self.offsets.append(d)
        if z < COLLAPSE_Z:
            self.collapse_steps += 1
        a = np.asarray(action, dtype=np.float64)
        if self.last_action is not None:
            self.action_deltas.append(float(np.abs(a - self.last_action).sum()))
        self.last_action = a
        while self.total >= (len(self.lap_marks) + 1) * self.road.length:
            self.lap_marks.append(self.steps)
        if len(self.lap_marks) >= TARGET_LAPS or self.steps >= STEP_CAP:
            self.finished = True

    def summary(self):
        marks = [0] + self.lap_marks
        laps = [(marks[k + 1] - marks[k]) * CONTROL_DT for k in range(len(self.lap_marks))]
        off = np.abs(np.array(self.offsets)) if self.offsets else np.zeros(1)
        return {
            "steps": self.steps,
            "laps_completed": len(self.lap_marks),
            "lap_times_s": [round(t, 3) for t in laps],
            "distance_m": round(self.total, 2),
            "mean_speed_mps": round(self.total / (self.steps * CONTROL_DT), 3) if self.steps else 0.0,
            "off_track": self.off_track,
            "collapse_steps": self.collapse_steps,
            "mean_abs_offset_m": round(float(off.mean()), 3),
            "p95_abs_offset_m": round(float(np.percentile(off, 95)), 3),
            "share_offset_gt_2m": round(float((off > 2.0).mean()), 4),
            "mean_action_delta": round(float(np.mean(self.action_deltas)), 4) if self.action_deltas else None,
            "end_centerline_s_m": round(self.last_s, 2),        # where on the lap the run ended
            "end_offset_m": round(self.offsets[-1], 3) if self.offsets else None,
        }


def aggregate(runs, road_length):
    flying = [t for r in runs for t in r["lap_times_s"][1:]]      # lap 1 = standing start
    first = [r["lap_times_s"][0] for r in runs if r["lap_times_s"]]
    return {
        "controller": CONTROLLER,
        "ground": GROUND,
        "runs": len(runs),
        "road_centerline_length_m": round(road_length, 3),
        "runs_completing_target_laps": sum(1 for r in runs if r["laps_completed"] >= TARGET_LAPS),
        "off_track_runs": sum(1 for r in runs if r["off_track"]),
        "runs_with_collapse": sum(1 for r in runs if r["collapse_steps"] > 0),
        "flying_laps": len(flying),
        "flying_lap_mean_s": round(float(np.mean(flying)), 3) if flying else None,
        "flying_lap_std_s": round(float(np.std(flying)), 3) if flying else None,
        "flying_lap_best_s": round(float(np.min(flying)), 3) if flying else None,
        "first_lap_mean_s": round(float(np.mean(first)), 3) if first else None,
        "mean_speed_mps": round(float(np.mean([r["mean_speed_mps"] for r in runs])), 3),
        "mean_abs_offset_m": round(float(np.mean([r["mean_abs_offset_m"] for r in runs])), 3),
        "p95_abs_offset_m": round(float(np.mean([r["p95_abs_offset_m"] for r in runs])), 3),
        "share_offset_gt_2m": round(float(np.mean([r["share_offset_gt_2m"] for r in runs])), 4),
        "mean_action_delta": round(float(np.mean([r["mean_action_delta"] for r in runs
                                                  if r["mean_action_delta"] is not None])), 4),
    }


def write_results(runs, road_length, paths=None):
    if TRACE:
        with open(f"{OUT_DIR}/trace_{TAG}.json", "w") as f:
            json.dump({"controller": CONTROLLER, "ground": GROUND, "run": runs[0], "path": paths[0]}, f)
        return
    agg = aggregate(runs, road_length)
    with open(f"{OUT_DIR}/common_metric_results_{TAG}.json", "w") as f:
        json.dump({"summary": agg, "runs": runs}, f, indent=2)
    with open(f"{OUT_DIR}/common_metric_results_{TAG}.txt", "w") as f:
        for k, v in agg.items():
            f.write(f"{k}: {v}\n")
    print(json.dumps(agg, indent=2), flush=True)


# ---------------------------------------------------------------------------
# single-environment controllers (car_track_env.py, simple_car.usd)
# ---------------------------------------------------------------------------
def eval_single():
    sys.path.insert(0, f"{THESIS}/Baselines/Single_Car_Baseline")
    import car_track_env as cte
    cte.MAX_EPISODE_STEPS = 10 ** 9                    # run past the 500-step training limit
    env = cte.CarTrackEnv(headless=True)
    road = RoadCenterline(reverse=False)               # same direction of travel as car_track_env

    model = None
    if CONTROLLER != "purepursuit":
        from stable_baselines3 import PPO
        model = PPO.load(MODELS[CONTROLLER], device="cpu")

    ref_pts, ref_cum = env._points, env._cumulative    # policy's own (shifted) reference line
    ref_len = ref_cum[-1]

    def pp_action(pos, yaw, fwd_speed):
        # identical controller and parameters to Baselines/Classical_Baseline/pure_pursuit_baseline.py
        lookahead, wheelbase, v_target, slowdown, kp = 4.0, 1.6, 5.0, 3.0, 0.6
        progress, _, tangent = env._progress_and_offset(pos[0], pos[1])
        s = (progress + lookahead) % ref_len
        tx, ty = ref_pts[0]
        t_tangent = tangent
        for i in range(len(ref_pts)):
            if ref_cum[i] <= s <= ref_cum[i + 1]:
                x0, y0 = ref_pts[i]
                x1, y1 = ref_pts[(i + 1) % len(ref_pts)]
                seg = ref_cum[i + 1] - ref_cum[i]
                t = (s - ref_cum[i]) / seg if seg > 1e-9 else 0.0
                tx, ty = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                t_tangent = math.atan2(y1 - y0, x1 - x0)
                break
        alpha = cte.wrap_to_pi(math.atan2(ty - pos[1], tx - pos[0]) - yaw)
        steer = float(np.clip(math.atan(wheelbase * 2.0 * math.sin(alpha) / lookahead) / cte.MAX_STEER_ANGLE,
                              -1.0, 1.0))
        target_speed = v_target / (1.0 + slowdown * abs(cte.wrap_to_pi(t_tangent - tangent)))
        throttle = float(np.clip(kp * (target_speed - fwd_speed), -1.0, 1.0))
        return np.array([throttle, steer], dtype=np.float32)

    runs, paths = [], []
    for k in range(N_RUNS):
        env._rng = np.random.default_rng(1000 + k)
        obs, _ = env.reset()
        pos, quat = env._car.get_world_poses()
        tr = RunTracker(road, pos[0][0], pos[0][1])
        while not tr.finished:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                w, x, y, z = quat[0]
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                action = pp_action(pos[0], yaw, float(obs[3]))
            obs, _, terminated, _, _ = env.step(action)
            pos, quat = env._car.get_world_poses()
            tr.update(pos[0][0], pos[0][1], pos[0][2], np.clip(action, -1, 1))
            if terminated:
                tr.off_track = True
                break
        runs.append(tr.summary())
        paths.append(tr.path)
        print(f"run {k}: {runs[-1]}", flush=True)
    write_results(runs, road.length, paths)
    env.close()


def patch_ground_plane():
    """Swap /World/GroundPlane (finite box) for the analytic plane when the stage is opened."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Gf, PhysicsSchemaTools, UsdPhysics, UsdShade
    orig = stage_utils.open_stage

    def open_with_plane(path):
        ok = orig(path)
        stage = stage_utils.get_current_stage()
        stage.RemovePrim("/World/GroundPlane")
        PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundPlane", "Z", 2000.0,
                                          Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5))
        mat = UsdShade.Material.Define(stage, "/World/GroundPhysicsMaterial")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(0.5)
        api.CreateDynamicFrictionAttr(0.5)
        api.CreateRestitutionAttr(0.8)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/GroundPlane/geom")).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        return ok

    stage_utils.open_stage = open_with_plane


# ---------------------------------------------------------------------------
# parallel-environment controllers (vec_train_car2.usd, 10 clones)
# ---------------------------------------------------------------------------
def eval_vec():
    if CONTROLLER == "racingline":
        sys.path.insert(0, f"{THESIS}/Racing/Racing_Line_Variant")
        import car_track_racingline_vec_env as mod
        cls = mod.CarTrackRacinglineVecEnv
    else:
        sys.path.insert(0, f"{THESIS}/Racing/Vec_Curvature_Lineage")
        import car_track_vec_env as mod
        cls = mod.CarTrackVecEnv
    mod.MAX_EPISODE_STEPS = 10 ** 9
    use_training_direction(mod, TRAINED_REVERSED[CONTROLLER])
    mod.get_simulation_app(headless=True)
    if GROUND == "plane":
        patch_ground_plane()
    from stable_baselines3 import PPO
    env = cls(num_envs=mod.NUM_ENVS, headless=True)
    model = PPO.load(MODELS[CONTROLLER], device="cpu")
    road = RoadCenterline(reverse=TRAINED_REVERSED[CONTROLLER])
    n = env.num_envs
    runs, paths = [], []
    for rnd in range(math.ceil(N_RUNS / n)):
        env.seed(2000 + rnd)
        obs = env.reset()
        pos, _ = env._car.get_world_poses()
        trackers = [RunTracker(road, pos[i][0] - env._offsets[i, 0], pos[i][1] - env._offsets[i, 1])
                    for i in range(n)]
        while not all(t.finished for t in trackers):
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(actions)
            pos, _ = env._car.get_world_poses()
            for i, tr in enumerate(trackers):
                if tr.finished:
                    continue
                if dones[i]:                         # only off-track can end a run (limit disabled)
                    tr.off_track = True
                    tr.finished = True
                    continue
                tr.update(pos[i][0] - env._offsets[i, 0], pos[i][1] - env._offsets[i, 1], pos[i][2],
                          np.clip(actions[i], -1, 1))
        for tr in trackers:
            runs.append(tr.summary())
            paths.append(tr.path)
            print(f"run {len(runs) - 1}: {runs[-1]}", flush=True)
    write_results(runs[:N_RUNS], road.length, paths[:N_RUNS])
    env.close()


if __name__ == "__main__":
    if CONTROLLER in ("baseline450", "baseline700", "purepursuit"):
        eval_single()
    elif CONTROLLER in ("vec", "racingline"):
        eval_vec()
    else:
        raise SystemExit(f"unknown controller {CONTROLLER}")
