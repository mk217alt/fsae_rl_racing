"""Does the racing-line (or vec-curvature) policy earn its training reward by driving, or by
holding throttle on a collapsed chassis?

Runs the policy in its own training environment (vec_train_car2.usd, 10 clones) for exactly one
500-step training episode per clone and records, per clone:
  * the episode reward exactly as the training environment computes it,
  * distance driven along the physical road centerline,
  * control steps with a collapsed chassis (height < 0.30 m) and the step of first collapse,
  * mean throttle / mean |steer| actually commanded.

Options:
  policy  : racingline | vec
  mode    : det (deterministic mean action, as in deployment) | stoch (sampled, as in training)
  ground  : box (stage as trained: finite box collider) | plane (box replaced by the analytic ground
            plane used by the fixed stages, same physics material)

Usage: python.bat collapse_reward_diagnostic.py <policy> <mode> <ground>
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is
import json
import math
import sys

import numpy as np

THESIS = ROOT
OUT_DIR = f"{THESIS}/4_results"
POLICY, MODE, GROUND = sys.argv[1], sys.argv[2], sys.argv[3]
EPISODE_STEPS = 500
SEED = 3000
COLLAPSE_Z = 0.30

if POLICY == "racingline":
    sys.path.insert(0, f"{THESIS}/2_python_scripts/f_racing_line")
    import car_track_racingline_vec_env as mod
    ENV_CLS = mod.CarTrackRacinglineVecEnv
    MODEL = f"{THESIS}/3_trained_models/car2_racingline_ppo_model.zip"
else:
    sys.path.insert(0, f"{THESIS}/2_python_scripts/e_curvature_lookahead")
    import car_track_vec_env as mod
    ENV_CLS = mod.CarTrackVecEnv
    MODEL = f"{THESIS}/3_trained_models/car2_vec_ppo_model.zip"

# The vec lineage was trained on the ORIGINAL point order; `reversed(points)` was added to
# car_track_vec_env.build_centerline() only afterwards (see common_metric_eval.py). Restore the
# training direction for it. Episodes are exactly EPISODE_STEPS long, so the env's own time limit
# is disabled to keep `dones` meaning "off-track" only.
TRAINED_REVERSED = POLICY == "racingline"
mod.MAX_EPISODE_STEPS = 10 ** 9
if not TRAINED_REVERSED:
    _orig_bc = mod.build_centerline

    def _training_order_centerline():
        pts, _ = _orig_bc()
        pts = list(reversed(pts))
        cum = [0.0]
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
        return pts, cum

    mod.build_centerline = _training_order_centerline

mod.get_simulation_app(headless=True)          # Kit must be up before any omni / pxr import

if GROUND == "plane":
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Gf, PhysicsSchemaTools, UsdPhysics, UsdShade
    _orig_open = stage_utils.open_stage

    def open_with_plane(path):
        ok = _orig_open(path)
        stage = stage_utils.get_current_stage()
        stage.RemovePrim("/World/GroundPlane")
        PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundPlane", "Z", 2000.0,
                                          Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5))
        mat = UsdShade.Material.Define(stage, "/World/GroundPhysicsMaterial")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(0.5)
        api.CreateDynamicFrictionAttr(0.5)
        api.CreateRestitutionAttr(0.8)
        geom = stage.GetPrimAtPath("/World/GroundPlane/geom")
        UsdShade.MaterialBindingAPI.Apply(geom).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        return ok

    stage_utils.open_stage = open_with_plane

from stable_baselines3 import PPO  # noqa: E402


def road_projector():
    a, r, n = 8.0, 10.0, 10
    corners = [((a, a), 90, 0), ((a, -a), 0, -90), ((-a, -a), -90, -180), ((-a, a), -180, -270)]
    pts = []
    for c, s0, s1 in corners:
        for k in range(n + 1):
            t = math.radians(s0 + (s1 - s0) * k / n)
            pts.append((c[0] + r * math.cos(t), c[1] + r * math.sin(t)))
    pts = [(x, y + a + r) for x, y in pts]
    if TRAINED_REVERSED:
        pts = list(reversed(pts))
    cum = [0.0]
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))

    def project(x, y):
        best = (float("inf"), 0, 0.0)
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            dx, dy = x1 - x0, y1 - y0
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)))
            d2 = (x - x0 - t * dx) ** 2 + (y - y0 - t * dy) ** 2
            if d2 < best[0]:
                best = (d2, i, t)
        _, i, t = best
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        return cum[i] + t * math.hypot(x1 - x0, y1 - y0)
    return project, cum[-1]


def main():
    env = ENV_CLS(num_envs=mod.NUM_ENVS, headless=True)
    model = PPO.load(MODEL, device="cpu")
    project, L = road_projector()
    n = env.num_envs
    env.seed(SEED)
    obs = env.reset()
    pos, _ = env._car.get_world_poses()
    last_s = [project(pos[i][0] - env._offsets[i, 0], pos[i][1] - env._offsets[i, 1]) for i in range(n)]
    ep_reward = np.zeros(n)
    dist = np.zeros(n)
    collapse = np.zeros(n, dtype=int)
    first_collapse = [None] * n
    thr = [[] for _ in range(n)]
    steer = [[] for _ in range(n)]
    ended = [None] * n                      # step at which an off-track termination ended the episode
    for step in range(EPISODE_STEPS):
        actions, _ = model.predict(obs, deterministic=(MODE == "det"))
        obs, rewards, dones, _ = env.step(actions)
        pos, _ = env._car.get_world_poses()
        for i in range(n):
            if ended[i] is not None:
                continue
            ep_reward[i] += rewards[i]
            a = np.clip(actions[i], -1, 1)
            thr[i].append(float(a[0]))
            steer[i].append(float(abs(a[1])))
            if dones[i]:
                ended[i] = step + 1
                continue
            s = project(pos[i][0] - env._offsets[i, 0], pos[i][1] - env._offsets[i, 1])
            ds = s - last_s[i]
            ds = ds - L if ds > L / 2 else ds + L if ds < -L / 2 else ds
            dist[i] += ds
            last_s[i] = s
            if pos[i][2] < COLLAPSE_Z:
                collapse[i] += 1
                if first_collapse[i] is None:
                    first_collapse[i] = step + 1
    clones = [{"episode_reward": round(float(ep_reward[i]), 2), "distance_m": round(float(dist[i]), 2),
               "off_track_at_step": ended[i], "collapse_steps": int(collapse[i]),
               "first_collapse_step": first_collapse[i],
               "mean_throttle": round(float(np.mean(thr[i])), 3), "mean_abs_steer": round(float(np.mean(steer[i])), 3)}
              for i in range(n)]
    summary = {
        "policy": POLICY, "mode": MODE, "ground": GROUND, "episode_steps": EPISODE_STEPS,
        "mean_episode_reward": round(float(ep_reward.mean()), 2),
        "full_length_episodes": sum(1 for e in ended if e is None),
        "mean_distance_m": round(float(dist.mean()), 2),
        "mean_speed_mps": round(float(dist.mean() / (EPISODE_STEPS * 4 / 60.0)), 3),
        "clones_with_collapse": int((collapse > 0).sum()),
        "mean_collapse_share": round(float(collapse.mean() / EPISODE_STEPS), 3),
        "mean_throttle": round(float(np.mean([c["mean_throttle"] for c in clones])), 3),
    }
    out = f"{OUT_DIR}/collapse_diag_{POLICY}_{MODE}_{GROUND}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "clones": clones}, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
