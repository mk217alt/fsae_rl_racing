"""Lap telemetry of the final self-play policy (100M steps) on the stadium track, in the style of Formula 1 timing:
both cars drive N laps from the grid of unknown_track.usd (Car A outside, Car B inside of the first corner), and
every control step (15 Hz) records, per car:

    speed, throttle command, measured longitudinal and lateral acceleration,
    steering command and the measured front-wheel steering angle, position, lap, sector, gap to the other car.

Timing: the timing line is the grid (both cars start level on it, as in lap_time_benchmark.py). Each lap is split
into three sectors of equal length along the centerline (a third of the 133.7 m lap each). Lap and sector times are
interpolated between control steps, so they are finer than the 1/15 s control step.

Same driving loop as lap_time_benchmark.py: shared policy, deterministic, headless, no recovery net. The start
positions are NOT swapped at random, so the run is repeatable; pass "swap" to exchange them. "a_ahead=2" moves
Car A 2 m forward in its lane (a controlled check of which car follows); the timing line stays at Car B's spot.

Usage: python.bat lap_telemetry.py [n_laps] [swap] [a_ahead=<metres>]
Output: 4_results/lap_telemetry_100M[_swapped][_a_ahead<m>m].csv (one row per car and control step)
        4_results/lap_telemetry_100M[_swapped][_a_ahead<m>m].json (lap and sector times, per-sector statistics,
        track geometry)
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import csv
import json
import math
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from stable_baselines3 import PPO

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

sys.path.insert(0, _os.path.join(ROOT, "2_python_scripts", "g_two_car_racing"))
from car_unknown_track_env import build_centerline, wrap_to_pi

CAR_USD = f"{ROOT}/1_simulation_scenes/unknown_track.usd"
MODEL_PATH = f"{ROOT}/3_trained_models/car_race_ppo_model"

N_LAPS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
SWAP = "swap" in sys.argv[1:]
A_AHEAD = next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("a_ahead=")), 0.0)
OUT_BASE = (f"{ROOT}/4_results/lap_telemetry_100M" + ("_swapped" if SWAP else "")
            + (f"_a_ahead{A_AHEAD:g}m" if A_AHEAD else ""))

PHYSICS_DT = 1.0 / 60.0
ACTION_REPEAT = 4
CONTROL_DT = PHYSICS_DT * ACTION_REPEAT
MAX_CONTROL_STEPS = 50000
N_SECTORS = 3
FULL_THROTTLE = 0.99

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0

open_stage(CAR_USD)
world = World(stage_units_in_meters=1.0)
world.reset()

car_a = Articulation(prim_paths_expr="/World/CarA/chassis")
car_a.initialize()
car_b = Articulation(prim_paths_expr="/World/CarB/chassis")
car_b.initialize()
cars = [car_a, car_b]
names = ["Car A", "Car B"]

if SWAP:
    pos_a, orient_a = car_a.get_world_poses()
    pos_b, orient_b = car_b.get_world_poses()
    car_a.set_world_poses(positions=pos_b, orientations=orient_b)
    car_b.set_world_poses(positions=pos_a, orientations=orient_a)
    car_a.set_velocities(np.zeros((1, 6)))
    car_b.set_velocities(np.zeros((1, 6)))
    for _ in range(2):
        world.step(render=False)

steer_idx = [[car.dof_names.index(j) for j in STEER_JOINTS] for car in cars]
policy = PPO.load(MODEL_PATH, device="cpu")
num_timesteps = int(policy.num_timesteps)
points, cumulative = build_centerline()
n_segments = len(points)
track_length = cumulative[-1]
sector_length = track_length / N_SECTORS


def nearest_segment(x, y):
    best_i, best_d2, best_t = 0, float("inf"), 0.0
    for i in range(n_segments):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n_segments]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        t = ((x - x0) * dx + (y - y0) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        px, py = x0 + t * dx, y0 + t * dy
        d2 = (x - px) ** 2 + (y - py) ** 2
        if d2 < best_d2:
            best_d2, best_i, best_t = d2, i, t
    return best_i, best_t, math.sqrt(best_d2)


def tangent_at_progress(s):
    s = s % track_length
    for i in range(n_segments):
        if cumulative[i] <= s <= cumulative[i + 1]:
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n_segments]
            return math.atan2(y1 - y0, x1 - x0)
    x0, y0 = points[-1]
    x1, y1 = points[0]
    return math.atan2(y1 - y0, x1 - x0)


def own_state(car):
    """Same observation as lap_time_benchmark.py, plus the quantities logged as telemetry."""
    positions, orientations = car.get_world_poses()
    pos = positions[0]
    w, x, y, z = orientations[0]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    i, t, dist = nearest_segment(pos[0], pos[1])
    x0, y0 = points[i]
    x1, y1 = points[(i + 1) % n_segments]
    dx, dy = x1 - x0, y1 - y0
    seg_len = math.hypot(dx, dy)
    tangent_angle = math.atan2(dy, dx)
    px, py = x0 + t * dx, y0 + t * dy
    side = math.copysign(1.0, dx * (pos[1] - py) - dy * (pos[0] - px))
    lateral_offset = dist * side
    progress = cumulative[i] + t * seg_len
    heading_error = wrap_to_pi(yaw - tangent_angle)

    tangent_ahead = tangent_at_progress(progress + LOOKAHEAD_DIST)
    curvature_ahead = wrap_to_pi(tangent_ahead - tangent_angle)

    lin_vel = car.get_linear_velocities()[0]
    ang_vel = car.get_angular_velocities()[0]
    forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
    yaw_rate = ang_vel[2]

    partial_obs = np.array(
        [lateral_offset, math.sin(heading_error), math.cos(heading_error),
         forward_speed, yaw_rate, math.sin(curvature_ahead), math.cos(curvature_ahead)],
        dtype=np.float32,
    )
    extra = {"x": float(pos[0]), "y": float(pos[1]), "lateral": float(lateral_offset),
             "speed": float(forward_speed), "yaw_rate": float(yaw_rate)}
    return partial_obs, progress, extra


def crossing_time(t0, d0, t1, d1, target):
    """time at which the travelled distance reached target, linear between two control steps"""
    return t0 + (target - d0) / (d1 - d0) * (t1 - t0) if d1 != d0 else t1


if A_AHEAD:
    # controlled check: Car A starts A_AHEAD metres further along the track than Car B, in its own lane
    pos_a, orient_a = car_a.get_world_poses()
    i, _, _ = nearest_segment(pos_a[0][0], pos_a[0][1])
    x0, y0 = points[i]
    x1, y1 = points[(i + 1) % n_segments]
    heading = math.atan2(y1 - y0, x1 - x0)
    pos_a = np.array(pos_a, dtype=float)
    pos_a[0][0] += A_AHEAD * math.cos(heading)
    pos_a[0][1] += A_AHEAD * math.sin(heading)
    car_a.set_world_poses(positions=pos_a, orientations=orient_a)
    car_a.set_velocities(np.zeros((1, 6)))

for _ in range(2):
    world.step(render=False)

rows = []
last_progress = [None, None]
distance = [A_AHEAD, 0.0]      # distance travelled along the centerline since the timing line (Car B's grid spot)
prev = [None, None]            # (time, distance, speed) of the previous control step
marks = [[], []]               # times at which each sector boundary was crossed (every third one ends a lap)
step = 0

while min(len(m) for m in marks) < N_LAPS * N_SECTORS and step < MAX_CONTROL_STEPS:
    now = step * CONTROL_DT
    states = [own_state(car) for car in cars]
    for idx in range(2):
        progress = states[idx][1]
        if last_progress[idx] is not None:
            delta = progress - last_progress[idx]
            if delta > track_length / 2.0:
                delta -= track_length
            elif delta < -track_length / 2.0:
                delta += track_length
            distance[idx] += delta
        last_progress[idx] = progress

    # observations and actions exactly as in lap_time_benchmark.py
    gap_ab = states[1][1] - states[0][1]
    if gap_ab > track_length / 2.0:
        gap_ab -= track_length
    elif gap_ab < -track_length / 2.0:
        gap_ab += track_length
    gap_ab = float(np.clip(gap_ab, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
    opp_gap = [gap_ab, -gap_ab]
    speeds = [states[i][2]["speed"] for i in range(2)]
    opp_speed_gap = [speeds[1] - speeds[0], speeds[0] - speeds[1]]
    lateral_a, lateral_b = states[0][0][0], states[1][0][0]
    opp_lat_gap = [lateral_b - lateral_a, lateral_a - lateral_b]

    actions = []
    for idx in range(2):
        obs = np.concatenate([states[idx][0], [opp_gap[idx], opp_lat_gap[idx], opp_speed_gap[idx]]])
        action, _ = policy.predict(obs, deterministic=True)
        actions.append(np.clip(action, -1.0, 1.0))

    for idx, car in enumerate(cars):
        ex = states[idx][2]
        d = distance[idx]
        if prev[idx] is not None:
            t0, d0, v0 = prev[idx]
            while len(marks[idx]) < N_LAPS * N_SECTORS and d >= (len(marks[idx]) + 1) * sector_length:
                marks[idx].append(crossing_time(t0, d0, now, d, (len(marks[idx]) + 1) * sector_length))
            accel = (ex["speed"] - v0) / CONTROL_DT
        else:
            accel = 0.0
        prev[idx] = (now, d, ex["speed"])
        steer_deg = float(np.degrees(np.mean(car.get_joint_positions(joint_indices=steer_idx[idx])[0])))
        lap = min(int(d // track_length) + 1, N_LAPS + 1)
        lap_dist = d - (lap - 1) * track_length
        rows.append([
            names[idx], step, round(now, 4), lap, round(lap_dist, 3), min(int(lap_dist // sector_length) + 1, 3),
            round(ex["x"], 3), round(ex["y"], 3), round(ex["lateral"], 3), round(ex["speed"], 4),
            round(accel, 4), round(ex["speed"] * ex["yaw_rate"], 4),
            round(float(actions[idx][0]), 4), round(float(actions[idx][1]), 4), round(steer_deg, 3),
            round(d - distance[1 - idx], 3),
        ])

    for idx, car in enumerate(cars):
        car.set_joint_velocity_targets(
            np.full((1, len(DRIVE_JOINTS)), float(actions[idx][0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
        )
        car.set_joint_position_targets(
            np.full((1, len(STEER_JOINTS)), float(actions[idx][1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
        )
    for _ in range(ACTION_REPEAT):
        world.step(render=False)
    step += 1

COLUMNS = ["car", "step", "t_s", "lap", "lap_dist_m", "sector", "x_m", "y_m", "lateral_m", "speed_mps",
           "accel_long_mps2", "accel_lat_mps2", "throttle_cmd", "steer_cmd", "steer_deg", "gap_m"]
with open(OUT_BASE + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLUMNS)
    w.writerows(rows)


# ------------------------------------------------------------------ lap and sector statistics
def mean_std(values):
    return (round(float(np.mean(values)), 3), round(float(np.std(values)), 3)) if values else (None, None)


col = {c: i for i, c in enumerate(COLUMNS)}
report = {
    "policy": "3_trained_models/car_race_ppo_model.zip", "num_timesteps": num_timesteps,
    "track": "1_simulation_scenes/unknown_track.usd", "track_length_m": round(track_length, 3),
    "sector_length_m": round(sector_length, 3), "n_laps": N_LAPS, "start_positions_swapped": SWAP,
    "control_dt_s": round(CONTROL_DT, 6), "deterministic": True, "car_a_ahead_m": A_AHEAD,
    "timing": "timing line = the grid; three equal sectors along the centerline; lap and sector times "
              "interpolated between control steps; flying laps = laps 2..n",
    "grid": {"Car A": ("inside" if SWAP else "outside") + " of the first corner"
                      + (f", {A_AHEAD:g} m ahead" if A_AHEAD else ""),
             "Car B": ("outside" if SWAP else "inside") + " of the first corner"},
    "centerline_xy": [[round(x, 3), round(y, 3)] for x, y in points],
    "centerline_cumulative_m": [round(c, 3) for c in cumulative],
    "cars": {},
}
for idx, name in enumerate(names):
    times = [0.0] + marks[idx]
    sectors = [times[k + 1] - times[k] for k in range(len(times) - 1)]
    laps = []
    for lap in range(1, len(sectors) // N_SECTORS + 1):
        sec = sectors[(lap - 1) * N_SECTORS: lap * N_SECTORS]
        lap_rows = [r for r in rows if r[0] == name and r[col["lap"]] == lap]
        per_sector = []
        for s in range(1, N_SECTORS + 1):
            rs = [r for r in lap_rows if r[col["sector"]] == s]
            per_sector.append({
                "time_s": round(sec[s - 1], 3),
                "mean_speed_mps": round(float(np.mean([r[col["speed_mps"]] for r in rs])), 3),
                "min_speed_mps": round(float(np.min([r[col["speed_mps"]] for r in rs])), 3),
                "full_throttle_share": round(float(np.mean([r[col["throttle_cmd"]] >= FULL_THROTTLE for r in rs])), 3),
                "mean_abs_steer_deg": round(float(np.mean([abs(r[col["steer_deg"]]) for r in rs])), 3),
                "max_abs_steer_deg": round(float(np.max([abs(r[col["steer_deg"]]) for r in rs])), 3),
                "max_accel_mps2": round(float(np.max([r[col["accel_long_mps2"]] for r in rs])), 3),
                "min_accel_mps2": round(float(np.min([r[col["accel_long_mps2"]] for r in rs])), 3),
            })
        steer_cmd = [r[col["steer_cmd"]] for r in lap_rows]
        throttle_cmd = [r[col["throttle_cmd"]] for r in lap_rows]
        laps.append({
            "lap": lap, "time_s": round(sum(sec), 3), "sectors_s": [round(x, 3) for x in sec],
            "mean_speed_mps": round(float(np.mean([r[col["speed_mps"]] for r in lap_rows])), 3),
            "max_speed_mps": round(float(np.max([r[col["speed_mps"]] for r in lap_rows])), 3),
            "full_throttle_share": round(float(np.mean([x >= FULL_THROTTLE for x in throttle_cmd])), 3),
            "mean_abs_steer_deg": round(float(np.mean([abs(r[col["steer_deg"]]) for r in lap_rows])), 3),
            "mean_abs_steer_change": round(float(np.mean(np.abs(np.diff(steer_cmd)))), 4),
            "mean_abs_throttle_change": round(float(np.mean(np.abs(np.diff(throttle_cmd)))), 4),
            "gap_at_line_m": round(float(lap_rows[-1][col["gap_m"]]), 3) if lap_rows else None,
            "sector_detail": per_sector,
        })
    flying = [l for l in laps if l["lap"] >= 2]
    best_sectors = [min(l["sectors_s"][s] for l in flying) for s in range(N_SECTORS)] if flying else []
    report["cars"][name] = {
        "laps": laps,
        "flying_laps": {
            "n": len(flying),
            "lap_mean_s": mean_std([l["time_s"] for l in flying])[0],
            "lap_std_s": mean_std([l["time_s"] for l in flying])[1],
            "best_lap_s": min(l["time_s"] for l in flying) if flying else None,
            "sector_mean_s": [mean_std([l["sectors_s"][s] for l in flying])[0] for s in range(N_SECTORS)],
            "sector_std_s": [mean_std([l["sectors_s"][s] for l in flying])[1] for s in range(N_SECTORS)],
            "best_sectors_s": [round(x, 3) for x in best_sectors],
            "theoretical_best_s": round(sum(best_sectors), 3) if flying else None,
            "sector_mean_speed_mps": [mean_std([l["sector_detail"][s]["mean_speed_mps"] for l in flying])[0]
                                      for s in range(N_SECTORS)],
            "sector_full_throttle_share": [mean_std([l["sector_detail"][s]["full_throttle_share"] for l in flying])[0]
                                           for s in range(N_SECTORS)],
            "sector_mean_abs_steer_deg": [mean_std([l["sector_detail"][s]["mean_abs_steer_deg"] for l in flying])[0]
                                          for s in range(N_SECTORS)],
            "mean_speed_mps": mean_std([l["mean_speed_mps"] for l in flying])[0],
            "max_speed_mps": max(l["max_speed_mps"] for l in flying) if flying else None,
            "full_throttle_share": mean_std([l["full_throttle_share"] for l in flying])[0],
            "mean_abs_steer_change": mean_std([l["mean_abs_steer_change"] for l in flying])[0],
            "mean_abs_throttle_change": mean_std([l["mean_abs_throttle_change"] for l in flying])[0],
        },
    }
report["control_steps"] = step
report["hit_step_cap"] = step >= MAX_CONTROL_STEPS
with open(OUT_BASE + ".json", "w") as f:
    json.dump(report, f, indent=1)

print()
print(f"LAP TELEMETRY  policy {num_timesteps:,} steps, stadium track {track_length:.1f} m, "
      f"3 sectors of {sector_length:.1f} m, start positions swapped: {SWAP}")
for name in names:
    c = report["cars"][name]
    print(f"{name} ({report['grid'][name]})")
    for l in c["laps"]:
        s = l["sectors_s"]
        print(f"   lap {l['lap']:2d}  {l['time_s']:6.2f} s   S1 {s[0]:5.2f}  S2 {s[1]:5.2f}  S3 {s[2]:5.2f}   "
              f"gap at the line {l['gap_at_line_m']:+6.2f} m")
    fl = c["flying_laps"]
    sm, ss = fl["sector_mean_s"], fl["sector_std_s"]
    print(f"   flying laps: {fl['lap_mean_s']:.2f} +/- {fl['lap_std_s']:.2f} s   "
          f"S1 {sm[0]:.2f}+/-{ss[0]:.2f}  S2 {sm[1]:.2f}+/-{ss[1]:.2f}  S3 {sm[2]:.2f}+/-{ss[2]:.2f}   "
          f"best {fl['best_lap_s']:.2f} s, theoretical best {fl['theoretical_best_s']:.2f} s")
print(f"wrote {OUT_BASE}.csv and .json")
print("TELEMETRY DONE")
sys.stdout.flush()
world.stop()
simulation_app.close()
