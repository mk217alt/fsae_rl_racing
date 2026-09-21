"""10-lap timing benchmark on the unknown track (unknown_track.usd, which
already has a checkered white/black start/finish line built at the track's
own start point - build_unknown_track_deploy.py - the same point used as
progress=0 for lap counting here). Both cars race simultaneously (self-play,
shared policy, deterministic), same driving logic as run_unknown_track_demo.py
but headless and running to a fixed lap count instead of indefinitely, with
no recovery net (matching the current raw-policy setup) - a car that
genuinely gets stuck just stops accumulating laps rather than being reset.

Lap detection: each car's per-step forward progress delta (wrap-corrected,
same convention as the training env) is accumulated into a running total;
a lap completes each time that total crosses another whole multiple of the
track length. Sim time uses Isaac Sim's default physics step (1/60s per
physics step; verified no script in this project overrides it) times
ACTION_REPEAT per control step.

Usage: python.bat lap_time_benchmark.py [n_laps]
"""

import math
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from stable_baselines3 import PPO

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

from car_unknown_track_env import CHASSIS_Z, build_centerline, wrap_to_pi

CAR_USD = "C:/Users/sanja/Desktop/thesis/Two_Car_Self_Play_Racing/unknown_track.usd"
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Two_Car_Self_Play_Racing/car_race_ppo_model"
RESULTS_PATH = "C:/Users/sanja/Desktop/thesis/Two_Car_Self_Play_Racing/lap_time_benchmark_results.txt"

N_LAPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PHYSICS_DT = 1.0 / 60.0
ACTION_REPEAT = 4
CONTROL_DT = PHYSICS_DT * ACTION_REPEAT
MAX_CONTROL_STEPS = 50000  # safety cap in case a car gets stuck and never finishes

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

policy = PPO.load(MODEL_PATH, device="cpu")
points, cumulative = build_centerline()
n_segments = len(points)
track_length = cumulative[-1]


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
    return partial_obs, progress


out = open(RESULTS_PATH, "w")


def log(msg):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()


log(f"10-lap timing benchmark, checkpoint={MODEL_PATH}.zip, track=unknown_track.usd, N_LAPS={N_LAPS}")

last_progress = [None, None]
cumulative_distance = [0.0, 0.0]
laps_completed = [0, 0]
last_lap_time = [0.0, 0.0]
lap_times = [[], []]
control_step_count = 0

for _ in range(2):
    world.step(render=False)

while min(laps_completed) < N_LAPS and control_step_count < MAX_CONTROL_STEPS:
    states = [own_state(car_a), own_state(car_b)]
    progresses = [s[1] for s in states]

    if last_progress[0] is None:
        last_progress = list(progresses)

    for idx in range(2):
        delta = progresses[idx] - last_progress[idx]
        if delta > track_length / 2.0:
            delta -= track_length
        elif delta < -track_length / 2.0:
            delta += track_length
        cumulative_distance[idx] += delta
        last_progress[idx] = progresses[idx]

        new_lap_count = int(cumulative_distance[idx] // track_length)
        while laps_completed[idx] < new_lap_count and laps_completed[idx] < N_LAPS:
            laps_completed[idx] += 1
            now = control_step_count * CONTROL_DT
            lap_time = now - last_lap_time[idx]
            last_lap_time[idx] = now
            lap_times[idx].append(lap_time)
            log(f"{names[idx]} lap {laps_completed[idx]}: {lap_time:.2f}s (t={now:.1f}s)")

    gap_ab = progresses[1] - progresses[0]
    if gap_ab > track_length / 2.0:
        gap_ab -= track_length
    elif gap_ab < -track_length / 2.0:
        gap_ab += track_length
    gap_ab = float(np.clip(gap_ab, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
    opp_gap = [gap_ab, -gap_ab]

    lin_vels = [car.get_linear_velocities()[0] for car in cars]
    yaws = []
    for idx, car in enumerate(cars):
        _, orientations = car.get_world_poses()
        w, x, y, z = orientations[0]
        yaws.append(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    speeds = [lin_vels[i][0] * math.cos(yaws[i]) + lin_vels[i][1] * math.sin(yaws[i]) for i in range(2)]
    opp_speed_gap = [speeds[1] - speeds[0], speeds[0] - speeds[1]]

    lateral_a = states[0][0][0]
    lateral_b = states[1][0][0]
    opp_lat_gap = [lateral_b - lateral_a, lateral_a - lateral_b]

    for idx, car in enumerate(cars):
        partial_obs = states[idx][0]
        obs = np.concatenate([partial_obs, [opp_gap[idx], opp_lat_gap[idx], opp_speed_gap[idx]]])
        action, _ = policy.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)
        car.set_joint_velocity_targets(
            np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
        )
        car.set_joint_position_targets(
            np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
        )

    for _ in range(ACTION_REPEAT):
        world.step(render=False)
    control_step_count += 1

log("")
if control_step_count >= MAX_CONTROL_STEPS:
    log(f"WARNING: hit the {MAX_CONTROL_STEPS}-control-step safety cap before both cars finished {N_LAPS} laps")
    log(f"laps completed: Car A={laps_completed[0]}, Car B={laps_completed[1]}")

for idx in range(2):
    times = lap_times[idx]
    if not times:
        log(f"{names[idx]}: no completed laps")
        continue
    log(
        f"{names[idx]}: {len(times)} laps, total={sum(times):.2f}s, "
        f"mean={np.mean(times):.2f}s, best={min(times):.2f}s, worst={max(times):.2f}s"
    )

if lap_times[0] and lap_times[1]:
    n_common = min(len(lap_times[0]), len(lap_times[1]))
    a_total = sum(lap_times[0][:n_common])
    b_total = sum(lap_times[1][:n_common])
    faster = "Car A" if a_total < b_total else "Car B"
    gap = abs(a_total - b_total)
    log(f"Head-to-head over {n_common} common laps: {faster} faster by {gap:.2f}s total ({gap / n_common:.2f}s/lap avg)")

out.close()
sys.stdout.flush()
world.stop()
simulation_app.close()
