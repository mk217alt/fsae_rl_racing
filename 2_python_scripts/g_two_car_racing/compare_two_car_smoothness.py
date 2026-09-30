"""Compares action smoothness between the newtrack_smooth_run1 and
newtrack_smooth_run20 checkpoints of the two-car racing lineage, by running
each deterministically (no exploration noise) on the unknown track and
measuring average consecutive-action change - the direct metric the
smoothness reward term was designed to reduce. Same methodology as the
single-car baseline's compare_smoothness.py.

Run1 is the checkpoint the user confirmed as "no jitteriness" right after the
smoothness penalty was added; run20 is 19 more training rounds later, where
the user reported jitteriness might be back despite the reward term still
being present in every step.

Usage: python.bat compare_two_car_smoothness.py
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import math
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from stable_baselines3 import PPO

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

from car_unknown_track_env import build_centerline, wrap_to_pi

CAR_USD = f"{ROOT}/1_simulation_scenes/unknown_track.usd"
RESULTS_PATH = f"{ROOT}/4_results/compare_two_car_smoothness_results.txt"

CHECKPOINTS = {
    "smooth_run1": f"{ROOT}/3_trained_models/checkpoints/car_race_ppo_model_newtrack_smooth_run1",
    "smooth_run20": f"{ROOT}/3_trained_models/checkpoints/car_race_ppo_model_newtrack_smooth_run20",
}

N_CONTROL_STEPS = 3000
PHYSICS_DT = 1.0 / 60.0
ACTION_REPEAT = 4

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0

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


log(f"Two-car action-smoothness comparison, N_CONTROL_STEPS={N_CONTROL_STEPS}, track=unknown_track.usd")

results = {}
for label, path in CHECKPOINTS.items():
    open_stage(CAR_USD)
    world = World(stage_units_in_meters=1.0)
    world.reset()

    car_a = Articulation(prim_paths_expr="/World/CarA/chassis")
    car_a.initialize()
    car_b = Articulation(prim_paths_expr="/World/CarB/chassis")
    car_b.initialize()
    cars = [car_a, car_b]

    policy = PPO.load(path, device="cpu")

    for _ in range(2):
        world.step(render=False)

    deltas = [[], []]
    prev_action = [None, None]
    last_progress = [None, None]

    for _ in range(N_CONTROL_STEPS):
        states = [own_state(car_a), own_state(car_b)]
        progresses = [s[1] for s in states]
        if last_progress[0] is None:
            last_progress = list(progresses)

        gap_ab = progresses[1] - progresses[0]
        if gap_ab > track_length / 2.0:
            gap_ab -= track_length
        elif gap_ab < -track_length / 2.0:
            gap_ab += track_length
        gap_ab = float(np.clip(gap_ab, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
        opp_gap = [gap_ab, -gap_ab]

        lin_vels = [car.get_linear_velocities()[0] for car in cars]
        yaws = []
        for car in cars:
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

            if prev_action[idx] is not None:
                deltas[idx].append(float(abs(action[0] - prev_action[idx][0]) + abs(action[1] - prev_action[idx][1])))
            prev_action[idx] = action

            car.set_joint_velocity_targets(
                np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
            )
            car.set_joint_position_targets(
                np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
            )

        for _ in range(ACTION_REPEAT):
            world.step(render=False)

    all_deltas = deltas[0] + deltas[1]
    results[label] = {
        "mean_action_delta": float(np.mean(all_deltas)),
        "std_action_delta": float(np.std(all_deltas)),
        "car_a_mean": float(np.mean(deltas[0])),
        "car_b_mean": float(np.mean(deltas[1])),
        "n_steps_measured": len(all_deltas),
    }
    log(
        f"{label}: mean_delta={results[label]['mean_action_delta']:.4f} "
        f"std={results[label]['std_action_delta']:.4f} "
        f"car_a_mean={results[label]['car_a_mean']:.4f} car_b_mean={results[label]['car_b_mean']:.4f} "
        f"n_steps={results[label]['n_steps_measured']}"
    )

    world.stop()

run1 = results["smooth_run1"]["mean_action_delta"]
run20 = results["smooth_run20"]["mean_action_delta"]
change = (run20 - run1) / run1 * 100
log(f"Smoothness change run1->run20: {change:+.1f}% ({'more jittery' if change > 0 else 'smoother'})")

out.close()
sys.stdout.flush()
simulation_app.close()
