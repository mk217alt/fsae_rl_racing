"""Classical-control baseline for comparison against the RL policies:
drives car2 around the SAME track/environment the pure-RL baseline trained
on (car_track_env.py, simple_car.usd), using pure pursuit for steering and
a curvature-aware speed target for throttle - a fixed geometric formula
computed fresh every step, no learning, no training data, nothing that can
get "stuck in a bad basin" the way the behavioral-cloning attempts did.

Pure pursuit: aim at a point a fixed lookahead distance ahead on the
centerline, compute the steering angle that would arc the car onto that
point (bicycle-model curvature formula), same idea used on real
Formula-Student-style path-tracking stacks before anything learned enters
the picture.

Logs episode reward/length in exactly the same format as the RL training
monitors, so results are directly comparable to car2_ppo_model.zip's
converged numbers (105-113 reward, ~99% full-length episodes).
"""


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Single_Car_Baseline"))
import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

from car_track_env import (
    ACTION_REPEAT,
    CAR_ROOT,
    CAR_USD,
    CHASSIS_Z,
    DRIVE_JOINTS,
    LATERAL_PENALTY_WEIGHT,
    MAX_EPISODE_STEPS,
    MAX_STEER_ANGLE,
    MAX_WHEEL_SPEED,
    OFF_TRACK_MARGIN,
    OFF_TRACK_PENALTY,
    STEER_JOINTS,
    THROTTLE_BONUS_WEIGHT,
    build_centerline,
    wrap_to_pi,
)

N_EPISODES = 40
LOOKAHEAD_DIST = 4.0  # meters ahead on the centerline to aim at
WHEELBASE = 1.6  # front-to-rear axle distance = 2 * WHEEL_X_OFFSET (0.8) from create_simple_car.py
TARGET_SPEED = 5.0  # m/s on straights
CORNER_SLOWDOWN = 3.0  # rad^-1, how much upcoming curvature reduces target speed
THROTTLE_KP = 0.6
STEER_SIGN = 1.0  # flip to -1.0 if a quick test shows the car steering the wrong way

open_stage(CAR_USD)
world = World(stage_units_in_meters=1.0)
world.reset()

car = Articulation(prim_paths_expr=CAR_ROOT)
car.initialize()

points, cumulative = build_centerline()
n_segments = len(points)
track_length = cumulative[-1]
rng = np.random.default_rng()


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


def point_and_tangent_at_progress(s):
    s = s % track_length
    for i in range(n_segments):
        if cumulative[i] <= s <= cumulative[i + 1]:
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n_segments]
            seg_len = cumulative[i + 1] - cumulative[i]
            t = (s - cumulative[i]) / seg_len if seg_len > 1e-9 else 0.0
            px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            return (px, py), math.atan2(y1 - y0, x1 - x0)
    return points[-1], 0.0


def get_state():
    positions, orientations = car.get_world_poses()
    pos = positions[0]
    w, x, y, z = orientations[0]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    i, t, dist = nearest_segment(pos[0], pos[1])
    x0, y0 = points[i]
    x1, y1 = points[(i + 1) % n_segments]
    seg_len = math.hypot(x1 - x0, y1 - y0)
    tangent_angle = math.atan2(y1 - y0, x1 - x0)
    px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    side = math.copysign(1.0, (x1 - x0) * (pos[1] - py) - (y1 - y0) * (pos[0] - px))
    lateral_offset = dist * side
    progress = cumulative[i] + t * seg_len
    lin_vel = car.get_linear_velocities()[0]
    forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
    return pos, yaw, progress, lateral_offset, forward_speed, tangent_angle


def pure_pursuit_action(pos, yaw, progress, forward_speed, tangent_angle):
    target_pt, target_tangent = point_and_tangent_at_progress(progress + LOOKAHEAD_DIST)
    dx, dy = target_pt[0] - pos[0], target_pt[1] - pos[1]
    target_heading = math.atan2(dy, dx)
    alpha = wrap_to_pi(target_heading - yaw)
    curvature = 2.0 * math.sin(alpha) / LOOKAHEAD_DIST
    steer_angle = math.atan(WHEELBASE * curvature) * STEER_SIGN
    steer = float(np.clip(steer_angle / MAX_STEER_ANGLE, -1.0, 1.0))

    upcoming_turn = abs(wrap_to_pi(target_tangent - tangent_angle))
    target_speed = TARGET_SPEED / (1.0 + CORNER_SLOWDOWN * upcoming_turn)
    throttle = float(np.clip(THROTTLE_KP * (target_speed - forward_speed), -1.0, 1.0))
    return throttle, steer


def reset_episode():
    seg = int(rng.integers(0, n_segments))
    x0, y0 = points[seg]
    x1, y1 = points[(seg + 1) % n_segments]
    tangent_angle = math.atan2(y1 - y0, x1 - x0)
    px, py = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    yaw = tangent_angle + float(rng.uniform(-0.1, 0.1))
    half_yaw = yaw / 2.0
    car.set_world_poses(
        positions=np.array([[px, py, CHASSIS_Z]]),
        orientations=np.array([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]]),
    )
    car.set_velocities(np.zeros((1, 6)))
    car.set_joint_velocities(np.zeros((1, car.num_dof)))
    car.set_joint_positions(np.zeros((1, car.num_dof)))
    for _ in range(2):
        world.step(render=False)


world.play()

results = []
for ep in range(N_EPISODES):
    reset_episode()
    pos, yaw, progress, lateral_offset, forward_speed, tangent_angle = get_state()
    last_progress = progress
    total_reward = 0.0
    step_count = 0

    for step in range(MAX_EPISODE_STEPS):
        throttle, steer = pure_pursuit_action(pos, yaw, progress, forward_speed, tangent_angle)
        car.set_joint_velocity_targets(np.full((1, len(DRIVE_JOINTS)), throttle * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
        car.set_joint_position_targets(np.full((1, len(STEER_JOINTS)), steer * MAX_STEER_ANGLE), joint_names=STEER_JOINTS)

        for _ in range(ACTION_REPEAT):
            world.step(render=False)

        pos, yaw, progress, lateral_offset, forward_speed, tangent_angle = get_state()
        delta = progress - last_progress
        if delta > track_length / 2.0:
            delta -= track_length
        elif delta < -track_length / 2.0:
            delta += track_length
        last_progress = progress

        reward = delta - LATERAL_PENALTY_WEIGHT * abs(lateral_offset) + THROTTLE_BONUS_WEIGHT * max(0.0, throttle)
        off_track = abs(lateral_offset) > OFF_TRACK_MARGIN
        if off_track:
            reward -= OFF_TRACK_PENALTY
        total_reward += reward
        step_count += 1

        if off_track:
            break

    results.append((total_reward, step_count))

rewards = [r for r, l in results]
lengths = [l for r, l in results]
n500 = sum(1 for r, l in results if l == MAX_EPISODE_STEPS)

lines = [f"episode {i}: reward={r:.2f} length={l}" for i, (r, l) in enumerate(results)]
lines.append(f"--- summary over {N_EPISODES} episodes ---")
lines.append(f"avg reward: {sum(rewards)/len(rewards):.2f}")
lines.append(f"avg length: {sum(lengths)/len(lengths):.2f}")
lines.append(f"max reward: {max(rewards):.2f}  min reward: {min(rewards):.2f}")
lines.append(f"full-length (500-step) episodes: {n500}/{N_EPISODES}")

with open("C:/Users/sanja/Desktop/thesis/pure_pursuit_baseline_log.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

with open("C:/Users/sanja/Desktop/thesis/pure_pursuit_baseline_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
