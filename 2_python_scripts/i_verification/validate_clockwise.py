"""Headless validation that car2_vec_ppo_model.zip still drives correctly
after reversing build_centerline()'s point order (clockwise direction).

Reasoning for why this should transfer without retraining: every piece of
the observation (heading_error, lateral_offset side, curvature_ahead) is
computed relative to the LOCAL tangent/centerline, never against an
absolute world direction - reversing the polyline's traversal order flips
which physical direction counts as "forward" and correspondingly flips the
sign of lateral_offset's side convention, but the policy's learned
response ("low heading error + forward speed = reward", "correct back
toward zero lateral_offset") stays internally consistent either way. This
script checks that reasoning empirically rather than trusting it blindly:
load the policy, drive it in the actual deploy world for a few hundred
steps under the new centerline convention, and confirm it makes real
forward progress with reward in a normal, healthy range - not driving
backward against its own training or immediately falling off.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e_curvature_lookahead"))
import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from stable_baselines3 import PPO

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

from car_track_vec_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi

CAR_USD = f"{ROOT}/1_simulation_scenes/vec_deploy_car2.usd"
MODEL_PATH = f"{ROOT}/3_trained_models/car2_vec_ppo_model"
DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
LATERAL_PENALTY_WEIGHT = 0.6
OFF_TRACK_PENALTY = 30.0
THROTTLE_BONUS_WEIGHT = 0.25

open_stage(CAR_USD)
world = World(stage_units_in_meters=1.0)
world.reset()

car2 = Articulation(prim_paths_expr="/World/Car2/chassis")
car2.initialize()

policy = PPO.load(MODEL_PATH, device="cpu")
points, cumulative = build_centerline()
n_segments = len(points)
track_length = cumulative[-1]


def spawn_at_segment(seg):
    x0, y0 = points[seg]
    x1, y1 = points[(seg + 1) % n_segments]
    tangent0 = math.atan2(y1 - y0, x1 - x0)
    half_yaw = tangent0 / 2.0
    car2.set_world_poses(
        positions=np.array([[x0, y0, CHASSIS_Z]]),
        orientations=np.array([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]]),
    )
    car2.set_velocities(np.zeros((1, 6)))
    car2.set_joint_velocities(np.zeros((1, car2.num_dof)))
    car2.set_joint_positions(np.zeros((1, car2.num_dof)))


world.play()
spawn_at_segment(0)
for _ in range(3):
    world.step(render=False)


def nearest_segment(x, y):
    best_i, best_d2, best_t = 0, float("inf"), 0.0
    for i in range(n_segments):
        px0, py0 = points[i]
        px1, py1 = points[(i + 1) % n_segments]
        dx, dy = px1 - px0, py1 - py0
        seg_len2 = dx * dx + dy * dy
        t = ((x - px0) * dx + (y - py0) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        cx, cy = px0 + t * dx, py0 + t * dy
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        if d2 < best_d2:
            best_d2, best_i, best_t = d2, i, t
    return best_i, best_t, math.sqrt(best_d2)


def tangent_at_progress(s):
    s = s % track_length
    for i in range(n_segments):
        if cumulative[i] <= s <= cumulative[i + 1]:
            px0, py0 = points[i]
            px1, py1 = points[(i + 1) % n_segments]
            return math.atan2(py1 - py0, px1 - px0)
    px0, py0 = points[-1]
    px1, py1 = points[0]
    return math.atan2(py1 - py0, px1 - px0)


def observe():
    positions, orientations = car2.get_world_poses()
    pos = positions[0]
    w, x, y, z = orientations[0]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    i, t, dist = nearest_segment(pos[0], pos[1])
    x0_, y0_ = points[i]
    x1_, y1_ = points[(i + 1) % n_segments]
    dx, dy = x1_ - x0_, y1_ - y0_
    seg_len = math.hypot(dx, dy)
    tangent_angle = math.atan2(dy, dx)
    px_, py_ = x0_ + t * dx, y0_ + t * dy
    side = math.copysign(1.0, dx * (pos[1] - py_) - dy * (pos[0] - px_))
    lateral_offset = dist * side
    progress = cumulative[i] + t * seg_len
    heading_error = wrap_to_pi(yaw - tangent_angle)

    tangent_ahead = tangent_at_progress(progress + LOOKAHEAD_DIST)
    curvature_ahead = wrap_to_pi(tangent_ahead - tangent_angle)

    lin_vel = car2.get_linear_velocities()[0]
    ang_vel = car2.get_angular_velocities()[0]
    forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
    yaw_rate = ang_vel[2]

    obs = np.array(
        [lateral_offset, math.sin(heading_error), math.cos(heading_error),
         forward_speed, yaw_rate, math.sin(curvature_ahead), math.cos(curvature_ahead)],
        dtype=np.float32,
    )
    return obs, progress, lateral_offset


lines = []
start_segments = [0, 8, 16, 24, 32]  # spread across the loop
for trial, seg in enumerate(start_segments):
    spawn_at_segment(seg)
    for _ in range(3):
        world.step(render=False)

    last_progress = None
    total_reward = 0.0
    final_step = 0
    for step in range(400):
        obs, progress, lateral_offset = observe()
        if last_progress is None:
            last_progress = progress
        action, _ = policy.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)
        car2.set_joint_velocity_targets(np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
        car2.set_joint_position_targets(np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS)

        for _ in range(4):
            world.step(render=False)

        obs2, progress2, lateral_offset2 = observe()
        delta = progress2 - last_progress
        if delta > track_length / 2.0:
            delta -= track_length
        elif delta < -track_length / 2.0:
            delta += track_length
        last_progress = progress2

        reward = delta - LATERAL_PENALTY_WEIGHT * abs(lateral_offset2) + THROTTLE_BONUS_WEIGHT * max(0.0, float(action[0]))
        off_track = abs(lateral_offset2) > OFF_TRACK_MARGIN
        if off_track:
            reward -= OFF_TRACK_PENALTY
        total_reward += reward
        final_step = step

        if off_track:
            break

    lines.append(f"trial {trial} (start seg {seg}): survived {final_step+1} steps, reward={total_reward:.2f}, off_track={off_track if final_step < 399 else False}")

with open(f"{ROOT}/run_output/validate_clockwise_log.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

with open(f"{ROOT}/run_output/validate_clockwise_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
