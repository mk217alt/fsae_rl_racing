"""Live demo of the vectorized+curvature-lookahead car2 policy
(car2_vec_ppo_model.zip), deployed on vec_deploy_car2.usd - a dedicated
single-track stage built by build_vec_deploy_stage.py with NO lane offset,
matching exactly what car_track_vec_env.py trained against.

Earlier version of this script reused the original simple_car.usd (car2
lane-shifted +3m from track center, a convention specific to the OLD
single-env policy). That mismatch meant this policy's observations (and
worse, its off-track recovery threshold) were anchored to the wrong
reference line, letting car2 drift ~7m from true center - well outside the
actual curbs - before recovery would even consider firing. Deploying on a
stage with the same (unshifted) centerline the policy actually trained on
fixes that at the root, rather than patching the symptom.

Controls (car 1 only):
    W / Up    - drive forward
    S / Down  - drive backward
    A / Left  - steer left
    D / Right - steer right
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import carb.input
import omni.appwindow
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
RECOVERY_MARGIN = OFF_TRACK_MARGIN
LOOKAHEAD_DIST = 5.0  # must match car_track_vec_env.py's training-time value

open_stage(CAR_USD)

settings = carb.settings.get_settings()
settings.set_bool("/physics/updateToUsd", True)
settings.set_bool("/app/useFabricSceneDelegate", False)
settings.set_bool("/physics/fabricUpdateTransformations", False)

world = World(stage_units_in_meters=1.0)
world.reset()

car1 = Articulation(prim_paths_expr="/World/Car1/chassis")
car1.initialize()
car2 = Articulation(prim_paths_expr="/World/Car2/chassis")
car2.initialize()

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


def car2_observation():
    positions, orientations = car2.get_world_poses()
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

    lin_vel = car2.get_linear_velocities()[0]
    ang_vel = car2.get_angular_velocities()[0]
    forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
    yaw_rate = ang_vel[2]

    obs = np.array(
        [lateral_offset, math.sin(heading_error), math.cos(heading_error),
         forward_speed, yaw_rate, math.sin(curvature_ahead), math.cos(curvature_ahead)],
        dtype=np.float32,
    )
    up_z = 1.0 - 2.0 * (x * x + y * y)  # world-Z component of the car's own up axis
    height_error = abs(pos[2] - CHASSIS_Z)
    return obs, lateral_offset, tangent_angle, px, py, up_z, height_error, progress


def recover_car2_onto_track(tangent_angle, px, py):
    half_yaw = tangent_angle / 2.0
    orientation = np.array([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
    position = np.array([[px, py, CHASSIS_Z]])
    car2.set_world_poses(positions=position, orientations=orientation)
    car2.set_velocities(np.zeros((1, 6)))
    car2.set_joint_velocities(np.zeros((1, car2.num_dof)))
    car2.set_joint_positions(np.zeros((1, car2.num_dof)))


# --- car 1: keyboard control (same as drive_car.py) ---
CAR1_RESET_POS = (-6.0, 0.0)  # must match CAR1_OFFSET in build_vec_deploy_stage.py

command = np.array([0.0, 0.0])
key_map = {
    "W": np.array([1.0, 0.0]), "UP": np.array([1.0, 0.0]),
    "S": np.array([-1.0, 0.0]), "DOWN": np.array([-1.0, 0.0]),
    "A": np.array([0.0, 1.0]), "LEFT": np.array([0.0, 1.0]),
    "D": np.array([0.0, -1.0]), "RIGHT": np.array([0.0, -1.0]),
}


def reset_car1():
    """Car1 has no automatic recovery (it's human-driven) - if it ever
    ends up in the same kind of stuck/collapsed state car2's safety net
    catches, press R to manually put it back on its feet."""
    position = np.array([[CAR1_RESET_POS[0], CAR1_RESET_POS[1], CHASSIS_Z]])
    orientation = np.array([[1.0, 0.0, 0.0, 0.0]])
    car1.set_world_poses(positions=position, orientations=orientation)
    car1.set_velocities(np.zeros((1, 6)))
    car1.set_joint_velocities(np.zeros((1, car1.num_dof)))
    car1.set_joint_positions(np.zeros((1, car1.num_dof)))


def on_keyboard_event(event, *args, **kwargs):
    global command
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input.name in key_map:
            command = command + key_map[event.input.name]
        elif event.input.name == "R":
            reset_car1()
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        if event.input.name in key_map:
            command = command - key_map[event.input.name]
    return True


_stuck_tracker = {"counter": 0, "last_progress": None}
STUCK_CHECK_INTERVAL = 300  # physics steps, ~5 sim-seconds at 60Hz
STUCK_MIN_PROGRESS = 1.5  # meters of centerline progress expected in that window


def control_step(step_size):
    throttle = float(np.clip(command[0], -1.0, 1.0))
    steer = float(np.clip(command[1], -1.0, 1.0))
    car1.set_joint_velocity_targets(
        np.full((1, len(DRIVE_JOINTS)), throttle * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
    )
    car1.set_joint_position_targets(
        np.full((1, len(STEER_JOINTS)), steer * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
    )

    obs, lateral_offset, tangent_angle, px, py, up_z, height_error, progress = car2_observation()

    # Periodically check whether car2 is actually making forward progress
    # around the track. None of the checks below (lateral drift, tipping,
    # wrong height) catch a car that's just weaving/oscillating near where
    # it started without physically failing - an untrained-policy failure
    # mode, not a physics one, so it needs its own explicit check.
    _stuck_tracker["counter"] += 1
    if _stuck_tracker["counter"] >= STUCK_CHECK_INTERVAL:
        _stuck_tracker["counter"] = 0
        if _stuck_tracker["last_progress"] is not None:
            delta = abs(progress - _stuck_tracker["last_progress"])
            if delta > track_length / 2.0:
                delta = track_length - delta
            if delta < STUCK_MIN_PROGRESS:
                recover_car2_onto_track(tangent_angle, px, py)
                _stuck_tracker["last_progress"] = None
                return
        _stuck_tracker["last_progress"] = progress

    # Recovery covers three distinct physical failure modes: drifting off
    # the track (lateral), tipping/rolling over (up_z), and the chassis
    # physically settling at the wrong height, e.g. a wheel/joint solver
    # hiccup under sustained torque or a collision (height_error).
    if abs(lateral_offset) > RECOVERY_MARGIN or up_z < 0.9 or height_error > 0.08:
        recover_car2_onto_track(tangent_angle, px, py)
        return

    action, _ = policy.predict(obs, deterministic=True)
    action = np.clip(action, -1.0, 1.0)
    car2.set_joint_velocity_targets(
        np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
    )
    car2.set_joint_position_targets(
        np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
    )


_telemetry_counter = [0]


def telemetry_step(step_size):
    _telemetry_counter[0] += 1
    if _telemetry_counter[0] % 120 != 0:
        return
    p1, _ = car1.get_world_poses()
    p2, _ = car2.get_world_poses()
    with open(f"{ROOT}/run_output/dual_car_vec_telemetry.txt", "a") as f:
        f.write(f"t={_telemetry_counter[0]} car1={p1[0]} car2={p2[0]}\n")


world.add_physics_callback("control_step", callback_fn=control_step)
world.add_physics_callback("telemetry_step", callback_fn=telemetry_step)

appwindow = omni.appwindow.get_default_app_window()
input_iface = carb.input.acquire_input_interface()
keyboard = appwindow.get_keyboard()
sub_id = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

with open(f"{ROOT}/run_output/run_dual_car_vec_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
