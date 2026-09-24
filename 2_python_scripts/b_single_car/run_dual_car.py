"""Drives car 1 with the keyboard while car 2 drives itself autonomously
using the PPO policy trained in train_car2.py.

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

from car_track_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi

CAR_USD = f"{ROOT}/1_simulation_scenes/simple_car.usd"
MODEL_PATH = f"{ROOT}/3_trained_models/car2_ppo_model"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
# Keeps car 2 looping forever: if it drifts past the same off-track margin
# used during training, teleport it back onto the centerline at wherever it
# currently is (rather than all the way back to the start line) so it just
# carries on around the lap instead of getting stuck for good.
RECOVERY_MARGIN = OFF_TRACK_MARGIN

open_stage(CAR_USD)

settings = carb.settings.get_settings()
settings.set_bool("/physics/updateToUsd", True)
settings.set_bool("/app/useFabricSceneDelegate", False)
settings.set_bool("/physics/fabricUpdateTransformations", False)

world = World(stage_units_in_meters=1.0)
world.reset()

car1 = Articulation(prim_paths_expr="/World/SimpleCar/chassis")
car1.initialize()
car2 = Articulation(prim_paths_expr="/World/SimpleCar2/chassis")
car2.initialize()

policy = PPO.load(MODEL_PATH, device="cpu")
points, cumulative = build_centerline()
n_segments = len(points)


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


def car2_observation():
    positions, orientations = car2.get_world_poses()
    pos = positions[0]
    w, x, y, z = orientations[0]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    i, t, dist = nearest_segment(pos[0], pos[1])
    x0, y0 = points[i]
    x1, y1 = points[(i + 1) % n_segments]
    dx, dy = x1 - x0, y1 - y0
    tangent_angle = math.atan2(dy, dx)
    px, py = x0 + t * dx, y0 + t * dy
    side = math.copysign(1.0, dx * (pos[1] - py) - dy * (pos[0] - px))
    lateral_offset = dist * side
    heading_error = wrap_to_pi(yaw - tangent_angle)

    lin_vel = car2.get_linear_velocities()[0]
    ang_vel = car2.get_angular_velocities()[0]
    forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
    yaw_rate = ang_vel[2]

    obs = np.array(
        [lateral_offset, math.sin(heading_error), math.cos(heading_error),
         forward_speed, yaw_rate],
        dtype=np.float32,
    )
    return obs, lateral_offset, tangent_angle, px, py


def recover_car2_onto_track(tangent_angle, px, py):
    half_yaw = tangent_angle / 2.0
    orientation = np.array([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
    position = np.array([[px, py, CHASSIS_Z]])
    car2.set_world_poses(positions=position, orientations=orientation)
    car2.set_velocities(np.zeros((1, 6)))
    car2.set_joint_velocities(np.zeros((1, car2.num_dof)))
    car2.set_joint_positions(np.zeros((1, car2.num_dof)))


# --- car 1: keyboard control (same as drive_car.py) ---
command = np.array([0.0, 0.0])
key_map = {
    "W": np.array([1.0, 0.0]), "UP": np.array([1.0, 0.0]),
    "S": np.array([-1.0, 0.0]), "DOWN": np.array([-1.0, 0.0]),
    "A": np.array([0.0, 1.0]), "LEFT": np.array([0.0, 1.0]),
    "D": np.array([0.0, -1.0]), "RIGHT": np.array([0.0, -1.0]),
}


def on_keyboard_event(event, *args, **kwargs):
    global command
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input.name in key_map:
            command = command + key_map[event.input.name]
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        if event.input.name in key_map:
            command = command - key_map[event.input.name]
    return True


def control_step(step_size):
    # car 1: human input
    throttle = float(np.clip(command[0], -1.0, 1.0))
    steer = float(np.clip(command[1], -1.0, 1.0))
    car1.set_joint_velocity_targets(
        np.full((1, len(DRIVE_JOINTS)), throttle * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
    )
    car1.set_joint_position_targets(
        np.full((1, len(STEER_JOINTS)), steer * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
    )

    # car 2: trained policy, with a safety net so it keeps looping forever
    obs, lateral_offset, tangent_angle, px, py = car2_observation()
    if abs(lateral_offset) > RECOVERY_MARGIN:
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
    if _telemetry_counter[0] % 120 != 0:  # roughly every 2 seconds
        return
    p1, _ = car1.get_world_poses()
    p2, _ = car2.get_world_poses()
    with open(f"{ROOT}/run_output/dual_car_telemetry.txt", "a") as f:
        f.write(f"t={_telemetry_counter[0]} car1={p1[0]} car2={p2[0]}\n")


world.add_physics_callback("control_step", callback_fn=control_step)
world.add_physics_callback("telemetry_step", callback_fn=telemetry_step)

appwindow = omni.appwindow.get_default_app_window()
input_iface = carb.input.acquire_input_interface()
keyboard = appwindow.get_keyboard()
sub_id = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

with open(f"{ROOT}/run_output/run_dual_car_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
