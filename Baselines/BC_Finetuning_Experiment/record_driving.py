"""Records YOUR driving of car 1 as (observation, action) pairs, for
behavioral-cloning pretraining of car 2's policy.

Controls:
    W / Up    - drive forward
    S / Down  - drive backward
    A / Left  - steer left
    D / Right - steer right

Drive around the loop a few times, then just close the window - the
recorded data is saved automatically to human_driving_data.npz.
"""


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Baselines/Single_Car_Baseline"))
import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import carb.input
import omni.appwindow

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

from car_track_env import ACTION_REPEAT, build_centerline, wrap_to_pi

CAR_USD = "C:/Users/sanja/Desktop/thesis/Simulation/Simulation_Environment_Setup/simple_car.usd"
OUT_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/human_driving_data.npz"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)

open_stage(CAR_USD)

settings = carb.settings.get_settings()
settings.set_bool("/physics/updateToUsd", True)
settings.set_bool("/app/useFabricSceneDelegate", False)
settings.set_bool("/physics/fabricUpdateTransformations", False)

world = World(stage_units_in_meters=1.0)
world.reset()

car1 = Articulation(prim_paths_expr="/World/SimpleCar/chassis")
car1.initialize()

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


def car1_observation():
    positions, orientations = car1.get_world_poses()
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

    lin_vel = car1.get_linear_velocities()[0]
    ang_vel = car1.get_angular_velocities()[0]
    forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
    yaw_rate = ang_vel[2]

    return np.array(
        [lateral_offset, math.sin(heading_error), math.cos(heading_error),
         forward_speed, yaw_rate],
        dtype=np.float32,
    )


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


recorded_obs = []
recorded_act = []
_step_counter = [0]


def save():
    np.savez(OUT_PATH, obs=np.array(recorded_obs), act=np.array(recorded_act))


def drive_step(step_size):
    throttle = float(np.clip(command[0], -1.0, 1.0))
    steer = float(np.clip(command[1], -1.0, 1.0))

    car1.set_joint_velocity_targets(
        np.full((1, len(DRIVE_JOINTS)), throttle * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
    )
    car1.set_joint_position_targets(
        np.full((1, len(STEER_JOINTS)), steer * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
    )

    # sample at the same decision cadence the RL environment trains at
    _step_counter[0] += 1
    if _step_counter[0] % ACTION_REPEAT == 0:
        recorded_obs.append(car1_observation())
        recorded_act.append(np.array([throttle, steer], dtype=np.float32))
        if len(recorded_obs) % 500 == 0:
            save()
            with open("C:/Users/sanja/Desktop/thesis/record_driving_progress.txt", "w") as f:
                f.write(f"{len(recorded_obs)} samples recorded so far\n")


world.add_physics_callback("drive_step", callback_fn=drive_step)

appwindow = omni.appwindow.get_default_app_window()
input_iface = carb.input.acquire_input_interface()
keyboard = appwindow.get_keyboard()
sub_id = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

with open("C:/Users/sanja/Desktop/thesis/record_driving_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

save()
with open("C:/Users/sanja/Desktop/thesis/record_driving_done.txt", "w") as f:
    f.write(f"recorded {len(recorded_obs)} samples to {OUT_PATH}\n")

simulation_app.close()
