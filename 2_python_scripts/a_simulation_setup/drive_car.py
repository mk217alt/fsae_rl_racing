"""Opens simple_car.usd and lets you drive it live with the keyboard.

Controls:
    W / Up    - drive forward
    S / Down  - drive backward
    A / Left  - steer left
    D / Right - steer right
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import carb.input
import omni.appwindow

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

CAR_USD = f"{ROOT}/1_simulation_scenes/simple_car.usd"
CAR_ROOT = "/World/SimpleCar/chassis"

DRIVE_JOINTS = [
    "front_left_drive_joint",
    "front_right_drive_joint",
]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]

MAX_WHEEL_SPEED = 20.0  # rad/s
MAX_STEER_ANGLE = np.radians(25.0)  # rad

# throttle, steer in [-1, 1]
command = np.array([0.0, 0.0])

# +1 command component per key while held, summed on press, subtracted on release
key_map = {
    "W": np.array([1.0, 0.0]),
    "UP": np.array([1.0, 0.0]),
    "S": np.array([-1.0, 0.0]),
    "DOWN": np.array([-1.0, 0.0]),
    "A": np.array([0.0, 1.0]),
    "LEFT": np.array([0.0, 1.0]),
    "D": np.array([0.0, -1.0]),
    "RIGHT": np.array([0.0, -1.0]),
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


open_stage(CAR_USD)

# Force PhysX to write simulated transforms back to the USD stage every step,
# and disable the separate Fabric-based render delegate so there is only one
# rendering pathway (plain USD Hydra) for physics-simulated prims. Without
# this, the fast in-memory "Fabric" simulation cache and the regular
# USD-authored geometry can render as two separate copies - a live one
# (from Fabric) and a frozen one (from the never-updated USD pose) - which is
# exactly the "two cars" artifact seen in the viewport.
settings = carb.settings.get_settings()
settings.set_bool("/physics/updateToUsd", True)
settings.set_bool("/app/useFabricSceneDelegate", False)
settings.set_bool("/physics/fabricUpdateTransformations", False)

world = World(stage_units_in_meters=1.0)
world.reset()

car = Articulation(prim_paths_expr=CAR_ROOT)
car.initialize()


def drive_step(step_size):
    throttle = float(np.clip(command[0], -1.0, 1.0))
    steer = float(np.clip(command[1], -1.0, 1.0))

    vel_cmd = np.full((1, len(DRIVE_JOINTS)), throttle * MAX_WHEEL_SPEED)
    car.set_joint_velocity_targets(vel_cmd, joint_names=DRIVE_JOINTS)

    pos_cmd = np.full((1, len(STEER_JOINTS)), steer * MAX_STEER_ANGLE)
    car.set_joint_position_targets(pos_cmd, joint_names=STEER_JOINTS)


world.add_physics_callback("drive_step", callback_fn=drive_step)

appwindow = omni.appwindow.get_default_app_window()
input_iface = carb.input.acquire_input_interface()
keyboard = appwindow.get_keyboard()
sub_id = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

with open(f"{ROOT}/run_output/drive_car_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
