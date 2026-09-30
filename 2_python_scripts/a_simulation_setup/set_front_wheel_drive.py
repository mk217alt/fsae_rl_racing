"""Converts simple_car.usd from driving all 4 wheels to front-wheel drive:
the rear wheel joints lose their velocity drive entirely (so they spin
freely, just following the car's motion) while the front wheels keep both
their steering joint and their drive joint, unchanged.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics

CAR_USD = f"{ROOT}/1_simulation_scenes/simple_car.usd"
REAR_DRIVE_JOINTS = [
    "/World/SimpleCar/rear_left_drive_joint",
    "/World/SimpleCar/rear_right_drive_joint",
]

stage = Usd.Stage.Open(CAR_USD)

for joint_path in REAR_DRIVE_JOINTS:
    prim = stage.GetPrimAtPath(joint_path)
    if prim and prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
        prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")

stage.Save()

with open(f"{ROOT}/run_output/set_front_wheel_drive_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
