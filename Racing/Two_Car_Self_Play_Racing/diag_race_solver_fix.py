"""Stress test: command sustained near-max throttle on both race-deploy
cars from a standstill for a long stretch, logging height/orientation each
step, to check whether the increased PhysX solver iteration counts fix the
chassis-collapse instability found earlier (chassis settling to
CHASSIS_SIZE[2]/2 = 0.2 instead of the intended 0.45)."""

import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

CAR_USD = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/race_deploy.usd"
DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
CHASSIS_Z = 0.45

open_stage(CAR_USD)
world = World(stage_units_in_meters=1.0)
world.reset()

car_a = Articulation(prim_paths_expr="/World/CarA/chassis")
car_a.initialize()
car_b = Articulation(prim_paths_expr="/World/CarB/chassis")
car_b.initialize()

world.play()

lines = []
for step in range(400):
    car_a.set_joint_velocity_targets(np.full((1, len(DRIVE_JOINTS)), 0.35 * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
    car_a.set_joint_position_targets(np.full((1, len(STEER_JOINTS)), 0.0), joint_names=STEER_JOINTS)
    car_b.set_joint_velocity_targets(np.full((1, len(DRIVE_JOINTS)), 0.35 * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
    car_b.set_joint_position_targets(np.full((1, len(STEER_JOINTS)), 0.0), joint_names=STEER_JOINTS)

    world.step(render=False)

    if step % 25 == 0 or step == 399:
        pa, oa = car_a.get_world_poses()
        pb, ob = car_b.get_world_poses()
        za, zb = pa[0][2], pb[0][2]
        height_err_a = abs(za - CHASSIS_Z)
        height_err_b = abs(zb - CHASSIS_Z)
        lines.append(f"step {step}: carA z={za:.3f} (err={height_err_a:.3f}) carB z={zb:.3f} (err={height_err_b:.3f})")

with open("C:/Users/sanja/Desktop/thesis/diag_race_solver_fix_log.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

with open("C:/Users/sanja/Desktop/thesis/diag_race_solver_fix_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
