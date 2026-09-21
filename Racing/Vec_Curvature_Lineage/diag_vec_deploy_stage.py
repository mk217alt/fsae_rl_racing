"""Diagnostic: open vec_deploy_car2.usd headlessly, command a strong constant
forward throttle on both cars with an UNAMBIGUOUS scripted input (no
keyboard involved), and log position + orientation + joint state every few
steps, starting from frame 0 (right after world.play(), before any
stepping) so we can see whether the cars are wrong from the very first
frame (an authoring/joint bug) or degrade over time (a dynamics bug)."""

import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

CAR_USD = "C:/Users/sanja/Desktop/thesis/Racing/Vec_Curvature_Lineage/vec_deploy_car2.usd"
DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0

open_stage(CAR_USD)
world = World(stage_units_in_meters=1.0)
world.reset()

car1 = Articulation(prim_paths_expr="/World/Car1/chassis")
car1.initialize()
car2 = Articulation(prim_paths_expr="/World/Car2/chassis")
car2.initialize()

lines = []
lines.append(f"car1 num_dof={car1.num_dof} dof_names={car1.dof_names}")
lines.append(f"car2 num_dof={car2.num_dof} dof_names={car2.dof_names}")

world.play()

# frame 0: right after play(), before any stepping or commands
p1, o1 = car1.get_world_poses()
p2, o2 = car2.get_world_poses()
lines.append(f"frame0 (before any step/command): car1 pos={p1[0]} quat={o1[0]}")
lines.append(f"frame0 (before any step/command): car2 pos={p2[0]} quat={o2[0]}")

for step in range(150):
    car1.set_joint_velocity_targets(np.full((1, len(DRIVE_JOINTS)), 0.6 * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
    car1.set_joint_position_targets(np.full((1, len(STEER_JOINTS)), 0.0), joint_names=STEER_JOINTS)
    car2.set_joint_velocity_targets(np.full((1, len(DRIVE_JOINTS)), 0.6 * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
    car2.set_joint_position_targets(np.full((1, len(STEER_JOINTS)), 0.0), joint_names=STEER_JOINTS)

    world.step(render=False)

    if step % 20 == 0 or step == 149:
        p1, o1 = car1.get_world_poses()
        p2, o2 = car2.get_world_poses()
        jv1 = car1.get_joint_velocities()
        jv2 = car2.get_joint_velocities()
        lines.append(f"step {step}: car1 pos={p1[0]} quat={o1[0]} joint_vel={jv1[0]}")
        lines.append(f"step {step}: car2 pos={p2[0]} quat={o2[0]} joint_vel={jv2[0]}")

with open("C:/Users/sanja/Desktop/thesis/diag_vec_deploy_stage_log.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

with open("C:/Users/sanja/Desktop/thesis/diag_vec_deploy_stage_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
