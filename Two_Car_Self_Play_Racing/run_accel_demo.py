"""Stripped-down diagnostic demo: both cars drive with constant forward
throttle and zero steering - no RL policy involved at all - to check
whether the two-car deploy stage can show smooth, continuous forward
movement without the cars getting physically stuck on each other.

Built after the racing-line-vs-self-play live demo showed both cars
freezing into a single overlapping blob shortly after starting. Likely
cause: the recovery-net workaround for the chassis-collapse bug (see
rl_journal.html §6) teleports each car to the nearest point ON the
centerline (zero lateral offset) whenever it fires - with no awareness of
where the OTHER car is. If both cars are at a similar progress point when
recovery fires for each of them, they get placed on top of each other,
collide/re-collapse, and never escape the loop. Fix applied here: each car
recovers to a small FIXED lateral offset from centerline (opposite sides)
instead of dead center, so a recovery event can't stack them.

Since there's no steering, the cars will drive straight off the road once
the track curves - expected, this is a movement/physics check, not a lap
attempt.
"""

import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

from car_race_vec_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi

CAR_USD = "C:/Users/sanja/Desktop/thesis/Two_Car_Self_Play_Racing/race_deploy.usd"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
RECOVERY_MARGIN = OFF_TRACK_MARGIN
CONST_THROTTLE = 0.7
RECOVERY_LATERAL_OFFSET = [-0.8, 0.8]  # CarA recovers slightly left, CarB slightly right

open_stage(CAR_USD)

settings = carb.settings.get_settings()
settings.set_bool("/physics/updateToUsd", True)
settings.set_bool("/app/useFabricSceneDelegate", False)
settings.set_bool("/physics/fabricUpdateTransformations", False)

world = World(stage_units_in_meters=1.0)
world.reset()

car_a = Articulation(prim_paths_expr="/World/CarA/chassis")
car_a.initialize()
car_b = Articulation(prim_paths_expr="/World/CarB/chassis")
car_b.initialize()
cars = [car_a, car_b]

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


def own_state(car):
    positions, orientations = car.get_world_poses()
    pos = positions[0]
    w, x, y, z = orientations[0]

    i, t, dist = nearest_segment(pos[0], pos[1])
    x0, y0 = points[i]
    x1, y1 = points[(i + 1) % n_segments]
    dx, dy = x1 - x0, y1 - y0
    tangent_angle = math.atan2(dy, dx)
    px, py = x0 + t * dx, y0 + t * dy
    side = math.copysign(1.0, dx * (pos[1] - py) - dy * (pos[0] - px))
    lateral_offset = dist * side

    up_z = 1.0 - 2.0 * (x * x + y * y)
    height_error = abs(pos[2] - CHASSIS_Z)
    return lateral_offset, tangent_angle, px, py, up_z, height_error


def recover_car(car, tangent_angle, px, py, lateral_offset):
    nx, ny = -math.sin(tangent_angle), math.cos(tangent_angle)
    rpx, rpy = px + nx * lateral_offset, py + ny * lateral_offset
    half_yaw = tangent_angle / 2.0
    orientation = np.array([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
    position = np.array([[rpx, rpy, CHASSIS_Z]])
    car.set_world_poses(positions=position, orientations=orientation)
    car.set_velocities(np.zeros((1, 6)))
    car.set_joint_velocities(np.zeros((1, car.num_dof)))
    car.set_joint_positions(np.zeros((1, car.num_dof)))


ACTION_REPEAT = 4
_control_counter = [0]


def control_step(step_size):
    _control_counter[0] += 1
    if _control_counter[0] % ACTION_REPEAT != 0:
        return

    for idx, car in enumerate(cars):
        lateral_offset, tangent_angle, px, py, up_z, height_error = own_state(car)

        if abs(lateral_offset) > RECOVERY_MARGIN or up_z < 0.9 or height_error > 0.08:
            recover_car(car, tangent_angle, px, py, RECOVERY_LATERAL_OFFSET[idx])
            continue

        car.set_joint_velocity_targets(
            np.full((1, len(DRIVE_JOINTS)), CONST_THROTTLE * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
        )
        car.set_joint_position_targets(
            np.full((1, len(STEER_JOINTS)), 0.0), joint_names=STEER_JOINTS
        )


_telemetry_counter = [0]


def telemetry_step(step_size):
    _telemetry_counter[0] += 1
    if _telemetry_counter[0] % 60 != 0:
        return
    pa, _ = car_a.get_world_poses()
    pb, _ = car_b.get_world_poses()
    with open("C:/Users/sanja/Desktop/thesis/accel_demo_telemetry.txt", "a") as f:
        f.write(f"t={_telemetry_counter[0]} carA={pa[0]} carB={pb[0]}\n")


world.add_physics_callback("control_step", callback_fn=control_step)
world.add_physics_callback("telemetry_step", callback_fn=telemetry_step)

with open("C:/Users/sanja/Desktop/thesis/run_accel_demo_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
