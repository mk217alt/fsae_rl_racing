"""Playable version of the two-car race: Car A (red) is driven by YOU with
the keyboard, Car B (blue) drives at constant throttle / no steering (the
"acceleration mode" from run_accel_demo.py) instead of the RL policy - a
simpler, predictable second car while you test your own driving. Same
track, same physics both cars trained/tested on all session.

Controls:
    W / Up    - throttle forward
    S / Down  - throttle backward
    A / Left  - steer left
    D / Right - steer right

Both cars get the same recovery-net workaround for the documented
chassis-collapse bug (see rl_journal.html §6) - if your own car suddenly
teleports back onto the track, that's the same bug catching you, not a
crash. It fires on ~45% of all control decisions in this stage regardless
of who (or what) is driving, so don't be surprised if it happens often.
Recovery causes are logged per car to race_playable_recovery_stats.txt so
you can check afterward what actually happened to your own car.
"""

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

from car_race_vec_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi

CAR_USD = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/race_deploy.usd"
CONST_THROTTLE = 0.7  # Car B's constant-throttle "acceleration mode"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
RECOVERY_MARGIN = OFF_TRACK_MARGIN

# throttle, steer in [-1, 1] - Car A (human)
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


open_stage(CAR_USD)

settings = carb.settings.get_settings()
settings.set_bool("/physics/updateToUsd", True)
settings.set_bool("/app/useFabricSceneDelegate", False)
settings.set_bool("/physics/fabricUpdateTransformations", False)

world = World(stage_units_in_meters=1.0)
world.reset()

car_a = Articulation(prim_paths_expr="/World/CarA/chassis")  # human
car_a.initialize()
car_b = Articulation(prim_paths_expr="/World/CarB/chassis")  # constant-throttle
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


def own_state(car):
    positions, orientations = car.get_world_poses()
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

    lin_vel = car.get_linear_velocities()[0]
    ang_vel = car.get_angular_velocities()[0]
    forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
    yaw_rate = ang_vel[2]

    partial_obs = np.array(
        [lateral_offset, math.sin(heading_error), math.cos(heading_error),
         forward_speed, yaw_rate, math.sin(curvature_ahead), math.cos(curvature_ahead)],
        dtype=np.float32,
    )
    up_z = 1.0 - 2.0 * (x * x + y * y)
    height_error = abs(pos[2] - CHASSIS_Z)
    return partial_obs, progress, lateral_offset, forward_speed, tangent_angle, px, py, up_z, height_error


RECOVERY_LATERAL_OFFSET = [-0.8, 0.8]


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
_action_counter = [0]

recovery_counts = {
    "carA": {"off_track": 0, "tilt": 0, "height": 0},
    "carB": {"off_track": 0, "tilt": 0, "height": 0},
}


def record_recovery(car_key, off_track, tilted, wrong_height):
    if off_track:
        recovery_counts[car_key]["off_track"] += 1
    elif tilted:
        recovery_counts[car_key]["tilt"] += 1
    elif wrong_height:
        recovery_counts[car_key]["height"] += 1


def control_step(step_size):
    _control_counter[0] += 1
    if _control_counter[0] % ACTION_REPEAT != 0:
        return
    _action_counter[0] += 1

    states = [own_state(car_a), own_state(car_b)]

    # Recovery net PAUSED for this testing session (user's request) - still
    # detected and logged (record_recovery), just not acted on, so you can
    # drive/observe through a collapse instead of being teleported out of it.
    RECOVERY_ENABLED = False

    # Car A: human keyboard control
    partial_obs, progress, lateral_offset, fwd_speed, tangent_angle, px, py, up_z, height_error = states[0]
    off_track, tilted, wrong_height = abs(lateral_offset) > RECOVERY_MARGIN, up_z < 0.9, height_error > 0.08
    if (off_track or tilted or wrong_height):
        record_recovery("carA", off_track, tilted, wrong_height)
    if (off_track or tilted or wrong_height) and RECOVERY_ENABLED:
        recover_car(car_a, tangent_angle, px, py, RECOVERY_LATERAL_OFFSET[0])
    else:
        throttle = float(np.clip(command[0], -1.0, 1.0))
        steer = float(np.clip(command[1], -1.0, 1.0))
        car_a.set_joint_velocity_targets(np.full((1, len(DRIVE_JOINTS)), throttle * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
        car_a.set_joint_position_targets(np.full((1, len(STEER_JOINTS)), steer * MAX_STEER_ANGLE), joint_names=STEER_JOINTS)

    # Car B: constant-throttle "acceleration mode" (from run_accel_demo.py) instead
    # of the RL policy - simpler, predictable second car while you test your own driving.
    partial_obs, progress, lateral_offset, fwd_speed, tangent_angle, px, py, up_z, height_error = states[1]
    off_track, tilted, wrong_height = abs(lateral_offset) > RECOVERY_MARGIN, up_z < 0.9, height_error > 0.08
    if (off_track or tilted or wrong_height):
        record_recovery("carB", off_track, tilted, wrong_height)
    if (off_track or tilted or wrong_height) and RECOVERY_ENABLED:
        recover_car(car_b, tangent_angle, px, py, RECOVERY_LATERAL_OFFSET[1])
    else:
        car_b.set_joint_velocity_targets(np.full((1, len(DRIVE_JOINTS)), CONST_THROTTLE * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
        car_b.set_joint_position_targets(np.full((1, len(STEER_JOINTS)), 0.0), joint_names=STEER_JOINTS)


_telemetry_counter = [0]


def telemetry_step(step_size):
    _telemetry_counter[0] += 1
    if _telemetry_counter[0] % 120 != 0:
        return
    with open("C:/Users/sanja/Desktop/thesis/race_playable_recovery_stats.txt", "w") as f:
        f.write(f"t={_telemetry_counter[0]} total_control_actions={_action_counter[0]}\n")
        for car_key, counts in recovery_counts.items():
            total = sum(counts.values())
            f.write(f"{car_key}: total_recoveries={total}\n")
            for cause, count in counts.items():
                f.write(f"  {cause}: {count} ({100.0 * count / total if total else 0:.1f}%)\n")


world.add_physics_callback("control_step", callback_fn=control_step)
world.add_physics_callback("telemetry_step", callback_fn=telemetry_step)

appwindow = omni.appwindow.get_default_app_window()
input_iface = carb.input.acquire_input_interface()
keyboard = appwindow.get_keyboard()
sub_id = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

with open("C:/Users/sanja/Desktop/thesis/run_race_playable_demo_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
