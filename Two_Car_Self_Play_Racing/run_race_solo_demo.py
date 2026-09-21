"""Solo comparison run: the two-car self-play racing policy
(car_race_ppo_model.zip) driving just ONE car alone on the race track, no
opponent at all. Car B is teleported far off the track once at startup and
never touched again - only Car A is driven and measured.

Purpose: the live two-car demo measured the chassis-collapse bug (see
rl_journal.html §6) firing on ~97-98% of all recovery events. That number
was always measured with two cars interacting. This solo run answers the
natural next question directly - does the collapse rate drop when there's
no second car to collide with / stress the wheel joints against, or does
it stay just as high (matching the original no-policy stress test finding
that this bug fires under sustained throttle alone, regardless of what's
driving)?

The policy's observation is 10-dim (7-dim curvature-aware state + 3
opponent-relative features) and was never trained on a genuine "no
opponent" input, so there's no fully faithful way to fill those 3 slots.
Filled with opp_gap=+OPPONENT_GAP_CLIP (reads as "opponent far ahead"),
opp_lat_gap=0, opp_speed_gap=0 - the most neutral, "nothing nearby to
react to" encoding available, not a trained-for input.
"""

import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
from stable_baselines3 import PPO

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

from car_race_vec_env import CHASSIS_Z, OFF_TRACK_MARGIN, OPPONENT_GAP_CLIP, build_centerline, wrap_to_pi

CAR_USD = "C:/Users/sanja/Desktop/thesis/Two_Car_Self_Play_Racing/race_deploy.usd"
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Two_Car_Self_Play_Racing/car_race_ppo_model"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
RECOVERY_MARGIN = OFF_TRACK_MARGIN

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

# Park Car B far away, once, so it can never physically interact with Car A.
car_b.set_world_poses(positions=np.array([[500.0, 500.0, -50.0]]))
car_b.set_velocities(np.zeros((1, 6)))

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
    return partial_obs, progress, lateral_offset, tangent_angle, px, py, up_z, height_error


def recover_car(car, tangent_angle, px, py):
    half_yaw = tangent_angle / 2.0
    orientation = np.array([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
    position = np.array([[px, py, CHASSIS_Z]])
    car.set_world_poses(positions=position, orientations=orientation)
    car.set_velocities(np.zeros((1, 6)))
    car.set_joint_velocities(np.zeros((1, car.num_dof)))
    car.set_joint_positions(np.zeros((1, car.num_dof)))


ACTION_REPEAT = 4
_control_counter = [0]
_action_counter = [0]

STUCK_CHECK_INTERVAL = 300
STUCK_MIN_PROGRESS = 1.5
_stuck_ref_progress = [None]
_stuck_ref_action = [0]

recovery_counts = {"off_track": 0, "tilt": 0, "height": 0, "stuck": 0}


def control_step(step_size):
    _control_counter[0] += 1
    if _control_counter[0] % ACTION_REPEAT != 0:
        return
    _action_counter[0] += 1

    partial_obs, progress, lateral_offset, tangent_angle, px, py, up_z, height_error = own_state(car_a)

    if _stuck_ref_progress[0] is None:
        _stuck_ref_progress[0] = progress
        _stuck_ref_action[0] = _action_counter[0]
    stuck = False
    if _action_counter[0] - _stuck_ref_action[0] >= STUCK_CHECK_INTERVAL:
        delta = progress - _stuck_ref_progress[0]
        if delta > track_length / 2.0:
            delta -= track_length
        elif delta < -track_length / 2.0:
            delta += track_length
        stuck = delta < STUCK_MIN_PROGRESS
        _stuck_ref_progress[0] = progress
        _stuck_ref_action[0] = _action_counter[0]

    off_track = abs(lateral_offset) > RECOVERY_MARGIN
    tilted = up_z < 0.9
    wrong_height = height_error > 0.08
    if off_track or tilted or wrong_height or stuck:
        if off_track:
            recovery_counts["off_track"] += 1
        elif tilted:
            recovery_counts["tilt"] += 1
        elif wrong_height:
            recovery_counts["height"] += 1
        else:
            recovery_counts["stuck"] += 1
        recover_car(car_a, tangent_angle, px, py)
        return

    obs = np.concatenate([partial_obs, [OPPONENT_GAP_CLIP, 0.0, 0.0]])
    action, _ = policy.predict(obs, deterministic=True)
    action = np.clip(action, -1.0, 1.0)
    car_a.set_joint_velocity_targets(
        np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
    )
    car_a.set_joint_position_targets(
        np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
    )


_telemetry_counter = [0]


def telemetry_step(step_size):
    _telemetry_counter[0] += 1
    if _telemetry_counter[0] % 120 != 0:
        return
    pa, _ = car_a.get_world_poses()
    with open("C:/Users/sanja/Desktop/thesis/race_solo_demo_telemetry.txt", "a") as f:
        f.write(f"t={_telemetry_counter[0]} carA={pa[0]}\n")

    total = sum(recovery_counts.values())
    with open("C:/Users/sanja/Desktop/thesis/race_solo_demo_recovery_stats.txt", "w") as f:
        f.write(f"t={_telemetry_counter[0]} total_control_actions={_action_counter[0]}\n")
        for cause, count in recovery_counts.items():
            f.write(f"  {cause}: {count} ({100.0 * count / total if total else 0:.1f}%)\n")


world.add_physics_callback("control_step", callback_fn=control_step)
world.add_physics_callback("telemetry_step", callback_fn=telemetry_step)

with open("C:/Users/sanja/Desktop/thesis/run_race_solo_demo_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
