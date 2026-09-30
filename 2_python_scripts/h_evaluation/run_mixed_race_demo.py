"""Live demo of the head-to-head from race_mixed_policy_eval.py: Car A is
driven by the racing-line policy (car2_racingline_ppo_model.zip, 7-dim
observation, no opponent awareness), Car B by the two-car self-play racing
policy (car_race_ppo_model.zip, 10-dim observation with 3 opponent-relative
features). Same track/stage (race_deploy.usd) and physics constants both
were trained under.

Includes the same physical-safety recovery net as run_race_demo.py - a
deliberate, documented workaround for the unresolved chassis-collapse bug
in this stage's wheel joints under sustained throttle (see rl_journal.html
§6). The head-to-head evaluation showed this bug fires noticeably more
often under two-policy interaction than solo driving, so expect visible
recovery teleports during this demo - that's the known limitation, not a
new bug.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g_two_car_racing"))
import math

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
from stable_baselines3 import PPO

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

from car_race_vec_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi

CAR_USD = f"{ROOT}/1_simulation_scenes/race_deploy.usd"
RACINGLINE_MODEL_PATH = f"{ROOT}/3_trained_models/car2_racingline_ppo_model"
RACE_MODEL_PATH = f"{ROOT}/3_trained_models/car_race_ppo_model"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0
NUDGE_PROGRESS_RANGE = 6.0  # meters of progress gap within which Car A starts avoiding
NUDGE_LATERAL_RANGE = 3.5  # meters of lateral gap within which Car A starts avoiding
NUDGE_STRENGTH = 0.7  # max extra steer command added to push Car A away
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
cars = [car_a, car_b]

policy_racingline = PPO.load(RACINGLINE_MODEL_PATH, device="cpu")  # Car A - red
policy_race = PPO.load(RACE_MODEL_PATH, device="cpu")  # Car B - blue

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


RECOVERY_LATERAL_OFFSET = [-0.8, 0.8]  # CarA recovers slightly left, CarB slightly right - stops them stacking


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


ACTION_REPEAT = 4  # must match both training pipelines' cadence
_control_counter = [0]
_action_counter = [0]

# Neither off-track, tilt, nor height checks fire when two cars are simply
# wedged against each other while upright and roughly level - which is
# exactly what happens once Car A (no opponent awareness) drives into Car
# B and both get stuck pushing against each other, going nowhere. Catch
# that directly: if a car hasn't advanced meaningfully in progress over a
# window of control steps, force a recovery regardless of its other state.
STUCK_CHECK_INTERVAL = 300  # control steps between stuck checks
STUCK_MIN_PROGRESS = 1.5  # meters a car must cover in that window to not count as stuck
_stuck_ref_progress = [None, None]
_stuck_ref_action = [0, 0]


def control_step(step_size):
    _control_counter[0] += 1
    if _control_counter[0] % ACTION_REPEAT != 0:
        return  # hold the previously-set joint targets, matching training's cadence
    _action_counter[0] += 1

    states = [own_state(car_a), own_state(car_b)]

    progresses = [s[1] for s in states]
    laterals = [s[2] for s in states]
    speeds = [s[3] for s in states]

    gap_ab = progresses[1] - progresses[0]
    if gap_ab > track_length / 2.0:
        gap_ab -= track_length
    elif gap_ab < -track_length / 2.0:
        gap_ab += track_length
    gap_ab = float(np.clip(gap_ab, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
    opp_gap = [gap_ab, -gap_ab]
    opp_lat_gap = [laterals[1] - laterals[0], laterals[0] - laterals[1]]
    opp_speed_gap = [speeds[1] - speeds[0], speeds[0] - speeds[1]]

    for idx, car in enumerate(cars):
        partial_obs, progress, lateral_offset, fwd_speed, tangent_angle, px, py, up_z, height_error = states[idx]

        if _stuck_ref_progress[idx] is None:
            _stuck_ref_progress[idx] = progress
            _stuck_ref_action[idx] = _action_counter[0]
        stuck = False
        if _action_counter[0] - _stuck_ref_action[idx] >= STUCK_CHECK_INTERVAL:
            delta = progress - _stuck_ref_progress[idx]
            if delta > track_length / 2.0:
                delta -= track_length
            elif delta < -track_length / 2.0:
                delta += track_length
            stuck = delta < STUCK_MIN_PROGRESS
            _stuck_ref_progress[idx] = progress
            _stuck_ref_action[idx] = _action_counter[0]

        if abs(lateral_offset) > RECOVERY_MARGIN or up_z < 0.9 or height_error > 0.08 or stuck:
            recover_car(car, tangent_angle, px, py, RECOVERY_LATERAL_OFFSET[idx])
            continue

        if idx == 0:
            obs = partial_obs  # Car A: racing-line policy, 7-dim, no opponent features
            action, _ = policy_racingline.predict(obs, deterministic=True)
            # Car A's policy has zero opponent awareness by construction, so
            # it will drive straight through Car B given the chance. Nudge
            # its steering away from the opponent when close, scaled by how
            # close it is in both progress and lateral gap, purely a demo-
            # script addition (not touching the policy or car physics).
            prox = max(0.0, 1.0 - abs(opp_gap[0]) / NUDGE_PROGRESS_RANGE)
            lat_close = max(0.0, 1.0 - abs(opp_lat_gap[0]) / NUDGE_LATERAL_RANGE)
            if prox > 0.0 and lat_close > 0.0:
                nudge_sign = 1.0 if opp_lat_gap[0] >= 0.0 else -1.0
                action[1] -= NUDGE_STRENGTH * prox * lat_close * nudge_sign
        else:
            obs = np.concatenate([partial_obs, [opp_gap[idx], opp_lat_gap[idx], opp_speed_gap[idx]]])
            action, _ = policy_race.predict(obs, deterministic=True)

        action = np.clip(action, -1.0, 1.0)
        car.set_joint_velocity_targets(
            np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
        )
        car.set_joint_position_targets(
            np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
        )


_telemetry_counter = [0]


def telemetry_step(step_size):
    _telemetry_counter[0] += 1
    if _telemetry_counter[0] % 120 != 0:
        return
    pa, _ = car_a.get_world_poses()
    pb, _ = car_b.get_world_poses()
    with open(f"{ROOT}/run_output/mixed_race_demo_telemetry.txt", "a") as f:
        f.write(f"t={_telemetry_counter[0]} carA(racing-line)={pa[0]} carB(self-play)={pb[0]}\n")


world.add_physics_callback("control_step", callback_fn=control_step)
world.add_physics_callback("telemetry_step", callback_fn=telemetry_step)

with open(f"{ROOT}/run_output/run_mixed_race_demo_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
