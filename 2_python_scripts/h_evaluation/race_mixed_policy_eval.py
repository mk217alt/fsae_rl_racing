"""Head-to-head evaluation: Car A driven by the racing-line policy
(car2_racingline_ppo_model.zip, single-car lineage, 7-dim observation, no
opponent awareness at all) vs Car B driven by the two-car self-play racing
policy (car_race_ppo_model.zip, 10-dim observation including 3
opponent-relative features). Both were trained on the identical track
geometry, action cadence, and physics constants (see car_race_vec_env.py /
car_track_racingline_vec_env.py) so this is a fair head-to-head, not a
transfer-learning test.

Question: does raw speed with zero opponent model (racing-line) beat a
policy that has actually learned to race against another car (self-play),
or does opponent-awareness win out even against a faster solo policy?

Runs headless on race_deploy.usd (the same single-pair deploy stage used by
run_race_demo.py), reusing that script's state/observation/recovery-net
logic exactly. Instead of an interactive GUI loop, runs for a fixed control-
step budget and logs distance-traveled/lead/recovery stats for both cars,
plus periodic leader snapshots, to a summary file.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g_two_car_racing"))
import math
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

# Pass "swap" on the command line to swap which physical lane (CarA/CarB)
# each policy drives, as a check against lane-spawn bias (see
# car_racing_status.md - "car B performs better than car A" was previously
# traced to spawn-position luck, not the policy).
SWAP = len(sys.argv) > 1 and sys.argv[1] == "swap"
OUT_SUFFIX = "_swapped" if SWAP else ""

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
RECOVERY_MARGIN = OFF_TRACK_MARGIN
ACTION_REPEAT = 4  # must match training cadence for both lineages

EVAL_CONTROL_STEPS = 5000  # ~10x a single training episode's length
SNAPSHOT_EVERY = 250  # control steps between leader snapshots

open_stage(CAR_USD)

world = World(stage_units_in_meters=1.0)
world.reset()

car_a = Articulation(prim_paths_expr="/World/CarA/chassis")
car_a.initialize()
car_b = Articulation(prim_paths_expr="/World/CarB/chassis")
car_b.initialize()

policy_racingline = PPO.load(RACINGLINE_MODEL_PATH, device="cpu")
policy_race = PPO.load(RACE_MODEL_PATH, device="cpu")
# policies[0] drives CarA, policies[1] drives CarB
policies_by_slot = [policy_race, policy_racingline] if SWAP else [policy_racingline, policy_race]
obs_is_racingline = [not SWAP, SWAP]  # which slot uses the 7-dim (no-opponent) observation

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


def recover_car(car, tangent_angle, px, py):
    half_yaw = tangent_angle / 2.0
    orientation = np.array([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
    position = np.array([[px, py, CHASSIS_Z]])
    car.set_world_poses(positions=position, orientations=orientation)
    car.set_velocities(np.zeros((1, 6)))
    car.set_joint_velocities(np.zeros((1, car.num_dof)))
    car.set_joint_positions(np.zeros((1, car.num_dof)))


world.play()

last_progress = [None, None]
total_distance = [0.0, 0.0]
recovery_count = [0, 0]
lead_control_steps = [0, 0]  # how many snapshots each car was ahead in total distance
snapshots = []

control_step = 0
physics_step = 0

while control_step < EVAL_CONTROL_STEPS:
    for _ in range(ACTION_REPEAT):
        world.step(render=False)
        physics_step += 1
    control_step += 1

    states = [own_state(car_a), own_state(car_b)]
    progresses = [s[1] for s in states]
    laterals = [s[2] for s in states]
    speeds = [s[3] for s in states]

    if last_progress[0] is None:
        last_progress = list(progresses)

    for idx in range(2):
        delta = progresses[idx] - last_progress[idx]
        if delta > track_length / 2.0:
            delta -= track_length
        elif delta < -track_length / 2.0:
            delta += track_length
        total_distance[idx] += delta
        last_progress[idx] = progresses[idx]

    gap_ab = progresses[1] - progresses[0]
    if gap_ab > track_length / 2.0:
        gap_ab -= track_length
    elif gap_ab < -track_length / 2.0:
        gap_ab += track_length
    gap_ab = float(np.clip(gap_ab, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
    opp_gap = [gap_ab, -gap_ab]
    opp_lat_gap = [laterals[1] - laterals[0], laterals[0] - laterals[1]]
    opp_speed_gap = [speeds[1] - speeds[0], speeds[0] - speeds[1]]

    cars = [car_a, car_b]
    for idx, car in enumerate(cars):
        partial_obs, progress, lateral_offset, fwd_speed, tangent_angle, px, py, up_z, height_error = states[idx]

        if abs(lateral_offset) > RECOVERY_MARGIN or up_z < 0.9 or height_error > 0.08:
            recover_car(car, tangent_angle, px, py)
            recovery_count[idx] += 1
            continue

        if obs_is_racingline[idx]:
            obs = partial_obs  # racing-line policy: 7-dim, no opponent features
        else:
            obs = np.concatenate([partial_obs, [opp_gap[idx], opp_lat_gap[idx], opp_speed_gap[idx]]])

        action, _ = policies_by_slot[idx].predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)
        car.set_joint_velocity_targets(
            np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
        )
        car.set_joint_position_targets(
            np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
        )

    if total_distance[0] > total_distance[1]:
        lead_control_steps[0] += 1
    elif total_distance[1] > total_distance[0]:
        lead_control_steps[1] += 1

    if control_step % SNAPSHOT_EVERY == 0:
        snapshots.append((control_step, total_distance[0], total_distance[1]))

laps_a = total_distance[0] / track_length
laps_b = total_distance[1] / track_length
label_a = "self-play racing" if SWAP else "racing-line"
label_b = "racing-line" if SWAP else "self-play racing"
winner = f"Car A ({label_a})" if total_distance[0] > total_distance[1] else f"Car B ({label_b})"

lines = [
    f"=== Mixed head-to-head: Car A = {label_a} policy, Car B = {label_b} policy ({'lanes swapped' if SWAP else 'original lanes'}) ===",
    f"track_length={track_length:.2f}m  eval_control_steps={EVAL_CONTROL_STEPS}  action_repeat={ACTION_REPEAT}",
    "",
    f"Car A total distance: {total_distance[0]:.2f}m  ({laps_a:.2f} laps)  recovery triggers: {recovery_count[0]}",
    f"Car B total distance: {total_distance[1]:.2f}m  ({laps_b:.2f} laps)  recovery triggers: {recovery_count[1]}",
    f"Distance gap (A - B): {total_distance[0] - total_distance[1]:.2f}m",
    f"Fraction of control steps A led: {lead_control_steps[0] / control_step:.1%}",
    f"Fraction of control steps B led: {lead_control_steps[1] / control_step:.1%}",
    f"Winner (more distance covered): {winner}",
    "",
    "--- leader snapshots (control_step, dist_A, dist_B) ---",
]
for cs, da, db in snapshots:
    lines.append(f"step {cs}: A={da:.1f}m  B={db:.1f}m  lead={'A' if da > db else 'B'}")

with open(f"{ROOT}/run_output/race_mixed_policy_eval_log{OUT_SUFFIX}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

with open(f"{ROOT}/run_output/race_mixed_policy_eval_done{OUT_SUFFIX}.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
