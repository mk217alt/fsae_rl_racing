"""Live demo of the two-car self-play racing policy (car_race_ppo_model.zip)
on the UNKNOWN track (unknown_track.usd, built by build_unknown_track_deploy.py)
used for the zero-shot generalization test - not the track it trained on.
Same self-play setup as run_race_demo.py (shared network, both cars
autonomous), just pointed at the new track's stage and centerline geometry,
so this is what to open to actually watch the generalization-test result
(see car_racing_status.md's "Unknown-track zero-shot generalization test"
entry for the numeric result: survival matches the trained track almost
exactly, but reward/driving-line quality collapses).

Includes the same physical-safety recovery net and the async-telemetry fix
as run_race_demo.py (see that file's 2026-09-21 note on the periodic-stutter
bug this avoids).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import math
import queue
import threading

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
from stable_baselines3 import PPO

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import get_current_stage, open_stage

from car_unknown_track_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi

CAR_USD = f"{ROOT}/1_simulation_scenes/unknown_track.usd"
MODEL_PATH = f"{ROOT}/3_trained_models/car_race_ppo_model"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0
RECOVERY_MARGIN = OFF_TRACK_MARGIN

open_stage(CAR_USD)

# unknown_track.usd's two spawn spots (build_unknown_track_deploy.py, (0,-1.5)
# and (0,1.5)) are fixed, and one is mildly favorable into the first corner,
# so the same car always won (swap test 2026-09-23: the win follows the spot,
# not the car). Randomly swap which car starts where. Done here, on the
# authored USD translates BEFORE World.reset(): every rigid body of each car
# (chassis, wheels, knuckles) is a direct child of /World/CarX with its own
# world-space translate, so moving them all by the chassis-to-chassis offset
# relocates the whole car with no runtime teleport. The earlier runtime
# version (set_world_poses + world.step before world.play()) froze this
# rendered demo at exactly 90 control actions.
_swapped = bool(np.random.default_rng().random() < 0.5)
if _swapped:
    _stage = get_current_stage()
    _root_a, _root_b = _stage.GetPrimAtPath("/World/CarA"), _stage.GetPrimAtPath("/World/CarB")
    _delta = (_root_b.GetChild("chassis").GetAttribute("xformOp:translate").Get()
              - _root_a.GetChild("chassis").GetAttribute("xformOp:translate").Get())
    for _root, _shift in ((_root_a, _delta), (_root_b, -_delta)):
        for _child in _root.GetChildren():
            _attr = _child.GetAttribute("xformOp:translate")
            if _attr and _attr.IsValid():
                _attr.Set(_attr.Get() + _shift)
print(f"[spawn-swap] cars swapped this run: {_swapped}", flush=True)

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
    return partial_obs, progress, lateral_offset, forward_speed, tangent_angle, px, py, up_z, height_error


ACTION_REPEAT = 4  # must match car_unknown_track_env.py - hold each action this many physics steps
_control_counter = [0]
_action_counter = [0]

STUCK_CHECK_INTERVAL = 300
STUCK_MIN_PROGRESS = 1.5
_stuck_ref_progress = [None, None]
_stuck_ref_action = [0, 0]

recovery_counts = {"off_track": 0, "tilt": 0, "height": 0, "stuck": 0}

CONTROL_DT = ACTION_REPEAT / 60.0
_prev_action = [None, None]
_delta_queue = queue.Queue()


def _delta_writer():
    with open(f"{ROOT}/run_output/unknown_track_demo_action_deltas.csv", "w") as f:
        f.write("t_sim,car,delta\n")
        f.flush()
        n = 0
        while True:
            item = _delta_queue.get()
            if item is None:
                break
            t, car, delta = item
            f.write(f"{t:.3f},{car},{delta:.5f}\n")
            n += 1
            if n % 30 == 0:
                f.flush()


_delta_thread = threading.Thread(target=_delta_writer, daemon=True)
_delta_thread.start()


def control_step(step_size):
    _control_counter[0] += 1
    if _control_counter[0] % ACTION_REPEAT != 0:
        return
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

        # 2026-09-21: recovery net dropped at the user's explicit request, now
        # that driving quality is consistently 85-98%+ full-length on both
        # tracks - the policy runs raw. These counters stay as a diagnostic
        # log of what WOULD have triggered a recovery before, not an
        # intervention (see run_race_demo.py's matching note).
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

        obs = np.concatenate([partial_obs, [opp_gap[idx], opp_lat_gap[idx], opp_speed_gap[idx]]])
        action, _ = policy.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        if _prev_action[idx] is not None:
            delta = float(abs(action[0] - _prev_action[idx][0]) + abs(action[1] - _prev_action[idx][1]))
            t_sim = _action_counter[0] * CONTROL_DT
            _delta_queue.put((t_sim, "A" if idx == 0 else "B", delta))
        _prev_action[idx] = action

        car.set_joint_velocity_targets(
            np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
        )
        car.set_joint_position_targets(
            np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
        )


_telemetry_counter = [0]
_telemetry_queue = queue.Queue()


def _telemetry_writer():
    while True:
        item = _telemetry_queue.get()
        if item is None:
            break
        t, pa0, pb0, action_count, recovery_snapshot = item
        with open(f"{ROOT}/run_output/unknown_track_demo_telemetry.txt", "a") as f:
            f.write(f"t={t} carA={pa0} carB={pb0}\n")

        total = sum(recovery_snapshot.values())
        with open(f"{ROOT}/run_output/unknown_track_demo_recovery_stats.txt", "w") as f:
            f.write(f"t={t} total_control_actions={action_count}\n")
            for cause, count in recovery_snapshot.items():
                f.write(f"  {cause}: {count} ({100.0 * count / total if total else 0:.1f}%)\n")


_telemetry_thread = threading.Thread(target=_telemetry_writer, daemon=True)
_telemetry_thread.start()


def telemetry_step(step_size):
    _telemetry_counter[0] += 1
    if _telemetry_counter[0] % 120 != 0:
        return
    pa, _ = car_a.get_world_poses()
    pb, _ = car_b.get_world_poses()
    _telemetry_queue.put((_telemetry_counter[0], pa[0], pb[0], _action_counter[0], dict(recovery_counts)))


world.add_physics_callback("control_step", callback_fn=control_step)
world.add_physics_callback("telemetry_step", callback_fn=telemetry_step)

with open(f"{ROOT}/run_output/run_unknown_track_demo_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

_telemetry_queue.put(None)
_telemetry_thread.join(timeout=2.0)
_delta_queue.put(None)
_delta_thread.join(timeout=2.0)
simulation_app.close()
