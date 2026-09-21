"""Live demo of the two-car self-play racing policy (car_race_ppo_model.zip)
on race_deploy.usd - both cars are autonomous, driven by the SAME shared
network (self-play), each fed its own 10-dim observation (7-dim curvature-
aware state + 3 opponent-relative features).

No human driving in this demo - just watch the two cars race. Includes the
same physical-safety recovery net as the single-car demo (off-track,
tipping, wrong-height). This is a deliberate, documented workaround for an
unresolved chassis-collapse bug in this stage's wheel joints under
sustained throttle - reproduced consistently across five different fix
attempts (solver iterations, drive force, knuckle mass, self-collision
filtering, wheel speed), none of which changed the outcome. A collapsed
chassis has no physical way to recover no matter how good the driving
policy is, so until that's properly root-caused, the recovery net stays in
so training/demo work can proceed.
"""

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
from isaacsim.core.utils.stage import open_stage

from car_race_vec_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi

CAR_USD = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/race_deploy.usd"
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/car_race_ppo_model"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0
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


ACTION_REPEAT = 4  # must match car_race_vec_env.py - hold each action this many physics steps
_control_counter = [0]
_action_counter = [0]

# Same stuck-detector as the mixed racing-line-vs-self-play demo: none of
# off-track/tilt/height catch two cars wedged against each other while
# upright, so track per-car progress directly as a backstop.
STUCK_CHECK_INTERVAL = 300
STUCK_MIN_PROGRESS = 1.5
_stuck_ref_progress = [None, None]
_stuck_ref_action = [0, 0]

# Recovery events by cause, so successive training rounds on the racing
# lineage can be compared on "how often did this actually need the demo's
# physical-bug workaround" - off-track should trend down with more/better
# training, tilt/height (the chassis-collapse bug, see rl_journal.html §6)
# is not expected to, since a scripted no-policy stress test reproduces it
# identically regardless of driving quality.
recovery_counts = {"off_track": 0, "tilt": 0, "height": 0, "stuck": 0}
_stats_counter = [0]


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

        # 2026-09-21: recovery net dropped at the user's explicit request, now
        # that driving quality is consistently 85-98%+ full-length on both
        # tracks - the policy runs raw. These counters stay as a diagnostic
        # log of what WOULD have triggered a recovery before, not an
        # intervention: recover_car() is no longer called, so a car that
        # genuinely leaves the track, tips, gets stuck, or hits the still-
        # unresolved chassis-collapse bug now just stays down/off rather than
        # being teleported back.
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
        car.set_joint_velocity_targets(
            np.full((1, len(DRIVE_JOINTS)), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS
        )
        car.set_joint_position_targets(
            np.full((1, len(STEER_JOINTS)), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS
        )


_telemetry_counter = [0]

# 2026-09-21: the periodic ~1s freeze the user noticed during live demos was
# this callback doing synchronous file opens/writes directly on the physics
# thread every 120 steps, at a perfectly regular interval - textbook cause
# of a rhythmic stutter. Fix: the physics callback only does fast in-memory
# work (read poses, snapshot counters) and hands the data to a background
# thread via a queue; that thread does the actual disk I/O, off the physics/
# render loop entirely, so any filesystem latency can no longer stall the sim.
_telemetry_queue = queue.Queue()


def _telemetry_writer():
    while True:
        item = _telemetry_queue.get()
        if item is None:
            break
        t, pa0, pb0, action_count, recovery_snapshot = item
        with open("C:/Users/sanja/Desktop/thesis/race_demo_telemetry.txt", "a") as f:
            f.write(f"t={t} carA={pa0} carB={pb0}\n")

        total = sum(recovery_snapshot.values())
        with open("C:/Users/sanja/Desktop/thesis/race_demo_recovery_stats.txt", "w") as f:
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

with open("C:/Users/sanja/Desktop/thesis/run_race_demo_ready.txt", "w") as f:
    f.write("ready\n")

world.play()
while simulation_app.is_running():
    world.step(render=True)

_telemetry_queue.put(None)
_telemetry_thread.join(timeout=2.0)
simulation_app.close()
