"""Many short heats from the benchmark's start (bottom straight, x~0, spots y=+-1.5) with tiny random start
perturbations, for BOTH spot assignments, to separate a SPOT effect (inside/outside), a LABEL effect (A/B)
and sensitive dependence on initial conditions. Usage: python.bat start_condition_eval.py <n_seeds> <control_steps> <out_csv> [checkpoint path without .zip] [mirror 0|1]
One CSV row per heat. Headless, deterministic policy, same obs construction as lap_time_benchmark.py."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is
import math
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from stable_baselines3 import PPO
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

sys.path.insert(0, f"{ROOT}/2_python_scripts/g_two_car_racing")
from car_unknown_track_env import build_centerline, wrap_to_pi  # noqa: E402

N_SEEDS = int(sys.argv[1])
STEPS = int(sys.argv[2])
OUT = sys.argv[3]
CAR_USD = f"{ROOT}/1_simulation_scenes/unknown_track.usd"
MODEL_PATH = sys.argv[4] if len(sys.argv) > 4 else f"{ROOT}/3_trained_models/car_race_ppo_model"
MIRROR = int(sys.argv[5]) if len(sys.argv) > 5 else 0
DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0
ACTION_REPEAT = 4
CHASSIS_Z = 0.45

open_stage(CAR_USD)
world = World(stage_units_in_meters=1.0)
world.reset()
car_a = Articulation(prim_paths_expr="/World/CarA/chassis")
car_a.initialize()
car_b = Articulation(prim_paths_expr="/World/CarB/chassis")
car_b.initialize()
cars = [car_a, car_b]
policy = PPO.load(MODEL_PATH, device="cpu")
points, cumulative = build_centerline()
n_seg = len(points)
track_len = cumulative[-1]


def nearest_segment(x, y):
    best_i, best_d2, best_t = 0, float("inf"), 0.0
    for i in range(n_seg):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n_seg]
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)))
        px, py = x0 + t * dx, y0 + t * dy
        d2 = (x - px) ** 2 + (y - py) ** 2
        if d2 < best_d2:
            best_d2, best_i, best_t = d2, i, t
    return best_i, best_t, math.sqrt(best_d2)


def tangent_at(s):
    s = s % track_len
    for i in range(n_seg):
        if cumulative[i] <= s <= cumulative[i + 1]:
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n_seg]
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
    x1, y1 = points[(i + 1) % n_seg]
    dx, dy = x1 - x0, y1 - y0
    seg_len = math.hypot(dx, dy)
    tang = math.atan2(dy, dx)
    px, py = x0 + t * dx, y0 + t * dy
    side = math.copysign(1.0, dx * (pos[1] - py) - dy * (pos[0] - px))
    lateral = dist * side
    progress = cumulative[i] + t * seg_len
    herr = wrap_to_pi(yaw - tang)
    curv = wrap_to_pi(tangent_at(progress + LOOKAHEAD_DIST) - tang)
    lin = car.get_linear_velocities()[0]
    ang = car.get_angular_velocities()[0]
    speed = lin[0] * math.cos(yaw) + lin[1] * math.sin(yaw)
    obs7 = np.array([lateral, math.sin(herr), math.cos(herr), speed, ang[2], math.sin(curv), math.cos(curv)], dtype=np.float32)
    return obs7, progress, lateral, speed


def place(car, x, y, yaw):
    half = yaw / 2.0
    car.set_world_poses(positions=np.array([[x, y, CHASSIS_Z]]), orientations=np.array([[math.cos(half), 0.0, 0.0, math.sin(half)]]))
    car.set_velocities(np.zeros((1, 6)))
    n = car.num_dof
    car.set_joint_velocities(np.zeros((1, n)))
    car.set_joint_positions(np.zeros((1, n)))


f = open(OUT, "w")
f.write("swap,seed,dxA,dxB,leader_label,leader_spot,margin_100,margin_200,margin_final,lead_changes,first_leader_label,min_dist250,contact_steps\n")
f.flush()
for swap in (0, 1):
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        dxa, dxb = rng.uniform(-0.5, 0.5, 2)
        dya, dyb = rng.uniform(-0.05, 0.05, 2)
        yaws = rng.uniform(-0.02, 0.02, 2)
        ya, yb = (-1.5, 1.5) if swap == 0 else (1.5, -1.5)
        if MIRROR and swap == 1:
            dxa, dxb = dxb, dxa
            dya, dyb = dyb, dya
            yaws = yaws[::-1]
        place(car_a, dxa, ya + dya, yaws[0])
        place(car_b, dxb, yb + dyb, yaws[1])
        for _ in range(2):
            world.step(render=False)
        last = [None, None]
        cum = [0.0, 0.0]
        margins = {}
        lead_sign = 0
        changes = 0
        first_leader = None
        min_d250 = 99.0
        contact = 0
        for step in range(STEPS):
            st = [own_state(car_a), own_state(car_b)]
            _pa = car_a.get_world_poses()[0][0]
            _pb = car_b.get_world_poses()[0][0]
            _d = math.hypot(_pa[0] - _pb[0], _pa[1] - _pb[1])
            if step < 250:
                min_d250 = min(min_d250, _d)
            if _d < 1.2:
                contact += 1
            prog = [s[1] for s in st]
            if last[0] is None:
                last = list(prog)
            for k in range(2):
                d = prog[k] - last[k]
                if d > track_len / 2:
                    d -= track_len
                elif d < -track_len / 2:
                    d += track_len
                cum[k] += d
                last[k] = prog[k]
            m = cum[1] - cum[0]  # >0 : B ahead
            if step in (100, 200):
                margins[step] = m
            if abs(m) >= 2.0:
                sign = 1 if m > 0 else -1
                if first_leader is None:
                    first_leader = "B" if sign > 0 else "A"
                if lead_sign != 0 and sign != lead_sign:
                    changes += 1
                lead_sign = sign
            gap_ab = prog[1] - prog[0]
            if gap_ab > track_len / 2:
                gap_ab -= track_len
            elif gap_ab < -track_len / 2:
                gap_ab += track_len
            gap_ab = float(np.clip(gap_ab, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
            opp_gap = [gap_ab, -gap_ab]
            lat = [st[0][2], st[1][2]]
            spd = [st[0][3], st[1][3]]
            for idx, car in enumerate(cars):
                obs = np.concatenate([st[idx][0], [opp_gap[idx], lat[1 - idx] - lat[idx], spd[1 - idx] - spd[idx]]])
                action, _ = policy.predict(obs, deterministic=True)
                action = np.clip(action, -1.0, 1.0)
                car.set_joint_velocity_targets(np.full((1, 2), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
                car.set_joint_position_targets(np.full((1, 2), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS)
            for _ in range(ACTION_REPEAT):
                world.step(render=False)
        final = cum[1] - cum[0]
        leader = "B" if final > 0 else "A"
        # spot of the leader: inside = y=+1.5 start (left of travel direction, the corner side)
        leader_y = (yb if leader == "B" else ya)
        spot = "inside" if leader_y > 0 else "outside"
        f.write(f"{swap},{seed},{dxa:.3f},{dxb:.3f},{leader},{spot},{margins.get(100, 0):.2f},{margins.get(200, 0):.2f},{final:.2f},{changes},{first_leader},{min_d250:.2f},{contact}\n")
        f.flush()
        print(f"swap={swap} seed={seed} leader={leader} ({spot}) final margin(B-A)={final:.2f}", flush=True)
f.close()
print("SPOTSTATS DONE", flush=True)
world.stop()
simulation_app.close()
