"""Diagnostic: per-control-step trace of both cars from the fixed benchmark start, with the spawn swap
forced on or off. Usage: python.bat start_trace.py <swap 0|1> <control_steps> <out_csv>
Mirrors lap_time_benchmark.py (headless, deterministic policy, same obs construction); only adds logging.
Also prints initial pose/yaw, the centerline tangent at the start, and a USD attribute comparison of
/World/CarA vs /World/CarB (masses, joint drives, sizes) to look for a physical asymmetry."""
import math
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from stable_baselines3 import PPO
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import get_current_stage, open_stage

sys.path.insert(0, "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing")
from car_unknown_track_env import build_centerline, wrap_to_pi  # noqa: E402

SWAP = int(sys.argv[1])
STEPS = int(sys.argv[2])
OUT = sys.argv[3]
CAR_USD = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/unknown_track.usd"
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/car_race_ppo_model"
DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0
ACTION_REPEAT = 4
CONTROL_DT = ACTION_REPEAT / 60.0

open_stage(CAR_USD)
stage = get_current_stage()

# ---- USD comparison of the two cars (relative to their own chassis offset) ----
def dump(car):
    rows = {}
    root = stage.GetPrimAtPath(f"/World/{car}")
    cx = root.GetChild("chassis").GetAttribute("xformOp:translate").Get()
    for child in root.GetChildren():
        for a in child.GetAttributes():
            v = a.Get()
            if v is None:
                continue
            if a.GetName() == "xformOp:translate":
                v = tuple(round(float(c), 4) for c in (v - cx))  # position relative to own chassis
            rows[f"{child.GetName()}.{a.GetName()}"] = str(v)
        for g in child.GetChildren():
            for a in g.GetAttributes():
                v = a.Get()
                if v is not None:
                    rows[f"{child.GetName()}/{g.GetName()}.{a.GetName()}"] = str(v)
    return rows, cx

ra, cxa = dump("CarA")
rb, cxb = dump("CarB")
print("CHASSIS translate A:", cxa, " B:", cxb, flush=True)
diffs = [(k, ra.get(k), rb.get(k)) for k in sorted(set(ra) | set(rb)) if ra.get(k) != rb.get(k)]
print(f"USD attribute differences A vs B (positions relative to own chassis): {len(diffs)}", flush=True)
for k, va, vb in diffs[:40]:
    print(f"  DIFF {k}: A={va[:90]}  B={vb[:90]}", flush=True)

world = World(stage_units_in_meters=1.0)
world.reset()
car_a = Articulation(prim_paths_expr="/World/CarA/chassis")
car_a.initialize()
car_b = Articulation(prim_paths_expr="/World/CarB/chassis")
car_b.initialize()
cars = [car_a, car_b]

if SWAP:
    pos_a, orient_a = car_a.get_world_poses()
    pos_b, orient_b = car_b.get_world_poses()
    car_a.set_world_poses(positions=pos_b, orientations=orient_b)
    car_b.set_world_poses(positions=pos_a, orientations=orient_a)
    car_a.set_velocities(np.zeros((1, 6)))
    car_b.set_velocities(np.zeros((1, 6)))
    for _ in range(2):
        world.step(render=False)

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
    return obs7, progress, lateral, speed, (pos[0], pos[1], yaw, herr, tang)


for _ in range(2):
    world.step(render=False)

st0 = [own_state(car_a), own_state(car_b)]
for n, s in zip("AB", st0):
    x, y, yaw, herr, tang = s[4]
    print(f"START car{n}: x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f}deg tangent={math.degrees(tang):.1f}deg heading_err={math.degrees(herr):.1f}deg lateral={s[2]:.2f} progress={s[1]:.2f}", flush=True)

f = open(OUT, "w")
f.write("step,t,car,x,y,yaw_deg,heading_err_deg,progress_cum,lateral,speed,throttle,steer,gap_ahead\n")
last_prog = [None, None]
cum_dist = [0.0, 0.0]
for step in range(STEPS):
    states = [own_state(car_a), own_state(car_b)]
    prog = [s[1] for s in states]
    if last_prog[0] is None:
        last_prog = list(prog)
    for k in range(2):
        d = prog[k] - last_prog[k]
        if d > track_len / 2:
            d -= track_len
        elif d < -track_len / 2:
            d += track_len
        cum_dist[k] += d
        last_prog[k] = prog[k]
    gap_ab = prog[1] - prog[0]
    if gap_ab > track_len / 2:
        gap_ab -= track_len
    elif gap_ab < -track_len / 2:
        gap_ab += track_len
    gap_ab = float(np.clip(gap_ab, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
    opp_gap = [gap_ab, -gap_ab]
    speeds = [states[0][3], states[1][3]]
    lat = [states[0][2], states[1][2]]
    for idx, car in enumerate(cars):
        obs = np.concatenate([states[idx][0], [opp_gap[idx], lat[1 - idx] - lat[idx], speeds[1 - idx] - speeds[idx]]])
        action, _ = policy.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)
        car.set_joint_velocity_targets(np.full((1, 2), float(action[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
        car.set_joint_position_targets(np.full((1, 2), float(action[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS)
        x, y, yaw, herr, tang = states[idx][4]
        # gap_ahead > 0 means THIS car is ahead of the other
        f.write(f"{step},{step*CONTROL_DT:.3f},{'AB'[idx]},{x:.3f},{y:.3f},{math.degrees(yaw):.1f},{math.degrees(herr):.1f},{cum_dist[idx]:.3f},{states[idx][2]:.3f},{states[idx][3]:.3f},{action[0]:.3f},{action[1]:.3f},{-opp_gap[idx]:.3f}\n")
    for _ in range(ACTION_REPEAT):
        world.step(render=False)
f.close()
print("TRACE DONE", flush=True)
world.stop()
simulation_app.close()
