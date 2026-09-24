"""Overtake-from-behind evaluation. The LEADER starts at x~0 on the benchmark's bottom straight, the TRAILING car
S metres behind it (S in 2,4,6), for every combination of trailing label (A/B) x trailing spot (inside y=+1.5 /
outside y=-1.5) x seeds (tiny perturbations). Runs STEPS control steps, deterministic policy, same obs
construction as lap_time_benchmark.py. Per heat: did the initially trailing car ever lead by >= 2 m, does it lead
at the end, final margin, min inter-car distance, contact steps.
Usage: python.bat stagger_eval.py <out_csv> <checkpoint path without .zip> [steps] [seeds]"""
import math
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from stable_baselines3 import PPO
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage

sys.path.insert(0, "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing")
from car_unknown_track_env import build_centerline, wrap_to_pi  # noqa: E402

OUT = sys.argv[1]
MODEL_PATH = sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 500
N_SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
CAR_USD = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/unknown_track.usd"
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
    return obs7, progress, lateral, speed, (pos[0], pos[1])


def place(car, x, y, yaw):
    half = yaw / 2.0
    car.set_world_poses(positions=np.array([[x, y, CHASSIS_Z]]), orientations=np.array([[math.cos(half), 0.0, 0.0, math.sin(half)]]))
    car.set_velocities(np.zeros((1, 6)))
    n = car.num_dof
    car.set_joint_velocities(np.zeros((1, n)))
    car.set_joint_positions(np.zeros((1, n)))


f = open(OUT, "w")
f.write("stagger_m,trail_label,trail_spot,seed,ever_led2m,leads_at_end,final_margin_trail_minus_lead,min_dist,contact_steps,lead_changes\n")
f.flush()
for S in (2.0, 4.0, 6.0):
    for trail_label in ("A", "B"):
        for trail_spot in ("inside", "outside"):
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(500 + seed)
                dl, dt = rng.uniform(-0.1, 0.1, 2)
                dyl, dyt = rng.uniform(-0.05, 0.05, 2)
                yl, yt = rng.uniform(-0.02, 0.02, 2)
                y_trail = 1.5 if trail_spot == "inside" else -1.5
                y_lead = -y_trail
                pos = {"lead": (0.0 + dl, y_lead + dyl, yl), "trail": (-S + dt, y_trail + dyt, yt)}
                lead_label = "B" if trail_label == "A" else "A"
                pa = pos["trail"] if trail_label == "A" else pos["lead"]
                pb = pos["trail"] if trail_label == "B" else pos["lead"]
                place(car_a, *pa)
                place(car_b, *pb)
                for _ in range(2):
                    world.step(render=False)
                idx_trail = 0 if trail_label == "A" else 1
                last = [None, None]
                cum = [0.0, 0.0]
                ever = False
                changes = 0
                sign_prev = 0
                min_d = 99.0
                contact = 0
                for step in range(STEPS):
                    st = [own_state(car_a), own_state(car_b)]
                    d_xy = math.hypot(st[0][4][0] - st[1][4][0], st[0][4][1] - st[1][4][1])
                    min_d = min(min_d, d_xy)
                    if d_xy < 1.2:
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
                    # margin of the initially trailing car relative to the leader (cum starts at 0 for both,
                    # so add the initial deficit S to the leader's side)
                    margin = (cum[idx_trail] - S) - cum[1 - idx_trail]
                    if margin >= 2.0:
                        ever = True
                    sgn = 1 if margin >= 2.0 else (-1 if margin <= -2.0 else 0)
                    if sgn != 0 and sign_prev != 0 and sgn != sign_prev:
                        changes += 1
                    if sgn != 0:
                        sign_prev = sgn
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
                final = (cum[idx_trail] - S) - cum[1 - idx_trail]
                f.write(f"{S},{trail_label},{trail_spot},{seed},{int(ever)},{int(final > 0)},{final:.2f},{min_d:.2f},{contact},{changes}\n")
                f.flush()
                print(f"S={S} trail={trail_label}/{trail_spot} seed={seed}: ever_led={ever} leads_at_end={final > 0} margin={final:.2f} min_dist={min_d:.2f}", flush=True)
f.close()
print("STAGGER EVAL DONE", flush=True)
world.stop()
simulation_app.close()
