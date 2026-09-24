"""Head-to-head race on the corrected two-car stage: racing-line policy vs. two-car self-play policy.

Same protocol as race_mixed_policy_eval.py (5,000 control steps per heat, deterministic actions, the
same recovery net, a second heat with the lanes swapped), with these changes:
  * race_deploy.usd now has the analytic ground plane (fixed 2026-09-19 17:06); both earlier heats
    (2026-09-19 10:39 / 12:19) ran on the box collider and were dominated by chassis-collapse recoveries.
  * the self-play car uses a fixed, named checkpoint (default: run49, the final original-track round,
    40.09 M steps) instead of whatever car_race_ppo_model.zip currently is (now stadium-adapted).
  * recoveries are counted by cause, and lap times, close-contact steps and lead changes are recorded.
  * the recovery net can be switched off (it teleports a car onto the centerline at its current
    progress, which in a two-car race can drop it onto the opponent).
  * a series mode races from random start segments (side by side, 1.5 m either side of the centerline,
    as in training), each start once per lane assignment, without the recovery net.

The racing-line policy (7-D observation, no opponent features) and the self-play policy (10-D) were
both trained on the reversed vertex order of build_centerline(), which race_deploy.usd also uses.

Usage:  python.bat head_to_head_eval.py fixed <normal|swap> [recovery|norecovery] [selfplay_label]
        python.bat head_to_head_eval.py series <n_starts> <control_steps> [selfplay_label]
Output: head_to_head_<label>_<...>.json in this folder.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is
import json
import math
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

THESIS = ROOT
sys.path.insert(0, f"{THESIS}/2_python_scripts/g_two_car_racing")

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage, open_stage  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from car_race_vec_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "fixed"
if MODE == "fixed":
    LANES = sys.argv[2] if len(sys.argv) > 2 else "normal"
    RECOVERY = (sys.argv[3] if len(sys.argv) > 3 else "recovery") == "recovery"
    SELFPLAY_LABEL = sys.argv[4] if len(sys.argv) > 4 else "run49"
    TAG = f"head_to_head_{SELFPLAY_LABEL}_{LANES}" + ("" if RECOVERY else "_norecovery")
else:
    N_STARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    SERIES_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    SELFPLAY_LABEL = sys.argv[4] if len(sys.argv) > 4 else "run49"
    TAG = f"head_to_head_{SELFPLAY_LABEL}_series{N_STARTS}x{SERIES_STEPS}"
OUT_DIR = f"{THESIS}/4_results"

STAGE = f"{THESIS}/1_simulation_scenes/race_deploy.usd"
RACINGLINE_MODEL = f"{THESIS}/3_trained_models/car2_racingline_ppo_model.zip"
SELFPLAY_MODEL = f"{THESIS}/3_trained_models/checkpoints/car_race_ppo_model_{SELFPLAY_LABEL}.zip"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0
ACTION_REPEAT = 4
CONTROL_DT = ACTION_REPEAT / 60.0
FIXED_CONTROL_STEPS = 5000
START_LATERAL_SPACING = 1.5  # as in training (car_race_vec_env.py)
CONTACT_DIST = 1.2           # proximity-penalty threshold of the self-play reward
NEAR_DIST = 2.0              # nose-to-tail contact distance of two 2 m long chassis

open_stage(STAGE)
ground = get_current_stage().GetPrimAtPath("/World/GroundPlane")
ground_types = sorted({p.GetTypeName() for p in ground.GetAllChildren()} | {ground.GetTypeName()}) if ground else []

world = World(stage_units_in_meters=1.0)
world.reset()
cars = [Articulation(prim_paths_expr="/World/CarA/chassis"), Articulation(prim_paths_expr="/World/CarB/chassis")]
for c in cars:
    c.initialize()

racingline = PPO.load(RACINGLINE_MODEL, device="cpu")
selfplay = PPO.load(SELFPLAY_MODEL, device="cpu")

points, cumulative = build_centerline()
n_seg = len(points)
L = cumulative[-1]


def nearest_segment(x, y):
    best = (float("inf"), 0, 0.0)
    for i in range(n_seg):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n_seg]
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)))
        d2 = (x - x0 - t * dx) ** 2 + (y - y0 - t * dy) ** 2
        if d2 < best[0]:
            best = (d2, i, t)
    return best[1], best[2], math.sqrt(best[0])


def tangent_at(s):
    s = s % L
    for i in range(n_seg):
        if cumulative[i] <= s <= cumulative[i + 1]:
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n_seg]
            return math.atan2(y1 - y0, x1 - x0)
    return math.atan2(points[0][1] - points[-1][1], points[0][0] - points[-1][0])


def state(car):
    pos, quat = car.get_world_poses()
    pos, (w, x, y, z) = pos[0], quat[0]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    i, t, dist = nearest_segment(pos[0], pos[1])
    x0, y0 = points[i]
    x1, y1 = points[(i + 1) % n_seg]
    dx, dy = x1 - x0, y1 - y0
    tangent = math.atan2(dy, dx)
    px, py = x0 + t * dx, y0 + t * dy
    lateral = dist * math.copysign(1.0, dx * (pos[1] - py) - dy * (pos[0] - px))
    progress = cumulative[i] + t * math.hypot(dx, dy)
    heading_err = wrap_to_pi(yaw - tangent)
    curv = wrap_to_pi(tangent_at(progress + LOOKAHEAD_DIST) - tangent)
    lin, ang = car.get_linear_velocities()[0], car.get_angular_velocities()[0]
    speed = lin[0] * math.cos(yaw) + lin[1] * math.sin(yaw)
    obs7 = np.array([lateral, math.sin(heading_err), math.cos(heading_err), speed, ang[2],
                     math.sin(curv), math.cos(curv)], dtype=np.float32)
    return dict(obs7=obs7, progress=progress, lateral=lateral, speed=speed, tangent=tangent, px=px, py=py,
                xy=(pos[0], pos[1]), up_z=1.0 - 2.0 * (x * x + y * y), height_err=abs(pos[2] - CHASSIS_Z))


def place(car, x, y, yaw):
    h = yaw / 2.0
    car.set_world_poses(positions=np.array([[x, y, CHASSIS_Z]]),
                        orientations=np.array([[math.cos(h), 0.0, 0.0, math.sin(h)]]))
    car.set_velocities(np.zeros((1, 6)))
    car.set_joint_velocities(np.zeros((1, car.num_dof)))
    car.set_joint_positions(np.zeros((1, car.num_dof)))


def start_side_by_side(seg):
    """Training-style start: both cars at the start vertex of segment seg, 1.5 m either side."""
    x0, y0 = points[seg]
    x1, y1 = points[(seg + 1) % n_seg]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy)
    nx, ny = -dy / n, dx / n
    yaw = math.atan2(dy, dx)
    place(cars[0], x0 + nx * START_LATERAL_SPACING, y0 + ny * START_LATERAL_SPACING, yaw)
    place(cars[1], x0 - nx * START_LATERAL_SPACING, y0 - ny * START_LATERAL_SPACING, yaw)
    for c in cars:  # no drive targets carried over from the previous heat
        c.set_joint_velocity_targets(np.zeros((1, 2)), joint_names=DRIVE_JOINTS)
        c.set_joint_position_targets(np.zeros((1, 2)), joint_names=STEER_JOINTS)
    for _ in range(2):
        world.step(render=False)


def laps(marks):
    m = [0] + marks
    return [round((m[i + 1] - m[i]) * CONTROL_DT, 2) for i in range(len(marks))]


def run_heat(swap, steps, recovery, snapshot_every=250):
    policy = [selfplay, racingline] if swap else [racingline, selfplay]   # index = physical slot A/B
    label = ["self-play", "racing-line"] if swap else ["racing-line", "self-play"]
    is_racingline = [lab == "racing-line" for lab in label]
    last_prog = [None, None]
    dist = [0.0, 0.0]
    lap_marks = [[], []]
    causes = [{"off_track": 0, "tilt": 0, "height": 0}, {"off_track": 0, "tilt": 0, "height": 0}]
    first_event = [None, None]
    lead_steps = [0, 0]
    lead_changes = 0
    last_leader = None
    contact_steps = near_steps = 0
    action_deltas = [[], []]
    last_action = [None, None]
    snapshots = []

    for step in range(1, steps + 1):
        for _ in range(ACTION_REPEAT):
            world.step(render=False)
        st = [state(c) for c in cars]
        if last_prog[0] is None:
            last_prog = [s["progress"] for s in st]
        for k in range(2):
            d = st[k]["progress"] - last_prog[k]
            d = d - L if d > L / 2 else d + L if d < -L / 2 else d
            dist[k] += d
            last_prog[k] = st[k]["progress"]
            while dist[k] >= (len(lap_marks[k]) + 1) * L:
                lap_marks[k].append(step)

        gap = st[1]["progress"] - st[0]["progress"]
        gap = gap - L if gap > L / 2 else gap + L if gap < -L / 2 else gap
        gap = float(np.clip(gap, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
        opp = [[gap, st[1]["lateral"] - st[0]["lateral"], st[1]["speed"] - st[0]["speed"]],
               [-gap, st[0]["lateral"] - st[1]["lateral"], st[0]["speed"] - st[1]["speed"]]]
        sep = math.hypot(st[0]["xy"][0] - st[1]["xy"][0], st[0]["xy"][1] - st[1]["xy"][1])
        contact_steps += sep < CONTACT_DIST
        near_steps += sep < NEAR_DIST

        for k, car in enumerate(cars):
            s = st[k]
            cause = ("off_track" if abs(s["lateral"]) > OFF_TRACK_MARGIN else
                     "tilt" if s["up_z"] < 0.9 else "height" if s["height_err"] > 0.08 else None)
            if cause:
                causes[k][cause] += 1                       # without the net: steps spent in that state
                if first_event[k] is None:
                    first_event[k] = [step, cause]
                if recovery:
                    place(car, s["px"], s["py"], s["tangent"])
                    last_action[k] = None
                    continue
            obs = s["obs7"] if is_racingline[k] else np.concatenate([s["obs7"], opp[k]]).astype(np.float32)
            a, _ = policy[k].predict(obs, deterministic=True)
            a = np.clip(a, -1.0, 1.0)
            if last_action[k] is not None:
                action_deltas[k].append(float(np.abs(a - last_action[k]).sum()))
            last_action[k] = a
            car.set_joint_velocity_targets(np.full((1, 2), float(a[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
            car.set_joint_position_targets(np.full((1, 2), float(a[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS)

        leader = 0 if dist[0] > dist[1] else 1 if dist[1] > dist[0] else last_leader
        if leader is not None:
            lead_steps[leader] += 1
            if last_leader is not None and leader != last_leader:
                lead_changes += 1
            last_leader = leader
        if step % snapshot_every == 0:
            snapshots.append([step, round(float(dist[0]), 1), round(float(dist[1]), 1)])

    return {
        "lanes": "swap" if swap else "normal", "recovery_net": recovery, "control_steps": steps,
        "sim_time_s": round(steps * CONTROL_DT, 1), "contact_steps_lt_1.2m": int(contact_steps),
        "near_steps_lt_2m": int(near_steps), "lead_changes": lead_changes,
        "cars": [{"slot": "AB"[k], "policy": label[k], "distance_m": round(float(dist[k]), 2), "laps": round(float(dist[k]) / L, 2),
                  "lap_times_s": laps(lap_marks[k]), "lead_share": round(lead_steps[k] / steps, 3),
                  "event_steps_by_cause": causes[k], "first_event": first_event[k],
                  "mean_action_delta": round(float(np.mean(action_deltas[k])), 4) if action_deltas[k] else None}
                 for k in range(2)],
        "snapshots_step_distA_distB": snapshots,
    }


world.play()
header = {"stage": STAGE, "ground_prim_types": ground_types, "selfplay_checkpoint": SELFPLAY_LABEL,
          "track_length_m": round(L, 2)}
if MODE == "fixed":
    result = {**header, **run_heat(LANES == "swap", FIXED_CONTROL_STEPS, RECOVERY)}
    printable = {k: v for k, v in result.items() if k != "snapshots_step_distA_distB"}
else:
    rng = np.random.default_rng(4000)
    starts = [int(s) for s in rng.choice(n_seg, size=N_STARTS, replace=False)]
    heats = []
    for seg in starts:
        for swap in (False, True):
            start_side_by_side(seg)
            h = run_heat(swap, SERIES_STEPS, recovery=False, snapshot_every=SERIES_STEPS)
            h["start_segment"] = seg
            heats.append(h)
            print(json.dumps({"seg": seg, "lanes": h["lanes"], "contact": h["contact_steps_lt_1.2m"],
                              "cars": [(c["policy"], c["distance_m"], c["event_steps_by_cause"]) for c in h["cars"]]}),
                  flush=True)

    def per_policy(name):
        return [c["distance_m"] for h in heats for c in h["cars"] if c["policy"] == name]

    rl, sp = per_policy("racing-line"), per_policy("self-play")
    diffs = [s - r for s, r in zip(sp, rl)]
    result = {**header, "mode": "series", "n_starts": N_STARTS, "control_steps_per_heat": SERIES_STEPS,
              "recovery_net": False, "start_segments": starts,
              "summary": {"racing_line_mean_distance_m": round(float(np.mean(rl)), 2),
                          "self_play_mean_distance_m": round(float(np.mean(sp)), 2),
                          "mean_diff_selfplay_minus_racingline_m": round(float(np.mean(diffs)), 2),
                          "std_diff_m": round(float(np.std(diffs)), 2),
                          "heats_won_by_self_play": int(sum(d > 0 for d in diffs)),
                          "heats_won_by_racing_line": int(sum(d < 0 for d in diffs)),
                          "heats_won_by_slot_B": int(sum(h["cars"][1]["distance_m"] > h["cars"][0]["distance_m"] for h in heats)),
                          "heats_with_contact": int(sum(h["contact_steps_lt_1.2m"] > 0 for h in heats)),
                          "heats": len(heats)},
              "heats": heats}
    printable = result["summary"]
with open(f"{OUT_DIR}/{TAG}.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(printable, indent=2), flush=True)
simulation_app.close()
