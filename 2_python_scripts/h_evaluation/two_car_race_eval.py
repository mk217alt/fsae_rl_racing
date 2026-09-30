"""Self-play race evaluation of two-car checkpoints on a single-pair deploy stage.

Both vehicles are driven by the same checkpoint, deterministically, without the recovery net. Every heat
starts from the training-style side-by-side start (start vertex of a segment, 1.5 m either side of the
centerline) on one of n seeded random segments; the same segments are used for every checkpoint of a track.
Per heat and vehicle it records distance along the centerline, lap times, speed, lateral offset (and how
far the vehicle moves to the inside of the corners), action delta and events (off-track, tilt, chassis
height); per pair it records proximity (< 1.2 m, < 2 m), the gap between the vehicles and lead changes.

A lead change is counted when the other vehicle establishes a lead of at least LEAD_HYSTERESIS metres
(one chassis length) over the vehicle that last held such a lead, so side-by-side jitter is not counted.

Usage:  python.bat two_car_race_eval.py <original|stadium> <n_starts> <control_steps> <label> [label ...]
        (label = 3_trained_models/checkpoints/car_race_ppo_model_<label>.zip)
Output: two_car_race_<track>_<label>.json in this folder.
        With the environment variable TRACE=1, the positions and speeds of both vehicles in every step of the
        first heat are written to two_car_race_trace_<track>_<label>.json instead (for path figures).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is
import json
import math
import os
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

TRACK = sys.argv[1]
N_STARTS = int(sys.argv[2])
STEPS = int(sys.argv[3])
LABELS = sys.argv[4:]
OUT_DIR = f"{THESIS}/4_results"

if TRACK == "original":
    from car_race_vec_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi  # noqa: E402
    STAGE = f"{THESIS}/1_simulation_scenes/race_deploy.usd"
else:
    from car_unknown_track_env import CHASSIS_Z, OFF_TRACK_MARGIN, build_centerline, wrap_to_pi  # noqa: E402
    STAGE = f"{THESIS}/1_simulation_scenes/unknown_track.usd"

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
LOOKAHEAD_DIST = 5.0
OPPONENT_GAP_CLIP = 30.0
ACTION_REPEAT = 4
CONTROL_DT = ACTION_REPEAT / 60.0
START_LATERAL_SPACING = 1.5
CONTACT_DIST = 1.2
NEAR_DIST = 2.0
LEAD_HYSTERESIS = 2.0
TRACE_EVERY = 5
TRACE = os.environ.get("TRACE") == "1"

open_stage(STAGE)
ground = get_current_stage().GetPrimAtPath("/World/GroundPlane")
ground_types = sorted({p.GetTypeName() for p in ground.GetAllChildren()} | {ground.GetTypeName()}) if ground else []

world = World(stage_units_in_meters=1.0)
world.reset()
cars = [Articulation(prim_paths_expr="/World/CarA/chassis"), Articulation(prim_paths_expr="/World/CarB/chassis")]
for c in cars:
    c.initialize()

points, cumulative = build_centerline()
n_seg = len(points)
L = cumulative[-1]
seg_tangent = [math.atan2(points[(i + 1) % n_seg][1] - points[i][1], points[(i + 1) % n_seg][0] - points[i][0])
               for i in range(n_seg)]
# turn direction of the corner a segment belongs to: +1 left, -1 right, 0 straight
seg_turn = []
for i in range(n_seg):
    seg_len = cumulative[i + 1] - cumulative[i]
    dpsi = wrap_to_pi(seg_tangent[(i + 1) % n_seg] - seg_tangent[i - 1])
    seg_turn.append(0 if seg_len > 3.0 else int(math.copysign(1, dpsi)))   # arcs are 0.9-1.6 m, straights >= 8 m


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
            return seg_tangent[i]
    return seg_tangent[-1]


def state(car):
    pos, quat = car.get_world_poses()
    pos, (w, x, y, z) = pos[0], quat[0]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    i, t, dist = nearest_segment(pos[0], pos[1])
    x0, y0 = points[i]
    x1, y1 = points[(i + 1) % n_seg]
    dx, dy = x1 - x0, y1 - y0
    tangent = seg_tangent[i]
    px, py = x0 + t * dx, y0 + t * dy
    lateral = dist * math.copysign(1.0, dx * (pos[1] - py) - dy * (pos[0] - px))
    progress = cumulative[i] + t * math.hypot(dx, dy)
    heading_err = wrap_to_pi(yaw - tangent)
    curv = wrap_to_pi(tangent_at(progress + LOOKAHEAD_DIST) - tangent)
    lin, ang = car.get_linear_velocities()[0], car.get_angular_velocities()[0]
    speed = lin[0] * math.cos(yaw) + lin[1] * math.sin(yaw)
    obs7 = np.array([lateral, math.sin(heading_err), math.cos(heading_err), speed, ang[2],
                     math.sin(curv), math.cos(curv)], dtype=np.float32)
    return dict(obs7=obs7, progress=progress, lateral=lateral, speed=speed, turn=seg_turn[i],
                xy=(pos[0], pos[1]), up_z=1.0 - 2.0 * (x * x + y * y), height_err=abs(pos[2] - CHASSIS_Z))


def place(car, x, y, yaw):
    h = yaw / 2.0
    car.set_world_poses(positions=np.array([[x, y, CHASSIS_Z]]),
                        orientations=np.array([[math.cos(h), 0.0, 0.0, math.sin(h)]]))
    car.set_velocities(np.zeros((1, 6)))
    car.set_joint_velocities(np.zeros((1, car.num_dof)))
    car.set_joint_positions(np.zeros((1, car.num_dof)))


def start_side_by_side(seg):
    x0, y0 = points[seg]
    x1, y1 = points[(seg + 1) % n_seg]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy)
    nx, ny = -dy / n, dx / n
    yaw = math.atan2(dy, dx)
    place(cars[0], x0 + nx * START_LATERAL_SPACING, y0 + ny * START_LATERAL_SPACING, yaw)   # A: left
    place(cars[1], x0 - nx * START_LATERAL_SPACING, y0 - ny * START_LATERAL_SPACING, yaw)   # B: right
    for c in cars:
        c.set_joint_velocity_targets(np.zeros((1, 2)), joint_names=DRIVE_JOINTS)
        c.set_joint_position_targets(np.zeros((1, 2)), joint_names=STEER_JOINTS)
    for _ in range(2):
        world.step(render=False)


def run_heat(policy, steps):
    last_prog = [None, None]
    dist = [0.0, 0.0]
    lap_marks = [[], []]
    events = [{"off_track": 0, "tilt": 0, "height": 0}, {"off_track": 0, "tilt": 0, "height": 0}]
    first_event = [None, None]
    abs_lat = [[], []]
    inside = [[], []]
    speeds = [[], []]
    deltas = [[], []]
    last_action = [None, None]
    contact = near = 0
    leader, lead_changes, lead_change_steps = None, 0, []
    gap_trace = []
    path = [[], []]
    st = [state(c) for c in cars]
    for step in range(1, steps + 1):
        if last_prog[0] is None:
            last_prog = [s["progress"] for s in st]
        gap = st[1]["progress"] - st[0]["progress"]
        gap = gap - L if gap > L / 2 else gap + L if gap < -L / 2 else gap
        gap = float(np.clip(gap, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))
        opp = [[gap, st[1]["lateral"] - st[0]["lateral"], st[1]["speed"] - st[0]["speed"]],
               [-gap, st[0]["lateral"] - st[1]["lateral"], st[0]["speed"] - st[1]["speed"]]]
        for k, car in enumerate(cars):
            obs = np.concatenate([st[k]["obs7"], opp[k]]).astype(np.float32)
            a, _ = policy.predict(obs, deterministic=True)
            a = np.clip(a, -1.0, 1.0)
            if last_action[k] is not None:
                deltas[k].append(float(np.abs(a - last_action[k]).sum()))
            last_action[k] = a
            car.set_joint_velocity_targets(np.full((1, 2), float(a[0]) * MAX_WHEEL_SPEED), joint_names=DRIVE_JOINTS)
            car.set_joint_position_targets(np.full((1, 2), float(a[1]) * MAX_STEER_ANGLE), joint_names=STEER_JOINTS)
        for _ in range(ACTION_REPEAT):
            world.step(render=False)

        st = [state(c) for c in cars]
        for k in range(2):
            s = st[k]
            d = s["progress"] - last_prog[k]
            d = d - L if d > L / 2 else d + L if d < -L / 2 else d
            dist[k] += d
            last_prog[k] = s["progress"]
            while dist[k] >= (len(lap_marks[k]) + 1) * L:
                lap_marks[k].append(step)
            abs_lat[k].append(abs(s["lateral"]))
            path[k].append([round(float(s["xy"][0]), 3), round(float(s["xy"][1]), 3), round(float(s["speed"]), 3)])
            speeds[k].append(s["speed"])
            if s["turn"] != 0:
                inside[k].append(s["lateral"] * s["turn"])
            cause = ("off_track" if abs(s["lateral"]) > OFF_TRACK_MARGIN else
                     "tilt" if s["up_z"] < 0.9 else "height" if s["height_err"] > 0.08 else None)
            if cause:
                events[k][cause] += 1
                if first_event[k] is None:
                    first_event[k] = [step, cause]
        sep = math.hypot(st[0]["xy"][0] - st[1]["xy"][0], st[0]["xy"][1] - st[1]["xy"][1])
        contact += sep < CONTACT_DIST
        near += sep < NEAR_DIST
        lead = dist[1] - dist[0]                    # > 0: B ahead
        new = 1 if lead >= LEAD_HYSTERESIS else 0 if lead <= -LEAD_HYSTERESIS else None
        if new is not None and new != leader:
            if leader is not None:
                lead_changes += 1
                lead_change_steps.append(step)
            leader = new
        if step % TRACE_EVERY == 0:
            gap_trace.append(round(float(lead), 2))

    def flying(marks):
        m = marks
        return [round((m[i + 1] - m[i]) * CONTROL_DT, 2) for i in range(len(m) - 1)]

    return {
        "contact_steps_lt_1.2m": int(contact), "near_steps_lt_2m": int(near),
        "lead_changes": lead_changes, "lead_change_steps": lead_change_steps,
        "final_leader": "AB"[leader] if leader is not None else None,
        "mean_abs_gap_m": round(float(np.mean(np.abs(gap_trace))), 2) if gap_trace else None,
        "gap_trace_B_minus_A_every5": gap_trace,
        "path_x_y_speed": path if TRACE else None,
        "lap_marks": lap_marks,
        "cars": [{"slot": "AB"[k], "distance_m": round(float(dist[k]), 2),
                  "first_lap_s": round(lap_marks[k][0] * CONTROL_DT, 2) if lap_marks[k] else None,
                  "flying_laps_s": flying(lap_marks[k]),
                  "mean_speed_mps": round(float(dist[k]) / (steps * CONTROL_DT), 3),
                  "mean_forward_speed_mps": round(float(np.mean(speeds[k])), 3),
                  "mean_abs_lateral_m": round(float(np.mean(abs_lat[k])), 3),
                  "mean_inside_offset_in_corners_m": round(float(np.mean(inside[k])), 3) if inside[k] else None,
                  "mean_action_delta": round(float(np.mean(deltas[k])), 4) if deltas[k] else None,
                  "event_steps": events[k], "first_event": first_event[k]} for k in range(2)],
    }


world.play()
rng = np.random.default_rng(6000)
starts = [int(s) for s in rng.choice(n_seg, size=N_STARTS, replace=False)]
for label in LABELS:
    policy = PPO.load(f"{THESIS}/3_trained_models/checkpoints/car_race_ppo_model_{label}.zip", device="cpu")
    if TRACE:
        start_side_by_side(starts[0])
        h = run_heat(policy, STEPS)
        with open(f"{OUT_DIR}/two_car_race_trace_{TRACK}_{label}.json", "w") as f:
            json.dump({"checkpoint": label, "track": TRACK, "start_segment": starts[0], "heat": h}, f)
        print("TRACE", label, [c["flying_laps_s"] for c in h["cars"]], flush=True)
        continue
    heats = []
    for seg in starts:
        start_side_by_side(seg)
        h = run_heat(policy, STEPS)
        h["start_segment"] = seg
        heats.append(h)
        print(json.dumps({"label": label, "seg": seg, "lead_changes": h["lead_changes"],
                          "contact": h["contact_steps_lt_1.2m"],
                          "cars": [(c["distance_m"], c["flying_laps_s"][:3], c["event_steps"]) for c in h["cars"]]}),
              flush=True)
    cars_all = [c for h in heats for c in h["cars"]]
    fl = [t for c in cars_all for t in c["flying_laps_s"]]
    clean = [h for h in heats if all(c["first_event"] is None for c in h["cars"])]
    summary = {
        "heats": len(heats), "clean_heats": len(clean),
        "vehicles_with_event": sum(c["first_event"] is not None for c in cars_all),
        "vehicles_off_track": sum(c["event_steps"]["off_track"] > 0 for c in cars_all),
        "flying_lap_mean_s": round(float(np.mean(fl)), 2) if fl else None,
        "flying_lap_std_s": round(float(np.std(fl)), 2) if fl else None,
        "flying_lap_median_s": round(float(np.median(fl)), 2) if fl else None,
        "flying_lap_best_s": round(float(np.min(fl)), 2) if fl else None,
        "n_flying_laps": len(fl),
        "mean_speed_mps": round(float(np.mean([c["mean_speed_mps"] for c in cars_all])), 3),
        "mean_abs_lateral_m": round(float(np.mean([c["mean_abs_lateral_m"] for c in cars_all])), 3),
        "mean_inside_offset_in_corners_m": round(float(np.mean([c["mean_inside_offset_in_corners_m"] for c in cars_all
                                                                if c["mean_inside_offset_in_corners_m"] is not None])), 3),
        "mean_action_delta": round(float(np.mean([c["mean_action_delta"] for c in cars_all])), 4),
        "lead_changes_total": int(sum(h["lead_changes"] for h in heats)),
        "heats_with_lead_change": int(sum(h["lead_changes"] > 0 for h in heats)),
        "contact_steps_total": int(sum(h["contact_steps_lt_1.2m"] for h in heats)),
        "near_steps_total": int(sum(h["near_steps_lt_2m"] for h in heats)),
        "heats_with_contact": int(sum(h["contact_steps_lt_1.2m"] > 0 for h in heats)),
        "mean_abs_gap_m": round(float(np.mean([h["mean_abs_gap_m"] for h in heats])), 2),
        "left_start_leads_at_end": int(sum(h["final_leader"] == "A" for h in heats)),
    }
    result = {"stage": _os.path.relpath(STAGE, ROOT).replace("\\", "/"), "ground_prim_types": ground_types, "checkpoint": label, "track": TRACK,
              "track_length_m": round(L, 2), "control_steps_per_heat": STEPS, "start_segments": starts,
              "lead_hysteresis_m": LEAD_HYSTERESIS, "summary": summary, "heats": heats}
    with open(f"{OUT_DIR}/two_car_race_{TRACK}_{label}.json", "w") as f:
        json.dump(result, f, indent=1)
    print("SUMMARY", label, json.dumps(summary), flush=True)
simulation_app.close()
