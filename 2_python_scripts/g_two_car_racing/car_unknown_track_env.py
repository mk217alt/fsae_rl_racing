"""VecEnv variant of car_race_vec_env.py, pointed at the unknown-track
geometry (40m long straight, 8m short straight, 6m corner radius - vs the
original lineage's trained-on 16m/16m/10m symmetric square) instead of the
original race_train.usd/race_deploy.usd.

Supports both uses: NUM_PAIRS=1 against unknown_track.usd (built by
build_unknown_track_deploy.py) for the single-pair zero-shot generalization
test, and NUM_PAIRS=8 against unknown_track_train.usd (built by
build_unknown_track_train.py) for actual continued training on this track,
matching the original lineage's parallel-pair throughput.

Everything except the stage path, pair count/grid, and centerline geometry
is identical to car_race_vec_env.py - same observation/reward/reset logic,
which is already shape-agnostic (it only consumes build_centerline()'s
points/cumulative-distance output, never the track's specific shape) - so
reusing it exactly is what makes this a fair comparison/continuation of the
same policy, not a different environment definition.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import math

import numpy as np
from gymnasium import spaces
from isaacsim import SimulationApp
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

_simulation_app = None


def get_simulation_app(headless=True):
    global _simulation_app
    if _simulation_app is None:
        _simulation_app = SimulationApp({"headless": headless})
    return _simulation_app


DEPLOY_STAGE_PATH = f"{ROOT}/1_simulation_scenes/unknown_track.usd"
TRAIN_STAGE_PATH = f"{ROOT}/1_simulation_scenes/unknown_track_train.usd"
NUM_PAIRS = 8
GRID_COLS = 3
GRID_SPACING = 70.0

# Must match build_unknown_track_deploy.py exactly.
HALF_LENGTH_X = 20.0
HALF_LENGTH_Y = 4.0
CORNER_RADIUS = 6.0
TRACK_WIDTH = 9.0
ARC_SEGMENTS = 10

DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]
MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
CHASSIS_Z = 0.45

ACTION_REPEAT = 4
MAX_EPISODE_STEPS = 500
OFF_TRACK_MARGIN = TRACK_WIDTH / 2.0 - 0.3
LATERAL_PENALTY_WEIGHT = 0.9
OFF_TRACK_PENALTY = 150.0
# 2026-09-21: raised 0.15->0.30 at the user's request after the lap-timing
# benchmark confirmed a real ~15% slowdown (22s->26s laps) developing over
# runs 6-12 - the policy had drifted toward a cautious, low-throttle style
# since nothing in the reward pushed back against it. Doubling (not a huge
# jump) to nudge speed back up without repeating this project's history of
# oversized weight changes causing overcorrections (see the collision-
# penalty arc in car_racing_status.md).
THROTTLE_BONUS_WEIGHT = 0.30
LOOKAHEAD_DIST = 5.0
START_LATERAL_SPACING = 1.5
STAGGER_MIN = 1.5
STAGGER_MAX = 8.0
COLLISION_DIST = 1.2
COLLISION_PENALTY = 1.5
OPPONENT_GAP_CLIP = 30.0
# 2026-09-21: overtake incentive, added at the user's request after
# observing one car just follows the other instead of actually racing.
# Root cause: neither car's reward depended at all on its position relative
# to the opponent - only individual progress + a shared collision-avoidance
# penalty - so self-play had nothing to gain by contesting position and
# converged to a safe follow-the-leader equilibrium. This term rewards a
# car for gaining relative ground on its opponent (closing the gap when
# behind, extending the lead when ahead) each step, and penalizes losing
# ground - computed from the change in the same opponent_progress_gap
# observation feature already used elsewhere, so the two cars in a pair
# always get exactly opposite-signed values (zero-sum, redistributing
# reward toward competitive racing rather than inflating the total reward
# budget). Kept modest (0.3) relative to the unweighted ~1.0/meter progress
# term, matching this project's repeated lesson that oversized new per-step
# terms cause runaway-suppression pathologies (the COLLISION_PENALTY=5.0
# incident, the run19-21 overcorrection arc).
# 2026-09-23: raised 0.3->0.5 at the user's request after judging the
# racing "still not competitive" even at the elite full-length/reward tier
# reached by run35/36 - dead-even lap times and near-perfect survival don't
# by themselves mean the two cars are actually contesting position rather
# than settling into a stable parallel gap. Stayed below a full doubling
# since it's still an open question whether more incentive alone fixes a
# qualitative "doesn't really race" complaint - being watched next round.
OVERTAKE_INCENTIVE_WEIGHT = 0.5
# 2026-09-21: smoothness penalty, added at the user's request after
# reporting jerky driving on the demo following the throttle-bonus raise
# and overtake incentive above - both reward bursty throttle/steer changes
# (accelerate hard to close a gap, brake into a corner) and nothing here
# previously penalized jerkiness, unlike the single-car baseline
# (car_track_env.py), which already uses this exact mechanism at the same
# weight and measured an 18.6% reduction in action jitter with no loss of
# consistency. Mirrors that fix exactly: penalize the per-step change in
# consecutive throttle/steer commands, kept at the same modest 0.08 weight
# relative to the unweighted ~1.0/meter progress term.
SMOOTHNESS_PENALTY_WEIGHT = 0.08


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def build_centerline():
    ax, ay, r = HALF_LENGTH_X, HALF_LENGTH_Y, CORNER_RADIUS
    corners = [((ax, ay), 90, 0), ((ax, -ay), 0, -90), ((-ax, -ay), -90, -180), ((-ax, ay), -180, -270)]
    points = []
    for center, start_deg, end_deg in corners:
        for k in range(ARC_SEGMENTS + 1):
            t = math.radians(start_deg + (end_deg - start_deg) * k / ARC_SEGMENTS)
            points.append((center[0] + r * math.cos(t), center[1] + r * math.sin(t)))
    shift = (0.0, ay + r)
    points = [(x + shift[0], y + shift[1]) for x, y in points]
    points = list(reversed(points))  # clockwise, matching car_race_vec_env
    cumulative = [0.0]
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        cumulative.append(cumulative[-1] + math.hypot(x1 - x0, y1 - y0))
    return points, cumulative


class UnknownTrackVecEnv(VecEnv):
    def __init__(self, num_pairs=NUM_PAIRS, headless=True, stagger_prob=0.0):
        get_simulation_app(headless=headless)

        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import open_stage

        # num_pairs=1 means the single-pair deploy stage (bare /World/CarA,
        # /World/CarB - matches build_unknown_track_deploy.py, used for demo
        # and the zero-shot generalization eval); anything else means the
        # multi-pair training stage (/World/Pair{i}/CarA - matches
        # build_unknown_track_train.py, used for actual continued training).
        if num_pairs == 1:
            open_stage(DEPLOY_STAGE_PATH)
            chassis_paths = ["/World/CarA/chassis", "/World/CarB/chassis"]
            pair_offsets = np.array([(0.0, 0.0)], dtype=np.float64)
        else:
            open_stage(TRAIN_STAGE_PATH)
            chassis_paths = []
            for i in range(num_pairs):
                chassis_paths.append(f"/World/Pair{i}/CarA/chassis")
                chassis_paths.append(f"/World/Pair{i}/CarB/chassis")
            pair_offsets = np.array(
                [((i % GRID_COLS) * GRID_SPACING, (i // GRID_COLS) * GRID_SPACING) for i in range(num_pairs)],
                dtype=np.float64,
            )

        self._world = World(stage_units_in_meters=1.0)
        self._world.reset()

        self.num_pairs = num_pairs
        num_envs = num_pairs * 2

        self._car = Articulation(prim_paths_expr=chassis_paths)
        self._car.initialize()

        self._offsets = np.repeat(pair_offsets, 2, axis=0)

        self._points, self._cumulative = build_centerline()
        self._track_length = self._cumulative[-1]
        self._n_segments = len(self._points)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self._step_count = np.zeros(num_pairs, dtype=np.int64)
        self._last_progress = np.zeros(num_envs, dtype=np.float64)
        self._last_opp_gap = np.zeros(num_envs, dtype=np.float64)
        self._last_action = np.zeros((num_envs, 2), dtype=np.float32)
        self._rng = np.random.default_rng()
        self._stagger_prob = float(stagger_prob)
        self._actions = None

        self._world.play()
        self._reset_pairs(np.arange(num_pairs))

    def _nearest_segment(self, x, y):
        best_i, best_d2, best_t = 0, float("inf"), 0.0
        for i in range(self._n_segments):
            x0, y0 = self._points[i]
            x1, y1 = self._points[(i + 1) % self._n_segments]
            dx, dy = x1 - x0, y1 - y0
            seg_len2 = dx * dx + dy * dy
            t = ((x - x0) * dx + (y - y0) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
            px, py = x0 + t * dx, y0 + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2, best_i, best_t = d2, i, t
        return best_i, best_t, math.sqrt(best_d2)

    def _tangent_at_progress(self, s):
        s = s % self._track_length
        for i in range(self._n_segments):
            if self._cumulative[i] <= s <= self._cumulative[i + 1]:
                x0, y0 = self._points[i]
                x1, y1 = self._points[(i + 1) % self._n_segments]
                return math.atan2(y1 - y0, x1 - x0)
        x0, y0 = self._points[-1]
        x1, y1 = self._points[0]
        return math.atan2(y1 - y0, x1 - x0)

    def _pose_at_progress(self, s):
        s = s % self._track_length
        for i in range(self._n_segments):
            if self._cumulative[i] <= s <= self._cumulative[i + 1]:
                break
        x0, y0 = self._points[i]
        x1, y1 = self._points[(i + 1) % self._n_segments]
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        t = (s - self._cumulative[i]) / seg_len
        return x0 + t * dx, y0 + t * dy, math.atan2(dy, dx), -dy / seg_len, dx / seg_len

    def _own_state(self, idx, pos, quat, lin_vel, ang_vel):
        lx, ly = pos[0] - self._offsets[idx, 0], pos[1] - self._offsets[idx, 1]
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        i, t, dist = self._nearest_segment(lx, ly)
        x0, y0 = self._points[i]
        x1, y1 = self._points[(i + 1) % self._n_segments]
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        tangent_angle = math.atan2(dy, dx)
        px, py = x0 + t * dx, y0 + t * dy
        side = math.copysign(1.0, dx * (ly - py) - dy * (lx - px))
        lateral_offset = dist * side
        progress = self._cumulative[i] + t * seg_len
        heading_error = wrap_to_pi(yaw - tangent_angle)

        tangent_ahead = self._tangent_at_progress(progress + LOOKAHEAD_DIST)
        curvature_ahead = wrap_to_pi(tangent_ahead - tangent_angle)

        forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
        yaw_rate = ang_vel[2]

        partial_obs = np.array(
            [lateral_offset, math.sin(heading_error), math.cos(heading_error),
             forward_speed, yaw_rate, math.sin(curvature_ahead), math.cos(curvature_ahead)],
            dtype=np.float32,
        )
        return partial_obs, progress, lateral_offset, forward_speed

    def _reset_pairs(self, pair_indices):
        if len(pair_indices) == 0:
            return
        slot_indices = []
        positions = []
        orientations = []
        for pi in pair_indices:
            seg = int(self._rng.integers(0, self._n_segments))
            x0, y0 = self._points[seg]
            x1, y1 = self._points[(seg + 1) % self._n_segments]
            dx, dy = x1 - x0, y1 - y0
            seg_len = math.hypot(dx, dy)
            tangent_angle = math.atan2(dy, dx)
            nx, ny = -dy / seg_len, dx / seg_len
            yaw = tangent_angle + float(self._rng.uniform(-0.1, 0.1))
            half_yaw = yaw / 2.0
            quat = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]

            slot_a, slot_b = 2 * pi, 2 * pi + 1
            ox, oy = self._offsets[slot_a]
            pos_a = [x0 + nx * START_LATERAL_SPACING + ox, y0 + ny * START_LATERAL_SPACING + oy, CHASSIS_Z]
            pos_b = [x0 - nx * START_LATERAL_SPACING + ox, y0 - ny * START_LATERAL_SPACING + oy, CHASSIS_Z]
            quat_a = quat_b = quat

            # 2026-09-24: staggered start (opt-in via stagger_prob, used by
            # train_unknown_track.py). Every earlier round started the two cars
            # exactly side by side, so the policy never had to overtake and
            # converged to a follow-the-leader lock-in (the car a few tenths of
            # a metre ahead after the first corner stays ~13 m ahead). Here the
            # trailing car starts STAGGER_MIN..STAGGER_MAX metres behind the
            # leader along the centerline, on a random side; which car leads
            # and which side each takes are both random.
            if self._stagger_prob > 0.0 and float(self._rng.random()) < self._stagger_prob:
                gap = float(self._rng.uniform(STAGGER_MIN, STAGGER_MAX))
                a_leads = bool(self._rng.random() < 0.5)
                a_side = 1.0 if self._rng.random() < 0.5 else -1.0
                s_lead = self._cumulative[seg]
                lead = self._pose_at_progress(s_lead)
                trail = self._pose_at_progress(s_lead - gap)
                pose_a, pose_b = (lead, trail) if a_leads else (trail, lead)
                quats = []
                poss = []
                for pose, side in ((pose_a, a_side), (pose_b, -a_side)):
                    px, py, tang, pnx, pny = pose
                    poss.append([px + pnx * side * START_LATERAL_SPACING + ox,
                                 py + pny * side * START_LATERAL_SPACING + oy, CHASSIS_Z])
                    half = (tang + float(self._rng.uniform(-0.1, 0.1))) / 2.0
                    quats.append([math.cos(half), 0.0, 0.0, math.sin(half)])
                pos_a, pos_b = poss
                quat_a, quat_b = quats

            slot_indices.extend([slot_a, slot_b])
            positions.extend([pos_a, pos_b])
            orientations.extend([quat_a, quat_b])

        idx_arr = np.array(slot_indices)
        self._car.set_world_poses(
            positions=np.array(positions), orientations=np.array(orientations), indices=idx_arr
        )
        self._car.set_velocities(np.zeros((len(idx_arr), 6)), indices=idx_arr)
        num_dof = self._car.num_dof
        self._car.set_joint_velocities(np.zeros((len(idx_arr), num_dof)), indices=idx_arr)
        self._car.set_joint_positions(np.zeros((len(idx_arr), num_dof)), indices=idx_arr)

        for _ in range(2):
            self._world.step(render=False)

        for pi in pair_indices:
            self._step_count[pi] = 0

    def _full_obs_and_progress(self):
        positions, orientations = self._car.get_world_poses()
        lin_vels = self._car.get_linear_velocities()
        ang_vels = self._car.get_angular_velocities()

        partials, progresses, laterals, speeds = [], [], [], []
        for i in range(self.num_envs):
            p, prog, lat, spd = self._own_state(i, positions[i], orientations[i], lin_vels[i], ang_vels[i])
            partials.append(p)
            progresses.append(prog)
            laterals.append(lat)
            speeds.append(spd)

        obs = np.zeros((self.num_envs, 10), dtype=np.float32)
        for pi in range(self.num_pairs):
            a, b = 2 * pi, 2 * pi + 1
            gap_ab = progresses[b] - progresses[a]
            if gap_ab > self._track_length / 2.0:
                gap_ab -= self._track_length
            elif gap_ab < -self._track_length / 2.0:
                gap_ab += self._track_length
            gap_ab = float(np.clip(gap_ab, -OPPONENT_GAP_CLIP, OPPONENT_GAP_CLIP))

            obs[a] = np.concatenate([partials[a], [gap_ab, laterals[b] - laterals[a], speeds[b] - speeds[a]]])
            obs[b] = np.concatenate([partials[b], [-gap_ab, laterals[a] - laterals[b], speeds[a] - speeds[b]]])

        return obs, np.array(progresses), np.array(laterals), positions

    def reset(self):
        self._reset_pairs(np.arange(self.num_pairs))
        obs, progresses, _, _ = self._full_obs_and_progress()
        self._last_progress = progresses
        self._last_opp_gap = obs[:, 7].astype(np.float64).copy()
        self._last_action = np.zeros((self.num_envs, 2), dtype=np.float32)
        return obs

    def step_async(self, actions):
        self._actions = np.clip(np.asarray(actions), -1.0, 1.0)

    def step_wait(self):
        actions = self._actions
        throttle = actions[:, 0]
        steer = actions[:, 1]

        vel_cmd = np.tile((throttle * MAX_WHEEL_SPEED).reshape(-1, 1), (1, len(DRIVE_JOINTS)))
        self._car.set_joint_velocity_targets(vel_cmd, joint_names=DRIVE_JOINTS)
        pos_cmd = np.tile((steer * MAX_STEER_ANGLE).reshape(-1, 1), (1, len(STEER_JOINTS)))
        self._car.set_joint_position_targets(pos_cmd, joint_names=STEER_JOINTS)

        for _ in range(ACTION_REPEAT):
            self._world.step(render=False)

        obs, progresses, laterals, positions = self._full_obs_and_progress()

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        pair_done = np.zeros(self.num_pairs, dtype=bool)
        pair_truncated = np.zeros(self.num_pairs, dtype=bool)

        for i in range(self.num_envs):
            delta = progresses[i] - self._last_progress[i]
            if delta > self._track_length / 2.0:
                delta -= self._track_length
            elif delta < -self._track_length / 2.0:
                delta += self._track_length
            reward = delta - LATERAL_PENALTY_WEIGHT * abs(laterals[i]) + THROTTLE_BONUS_WEIGHT * max(0.0, float(throttle[i]))
            overtake_gain = self._last_opp_gap[i] - float(obs[i][7])
            reward += OVERTAKE_INCENTIVE_WEIGHT * overtake_gain
            action_delta = abs(float(throttle[i]) - self._last_action[i, 0]) + abs(float(steer[i]) - self._last_action[i, 1])
            reward -= SMOOTHNESS_PENALTY_WEIGHT * action_delta
            self._last_action[i] = [throttle[i], steer[i]]
            off_track = abs(laterals[i]) > OFF_TRACK_MARGIN
            if off_track:
                reward -= OFF_TRACK_PENALTY
                pair_done[i // 2] = True
            rewards[i] = reward
        self._last_progress = progresses

        for pi in range(self.num_pairs):
            a, b = 2 * pi, 2 * pi + 1
            dist = math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1])
            if dist < COLLISION_DIST:
                rewards[a] -= COLLISION_PENALTY
                rewards[b] -= COLLISION_PENALTY

            self._step_count[pi] += 1
            if self._step_count[pi] >= MAX_EPISODE_STEPS:
                pair_truncated[pi] = True
                pair_done[pi] = True

        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        reset_pairs = []
        for pi in range(self.num_pairs):
            if pair_done[pi]:
                a, b = 2 * pi, 2 * pi + 1
                dones[a] = dones[b] = True
                infos[a]["TimeLimit.truncated"] = bool(pair_truncated[pi])
                infos[b]["TimeLimit.truncated"] = bool(pair_truncated[pi])
                infos[a]["terminal_observation"] = obs[a].copy()
                infos[b]["terminal_observation"] = obs[b].copy()
                reset_pairs.append(pi)

        if reset_pairs:
            self._reset_pairs(np.array(reset_pairs))
            obs2, progresses2, _, _ = self._full_obs_and_progress()
            for pi in reset_pairs:
                a, b = 2 * pi, 2 * pi + 1
                obs[a], obs[b] = obs2[a], obs2[b]
                self._last_progress[a], self._last_progress[b] = progresses2[a], progresses2[b]
                self._last_action[a] = 0.0
                self._last_action[b] = 0.0

        self._last_opp_gap = obs[:, 7].astype(np.float64).copy()

        return obs, rewards, dones, infos

    def close(self):
        if _simulation_app is not None:
            _simulation_app.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed] * self.num_envs
