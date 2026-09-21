"""Two-car self-play racing environment: N independent track-pair clones
(built by build_race_track_env.py), each hosting two cars sharing one
track. A SINGLE shared PPO policy controls every car slot (self-play) -
every car's experience, whichever pair or slot it comes from, trains the
same network.

Observation (10,): the single-car curvature-aware observation (7,) plus
3 opponent-relative features so each car is aware of the other:
  [lateral_offset, sin(heading_error), cos(heading_error), forward_speed,
   yaw_rate, sin(curvature_ahead), cos(curvature_ahead),
   opponent_progress_gap, opponent_lateral_gap, opponent_relative_speed]
Action (2,): [throttle, steer], same convention as the single-car work.

Reward: the same per-car progress-based reward as car_track_vec_env.py,
plus a small shared penalty when the two cars in a pair get too close -
without it there's no pressure to avoid just barrelling through each other.

A pair's two cars reset TOGETHER (as one race) whenever EITHER one ends
its episode (off-track or 500-step truncation) - keeps the opponent-relative
features well-defined instead of one car mid-race suddenly seeing a
teleported opponent.
"""

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


STAGE_PATH = "C:/Users/sanja/Desktop/thesis/Two_Car_Self_Play_Racing/race_train.usd"
NUM_PAIRS = 8
GRID_COLS = 3
GRID_SPACING = 60.0

HALF_STRAIGHT = 8.0
CORNER_RADIUS = 10.0
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
# 2026-09-19: raised centering weight / off-track penalty, lowered throttle
# bonus - user wants this lineage to prioritize staying on track over raw
# speed (opposite direction from the racing-line variant's looser weights).
# Raised again 0.9->1.2 alongside the training-stage ground-plane fix (run15)
# but that combined change regressed and 2M steps of continued training
# failed to recover it (see car_racing_status.md). Reverted back to 0.9 so
# the policy only has to re-adapt to the ground-plane physics change alone -
# the higher weight can be reintroduced separately once that recovers.
LATERAL_PENALTY_WEIGHT = 0.9
# 2026-09-19: raised to max (40->150) at user's explicit request to punish
# off-track drift as hard as possible. Safe to push this far since it's a
# one-time terminal penalty (paid once when an episode ends off-track), not
# a per-step penalty - no risk of the runaway-suppression pathology seen
# with the earlier oversized per-step collision penalty. Note: the last
# controlled measurement already put off-track recoveries at ~0-0.5%, so
# there's very little room left for this to visibly move - if the demo
# still "drifts", it's most likely the untrainable chassis-collapse bug
# (~97-98% of resets), not lateral off-track drift.
OFF_TRACK_PENALTY = 150.0
THROTTLE_BONUS_WEIGHT = 0.15
LOOKAHEAD_DIST = 5.0
START_LATERAL_SPACING = 1.5  # side-by-side starting grid offset
COLLISION_DIST = 1.2
# 2026-09-19: raised 0.4->0.8, then ->3.0 at user's explicit request for
# severe collision punishment (run19). NOTE this is a PER-STEP penalty
# (applied every physics step the cars stay within COLLISION_DIST), unlike
# OFF_TRACK_PENALTY's one-time terminal payout - a documented earlier value
# of 5.0 already broke training via a runaway-suppression pathology (policy
# learns to avoid all contact risk rather than race).
# 2026-09-20: 3.0 looked great for one round (run19) but then produced two
# consecutive rounds of real decline (run20 dipped late, run21 declined
# across every quartile and failed to set a new peak reward for the first
# time in the whole recovery arc) - a slower version of the same
# runaway-suppression pathology the 5.0 value caused outright. Backed off
# to 1.5 (still ~2x the pre-run19 value of 0.8, still clearly "severe", but
# pulling back from the value that was measurably hurting consolidation).
COLLISION_PENALTY = 1.5
OPPONENT_GAP_CLIP = 30.0  # meters, clip range for the progress-gap feature


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def build_centerline():
    a, r = HALF_STRAIGHT, CORNER_RADIUS
    corners = [((a, a), 90, 0), ((a, -a), 0, -90), ((-a, -a), -90, -180), ((-a, a), -180, -270)]
    points = []
    for center, start_deg, end_deg in corners:
        for k in range(ARC_SEGMENTS + 1):
            t = math.radians(start_deg + (end_deg - start_deg) * k / ARC_SEGMENTS)
            points.append((center[0] + r * math.cos(t), center[1] + r * math.sin(t)))
    shift = (0.0, a + r)
    points = [(x + shift[0], y + shift[1]) for x, y in points]
    points = list(reversed(points))  # clockwise, matching car_track_vec_env
    cumulative = [0.0]
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        cumulative.append(cumulative[-1] + math.hypot(x1 - x0, y1 - y0))
    return points, cumulative


class RaceVecEnv(VecEnv):
    def __init__(self, num_pairs=NUM_PAIRS, headless=True):
        get_simulation_app(headless=headless)

        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import open_stage

        open_stage(STAGE_PATH)
        self._world = World(stage_units_in_meters=1.0)
        self._world.reset()

        self.num_pairs = num_pairs
        num_envs = num_pairs * 2

        # slot layout: [pair0 CarA, pair0 CarB, pair1 CarA, pair1 CarB, ...]
        chassis_paths = []
        for i in range(num_pairs):
            chassis_paths.append(f"/World/Pair{i}/CarA/chassis")
            chassis_paths.append(f"/World/Pair{i}/CarB/chassis")
        self._car = Articulation(prim_paths_expr=chassis_paths)
        self._car.initialize()

        pair_offsets = np.array(
            [((i % GRID_COLS) * GRID_SPACING, (i // GRID_COLS) * GRID_SPACING) for i in range(num_pairs)],
            dtype=np.float64,
        )
        self._offsets = np.repeat(pair_offsets, 2, axis=0)  # per car-slot

        self._points, self._cumulative = build_centerline()
        self._track_length = self._cumulative[-1]
        self._n_segments = len(self._points)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self._step_count = np.zeros(num_pairs, dtype=np.int64)
        self._last_progress = np.zeros(num_envs, dtype=np.float64)
        self._rng = np.random.default_rng()
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

            slot_indices.extend([slot_a, slot_b])
            positions.extend([pos_a, pos_b])
            orientations.extend([quat, quat])

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
