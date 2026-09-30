"""A vectorized version of car_track_env.py: N independent car+track clones
(built by build_vec_track_env.py) stepped together in ONE Isaac Sim process,
so PPO collects rollouts from all N cars per `world.step()` call instead of
needing N separate headless processes (which this machine can't run anyway -
only one kit.exe at a time without severe resource contention).

Same reward and dynamics as the single-env version, with one addition to
the observation: curvature look-ahead, i.e. how much the track is about to
turn a fixed distance ahead of the car, so the policy can anticipate corners
instead of only reacting to its current lateral offset.

Observation (7,): [lateral_offset, sin(heading_error), cos(heading_error),
                    forward_speed, yaw_rate, sin(curvature_ahead), cos(curvature_ahead)]
Action (2,):       [throttle, steer] in [-1, 1], same convention as the single-env version.
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


STAGE_PATH = f"{ROOT}/1_simulation_scenes/vec_train_car2.usd"
NUM_ENVS = 10
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
LATERAL_PENALTY_WEIGHT = 0.6
OFF_TRACK_PENALTY = 30.0
THROTTLE_BONUS_WEIGHT = 0.25
LOOKAHEAD_DIST = 5.0  # meters ahead of the car to sample curvature from


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
    points = list(reversed(points))  # clockwise direction of travel
    cumulative = [0.0]
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        cumulative.append(cumulative[-1] + math.hypot(x1 - x0, y1 - y0))
    return points, cumulative


class CarTrackVecEnv(VecEnv):
    def __init__(self, num_envs=NUM_ENVS, headless=True):
        get_simulation_app(headless=headless)

        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import open_stage

        open_stage(STAGE_PATH)
        self._world = World(stage_units_in_meters=1.0)
        self._world.reset()

        chassis_paths = [f"/World/Env{i}/Car/chassis" for i in range(num_envs)]
        self._car = Articulation(prim_paths_expr=chassis_paths)
        self._car.initialize()

        self._offsets = np.array(
            [((i % GRID_COLS) * GRID_SPACING, (i // GRID_COLS) * GRID_SPACING) for i in range(num_envs)],
            dtype=np.float64,
        )

        self._points, self._cumulative = build_centerline()
        self._track_length = self._cumulative[-1]
        self._n_segments = len(self._points)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self._step_count = np.zeros(num_envs, dtype=np.int64)
        self._last_progress = np.zeros(num_envs, dtype=np.float64)
        self._rng = np.random.default_rng()
        self._actions = None

        self._world.play()
        self._reset_envs(np.arange(num_envs))

    # -- centerline geometry, operating in per-clone LOCAL coordinates --------
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

    def _obs_for_env(self, idx, pos, quat, lin_vel, ang_vel):
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

        obs = np.array(
            [lateral_offset, math.sin(heading_error), math.cos(heading_error),
             forward_speed, yaw_rate, math.sin(curvature_ahead), math.cos(curvature_ahead)],
            dtype=np.float32,
        )
        return obs, progress, lateral_offset

    def _reset_envs(self, indices):
        if len(indices) == 0:
            return
        idx_arr = np.array(indices)
        positions = np.zeros((len(indices), 3))
        orientations = np.zeros((len(indices), 4))
        for k, idx in enumerate(indices):
            seg = int(self._rng.integers(0, self._n_segments))
            x0, y0 = self._points[seg]
            x1, y1 = self._points[(seg + 1) % self._n_segments]
            tangent_angle = math.atan2(y1 - y0, x1 - x0)
            px, py = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            yaw = tangent_angle + float(self._rng.uniform(-0.1, 0.1))
            half_yaw = yaw / 2.0
            orientations[k] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
            positions[k] = [px + self._offsets[idx, 0], py + self._offsets[idx, 1], CHASSIS_Z]

        self._car.set_world_poses(positions=positions, orientations=orientations, indices=idx_arr)
        self._car.set_velocities(np.zeros((len(indices), 6)), indices=idx_arr)
        num_dof = self._car.num_dof
        self._car.set_joint_velocities(np.zeros((len(indices), num_dof)), indices=idx_arr)
        self._car.set_joint_positions(np.zeros((len(indices), num_dof)), indices=idx_arr)

        for _ in range(2):
            self._world.step(render=False)

        for idx in indices:
            self._step_count[idx] = 0

    def reset(self):
        self._reset_envs(np.arange(self.num_envs))
        positions, orientations = self._car.get_world_poses()
        lin_vels = self._car.get_linear_velocities()
        ang_vels = self._car.get_angular_velocities()
        obs = np.zeros((self.num_envs, 7), dtype=np.float32)
        for i in range(self.num_envs):
            o, progress, _ = self._obs_for_env(i, positions[i], orientations[i], lin_vels[i], ang_vels[i])
            obs[i] = o
            self._last_progress[i] = progress
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

        positions, orientations = self._car.get_world_poses()
        lin_vels = self._car.get_linear_velocities()
        ang_vels = self._car.get_angular_velocities()

        obs = np.zeros((self.num_envs, 7), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        reset_indices = []

        for i in range(self.num_envs):
            o, progress, lateral_offset = self._obs_for_env(i, positions[i], orientations[i], lin_vels[i], ang_vels[i])

            delta = progress - self._last_progress[i]
            if delta > self._track_length / 2.0:
                delta -= self._track_length
            elif delta < -self._track_length / 2.0:
                delta += self._track_length
            self._last_progress[i] = progress

            reward = delta - LATERAL_PENALTY_WEIGHT * abs(lateral_offset)
            reward += THROTTLE_BONUS_WEIGHT * max(0.0, float(throttle[i]))

            self._step_count[i] += 1
            off_track = abs(lateral_offset) > OFF_TRACK_MARGIN
            terminated = bool(off_track)
            if off_track:
                reward -= OFF_TRACK_PENALTY
            truncated = self._step_count[i] >= MAX_EPISODE_STEPS
            done = terminated or truncated

            obs[i] = o
            rewards[i] = reward
            dones[i] = done
            if done:
                infos[i]["TimeLimit.truncated"] = truncated and not terminated
                infos[i]["terminal_observation"] = o.copy()
                reset_indices.append(i)

        if reset_indices:
            self._reset_envs(np.array(reset_indices))
            positions2, orientations2 = self._car.get_world_poses()
            for i in reset_indices:
                o2, progress2, _ = self._obs_for_env(
                    i, positions2[i], orientations2[i], np.zeros(3), np.zeros(3)
                )
                obs[i] = o2
                self._last_progress[i] = progress2

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
