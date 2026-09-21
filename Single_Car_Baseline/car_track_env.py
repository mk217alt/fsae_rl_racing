"""A Gymnasium environment that trains SimpleCar2 to drive itself around the
track via trial-and-error reinforcement learning (no human demonstrations).

Observation (5,): [lateral_offset, sin(heading_error), cos(heading_error),
                    forward_speed, yaw_rate]
Action (2,):       [throttle, steer] in [-1, 1], same convention as drive_car.py

Reward: progress made along the track centerline since the last step, minus
a penalty for drifting away from the centerline. Going off the road (beyond
half the track width) ends the episode with a penalty.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from isaacsim import SimulationApp

_simulation_app = None


def get_simulation_app(headless=True):
    global _simulation_app
    if _simulation_app is None:
        _simulation_app = SimulationApp({"headless": headless})
    return _simulation_app


# ---------------------------------------------------------------------------
# track geometry (must match create_track.py)
# ---------------------------------------------------------------------------
HALF_STRAIGHT = 8.0
CORNER_RADIUS = 10.0
TRACK_WIDTH = 9.0
ARC_SEGMENTS = 10

CAR_USD = "C:/Users/sanja/Desktop/thesis/Simulation_Environment_Setup/simple_car.usd"
CAR_ROOT = "/World/SimpleCar2/chassis"
DRIVE_JOINTS = ["front_left_drive_joint", "front_right_drive_joint"]
STEER_JOINTS = ["front_left_steer_joint", "front_right_steer_joint"]

MAX_WHEEL_SPEED = 20.0
MAX_STEER_ANGLE = math.radians(25.0)
CHASSIS_Z = 0.45
LATERAL_OFFSET = 3.0  # matches add_second_car.py

ACTION_REPEAT = 4
MAX_EPISODE_STEPS = 500
OFF_TRACK_MARGIN = TRACK_WIDTH / 2.0 - 0.3
# Priority 1: hug the centerline - a strong, ongoing penalty for drifting off
# center plus a heavy one-off penalty for actually leaving the road.
LATERAL_PENALTY_WEIGHT = 0.6
OFF_TRACK_PENALTY = 30.0
# Priority 2: accelerate at max throttle - scales with throttle so full
# throttle earns the full bonus; still small enough that the centering/
# off-track penalties above dominate if going faster would risk leaving the road.
THROTTLE_BONUS_WEIGHT = 0.25
# 2026-09-21: smoothness experiment, at user's request to address the
# professor's "smooth driving" milestone item - penalize jerky consecutive
# steer/throttle changes. Kept modest (0.08) relative to the per-step
# progress/centering terms above (which are the dominant, already-converged
# signal) so this nudges toward smoother control without fighting the
# existing reward balance or risking a repeat of the collision-penalty-style
# runaway-suppression pathology seen in the two-car lineage when a new
# per-step penalty was introduced too aggressively.
SMOOTHNESS_PENALTY_WEIGHT = 0.08


def _arc_points(center, radius, start_deg, end_deg, n):
    pts = []
    for k in range(n + 1):
        t = math.radians(start_deg + (end_deg - start_deg) * k / n)
        pts.append((center[0] + radius * math.cos(t), center[1] + radius * math.sin(t)))
    return pts


def build_centerline():
    a, r = HALF_STRAIGHT, CORNER_RADIUS
    corners = [
        ((a, a), 90, 0),
        ((a, -a), 0, -90),
        ((-a, -a), -90, -180),
        ((-a, a), -180, -270),
    ]
    points = []
    for center, start_deg, end_deg in corners:
        points.extend(_arc_points(center, r, start_deg, end_deg, ARC_SEGMENTS))
    shift = (0.0, a + r)
    points = [(x + shift[0], y + shift[1] + LATERAL_OFFSET) for x, y in points]

    cumulative = [0.0]
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        cumulative.append(cumulative[-1] + math.hypot(x1 - x0, y1 - y0))
    return points, cumulative


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class CarTrackEnv(gym.Env):
    def __init__(self, headless=True):
        get_simulation_app(headless=headless)

        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import open_stage

        open_stage(CAR_USD)
        self._world = World(stage_units_in_meters=1.0)
        self._world.reset()

        self._car = Articulation(prim_paths_expr=CAR_ROOT)
        self._car.initialize()

        self._points, self._cumulative = build_centerline()
        self._track_length = self._cumulative[-1]
        self._n_segments = len(self._points)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self._step_count = 0
        self._last_progress = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)
        self._rng = np.random.default_rng()

        self._world.play()

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

    def _progress_and_offset(self, x, y):
        i, t, dist = self._nearest_segment(x, y)
        x0, y0 = self._points[i]
        x1, y1 = self._points[(i + 1) % self._n_segments]
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        tangent_angle = math.atan2(dy, dx)
        # signed lateral offset: which side of the tangent the car is on
        px, py = x0 + t * dx, y0 + t * dy
        side = math.copysign(1.0, dx * (y - py) - dy * (x - px))
        progress = self._cumulative[i] + t * seg_len
        return progress, dist * side, tangent_angle

    def _get_obs_and_state(self):
        positions, orientations = self._car.get_world_poses()
        pos = positions[0]
        quat = orientations[0]  # (w, x, y, z)
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        progress, lateral_offset, tangent_angle = self._progress_and_offset(pos[0], pos[1])
        heading_error = wrap_to_pi(yaw - tangent_angle)

        lin_vel = self._car.get_linear_velocities()[0]
        ang_vel = self._car.get_angular_velocities()[0]
        forward_speed = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
        yaw_rate = ang_vel[2]

        obs = np.array(
            [lateral_offset, math.sin(heading_error), math.cos(heading_error),
             forward_speed, yaw_rate],
            dtype=np.float32,
        )
        return obs, progress, lateral_offset

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        i = int(self._rng.integers(0, self._n_segments))
        x0, y0 = self._points[i]
        x1, y1 = self._points[(i + 1) % self._n_segments]
        # face the same direction the point list travels in, so that driving
        # forward (car's own +X) corresponds to increasing progress
        tangent_angle = math.atan2(y1 - y0, x1 - x0)

        px, py = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        yaw = tangent_angle + float(self._rng.uniform(-0.1, 0.1))

        half_yaw = yaw / 2.0
        orientation = np.array([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
        position = np.array([[px, py, CHASSIS_Z]])

        self._car.set_world_poses(positions=position, orientations=orientation)
        self._car.set_velocities(np.zeros((1, 6)))
        self._car.set_joint_velocities(np.zeros((1, self._car.num_dof)))
        self._car.set_joint_positions(np.zeros((1, self._car.num_dof)))

        for _ in range(2):
            self._world.step(render=False)

        obs, progress, _ = self._get_obs_and_state()
        self._last_progress = progress
        self._last_action = np.zeros(2, dtype=np.float32)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        throttle, steer = float(action[0]), float(action[1])

        vel_cmd = np.full((1, len(DRIVE_JOINTS)), throttle * MAX_WHEEL_SPEED)
        self._car.set_joint_velocity_targets(vel_cmd, joint_names=DRIVE_JOINTS)
        pos_cmd = np.full((1, len(STEER_JOINTS)), steer * MAX_STEER_ANGLE)
        self._car.set_joint_position_targets(pos_cmd, joint_names=STEER_JOINTS)

        for _ in range(ACTION_REPEAT):
            self._world.step(render=False)

        obs, progress, lateral_offset = self._get_obs_and_state()

        delta = progress - self._last_progress
        if delta > self._track_length / 2.0:
            delta -= self._track_length
        elif delta < -self._track_length / 2.0:
            delta += self._track_length
        self._last_progress = progress

        reward = delta - LATERAL_PENALTY_WEIGHT * abs(lateral_offset)
        reward += THROTTLE_BONUS_WEIGHT * max(0.0, throttle)

        action_delta = abs(throttle - self._last_action[0]) + abs(steer - self._last_action[1])
        reward -= SMOOTHNESS_PENALTY_WEIGHT * action_delta
        self._last_action = np.array([throttle, steer], dtype=np.float32)

        self._step_count += 1
        off_track = abs(lateral_offset) > OFF_TRACK_MARGIN
        terminated = bool(off_track)
        if off_track:
            reward -= OFF_TRACK_PENALTY
        truncated = self._step_count >= MAX_EPISODE_STEPS

        return obs, reward, terminated, truncated, {}

    def close(self):
        if _simulation_app is not None:
            _simulation_app.close()
