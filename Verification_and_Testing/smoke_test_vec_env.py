"""Sanity check for CarTrackVecEnv before committing to a full training run:
reset, then drive all N clones forward at constant throttle for a while,
logging obs/reward per env per step to a file (print() is swallowed by
Isaac Sim's own logging, so everything goes to a file as usual for this
project).

Checks: no crash, reward rises as speed builds (mirrors the original
single-env Phase 6 smoke test), curvature-ahead observation actually
changes value as each car goes around a corner, no cross-env contamination
(each env's position stays near its own grid offset).
"""


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Vec_Curvature_Lineage"))
import numpy as np

from car_track_vec_env import CarTrackVecEnv

N_STEPS = 150
LOG_PATH = "C:/Users/sanja/Desktop/thesis/smoke_test_vec_env_log.txt"

env = CarTrackVecEnv(num_envs=6, headless=True)
obs = env.reset()

lines = [f"initial obs (env0): {obs[0]}"]

action = np.tile(np.array([[0.6, 0.0]]), (6, 1))  # constant forward throttle, straight

reward_sum = np.zeros(6)
done_count = np.zeros(6, dtype=int)

for step in range(N_STEPS):
    env.step_async(action)
    obs, rewards, dones, infos = env.step_wait()
    reward_sum += rewards
    done_count += dones.astype(int)
    if step % 20 == 0 or step == N_STEPS - 1:
        lines.append(
            f"step {step}: rewards={np.round(rewards, 3)} "
            f"lateral_offsets={np.round(obs[:, 0], 3)} "
            f"curvature_ahead_sin={np.round(obs[:, 5], 3)} "
            f"dones={dones}"
        )

lines.append(f"cumulative reward per env: {np.round(reward_sum, 2)}")
lines.append(f"done count per env (resets triggered): {done_count.tolist()}")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

with open("C:/Users/sanja/Desktop/thesis/smoke_test_vec_env_done.txt", "w") as f:
    f.write("done\n")

env.close()
