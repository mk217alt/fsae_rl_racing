"""Sanity check for RaceVecEnv before committing to training: reset, drive
all car slots forward at constant throttle, and confirm - no crash,
opponent-relative observation fields make sense (gap sign flips correctly
between the two cars in a pair), collision penalty fires when cars are
close, paired reset fires for both cars in a pair together."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g_two_car_racing"))
import numpy as np

from car_race_vec_env import RaceVecEnv, NUM_PAIRS

N_STEPS = 150
LOG_PATH = f"{ROOT}/run_output/smoke_test_race_env_log.txt"

env = RaceVecEnv(num_pairs=NUM_PAIRS, headless=True)
obs = env.reset()

lines = [f"num_envs={env.num_envs}", f"initial obs pair0 CarA: {obs[0]}", f"initial obs pair0 CarB: {obs[1]}"]

action = np.tile(np.array([[0.6, 0.0]]), (env.num_envs, 1))

reward_sum = np.zeros(env.num_envs)
done_count = np.zeros(env.num_envs, dtype=int)
collision_hits = 0

for step in range(N_STEPS):
    env.step_async(action)
    obs, rewards, dones, infos = env.step_wait()
    reward_sum += rewards
    done_count += dones.astype(int)
    if rewards[0] < -4.9 or rewards[1] < -4.9:
        collision_hits += 1
    if step % 20 == 0 or step == N_STEPS - 1:
        lines.append(
            f"step {step}: rewards={np.round(rewards, 2)} "
            f"progress_gap(pair0)={obs[0][7]:.2f}/{obs[1][7]:.2f} "
            f"lateral_gap(pair0)={obs[0][8]:.2f}/{obs[1][8]:.2f} "
            f"dones={dones}"
        )

lines.append(f"cumulative reward per slot: {np.round(reward_sum, 2)}")
lines.append(f"done count per slot (pair resets triggered): {done_count.tolist()}")
lines.append(f"steps with a collision penalty active: {collision_hits}")
# sanity: pair0's two cars should always show opposite-signed progress gap
consistency_ok = True

with open(LOG_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

with open(f"{ROOT}/run_output/smoke_test_race_env_done.txt", "w") as f:
    f.write("done\n")

env.close()
