"""Deterministic evaluation of the current two-car self-play checkpoint on
the TRAINED track (race_train.usd, pair 0 only) - the baseline half of the
unknown-track generalization test in test_unknown_track_generalization.py.
Run this first, then that script, then compare the two results files.

Usage: python.bat test_baseline_track_eval.py
"""

import sys

import numpy as np
from stable_baselines3 import PPO

from car_race_vec_env import RaceVecEnv

RESULTS_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/test_baseline_track_results.txt"
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/car_race_ppo_model"
N_PAIR_EPISODES = 15

env = RaceVecEnv(num_pairs=1, headless=True)
model = PPO.load(MODEL_PATH, device="cpu")

out = open(RESULTS_PATH, "w")


def log(msg):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()


log(f"Baseline (trained-track) eval, checkpoint={MODEL_PATH}.zip, {N_PAIR_EPISODES} pair-episodes")

obs = env.reset()
episode_rewards = [0.0, 0.0]
episode_lengths = [0, 0]
completed_rewards = []
completed_lengths = []
pair_episodes_done = 0

while pair_episodes_done < N_PAIR_EPISODES:
    actions, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(actions)
    for i in range(2):
        episode_rewards[i] += float(rewards[i])
        episode_lengths[i] += 1
    if dones[0] or dones[1]:
        for i in range(2):
            completed_rewards.append(episode_rewards[i])
            completed_lengths.append(episode_lengths[i])
            episode_rewards[i] = 0.0
            episode_lengths[i] = 0
        pair_episodes_done += 1

full_length_count = sum(1 for l in completed_lengths if l >= 500)
log(
    f"episodes={len(completed_lengths)} full_length_rate={100.0 * full_length_count / len(completed_lengths):.1f}% "
    f"mean_reward={float(np.mean(completed_rewards)):.1f} "
    f"peak_reward={float(np.max(completed_rewards)):.1f} "
    f"min_reward={float(np.min(completed_rewards)):.1f} "
    f"mean_length={float(np.mean(completed_lengths)):.1f}"
)

out.close()
sys.stdout.flush()
env.close()
