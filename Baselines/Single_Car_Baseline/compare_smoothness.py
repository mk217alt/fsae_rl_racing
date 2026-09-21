"""Compares action smoothness between the pre- and post-smoothness-reward
checkpoints of the single-car baseline, by running each deterministically
(no exploration noise) and measuring average consecutive-action change -
the direct metric the smoothness reward term was designed to reduce.

Usage: python.bat compare_smoothness.py
"""

import sys

import numpy as np
from stable_baselines3 import PPO

from car_track_env import CarTrackEnv

RESULTS_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/Single_Car_Baseline/compare_smoothness_results.txt"

N_EPISODES = 10

CHECKPOINTS = {
    "pre_smoothness": "C:/Users/sanja/Desktop/thesis/checkpoint_backups/car2_ppo_model_pre_smoothness",
    "post_smoothness": "C:/Users/sanja/Desktop/thesis/Baselines/Single_Car_Baseline/car2_ppo_model",
}

env = CarTrackEnv(headless=True)

out = open(RESULTS_PATH, "w")


def log(msg):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()


results = {}
for label, path in CHECKPOINTS.items():
    model = PPO.load(path, device="cpu")
    deltas = []
    full_length_count = 0
    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        prev_action = None
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            if prev_action is not None:
                deltas.append(float(abs(action[0] - prev_action[0]) + abs(action[1] - prev_action[1])))
            prev_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        if steps >= 500:
            full_length_count += 1
    results[label] = {
        "mean_action_delta": float(np.mean(deltas)),
        "std_action_delta": float(np.std(deltas)),
        "full_length_rate": full_length_count / N_EPISODES,
        "n_steps_measured": len(deltas),
    }
    log(
        f"{label}: mean_delta={results[label]['mean_action_delta']:.4f} "
        f"std={results[label]['std_action_delta']:.4f} "
        f"full_length_rate={results[label]['full_length_rate'] * 100:.1f}% "
        f"n_steps={results[label]['n_steps_measured']}"
    )

pre = results["pre_smoothness"]["mean_action_delta"]
post = results["post_smoothness"]["mean_action_delta"]
change = (pre - post) / pre * 100
log(f"Smoothness change: {change:.1f}% {'reduction' if change > 0 else 'increase'} in mean consecutive-action delta")
out.close()
sys.stdout.flush()

env.close()
