"""Trains car2's policy using the vectorized, curvature-lookahead environment
(car_track_vec_env.py) - N car+track clones stepped together in one Isaac
Sim process. Separate lineage from the single-env car2_ppo_model.zip: the
7-dim observation (vs. 5-dim) means this can't resume from that checkpoint,
by design - a fresh network, trained faster per wall-clock hour via the
parallel envs, compared against the accepted single-env result afterward.

Usage: python.bat train_car2_vec.py [total_timesteps]
"""

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from car_track_vec_env import CarTrackVecEnv, NUM_ENVS

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Vec_Curvature_Lineage/car2_vec_ppo_model"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Vec_Curvature_Lineage/car2_vec_train_monitor.csv"

env = VecMonitor(CarTrackVecEnv(num_envs=NUM_ENVS, headless=True), filename=LOG_PATH)

if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, env=env, device="cpu")
    print(f"Resumed training from existing checkpoint at {MODEL_PATH}.zip")
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        device="cpu",
    )

model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
model.save(MODEL_PATH)

with open("C:/Users/sanja/Desktop/thesis/train_car2_vec_done.txt", "w") as f:
    f.write(f"trained {TOTAL_TIMESTEPS} timesteps, saved to {MODEL_PATH}.zip\n")

env.close()
