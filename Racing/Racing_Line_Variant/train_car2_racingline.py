"""Trains the racing-line reward variant (car_track_racingline_vec_env.py):
same parallel single-car architecture as train_car2_vec.py, loosened
lateral-centering penalty, stronger throttle incentive. Separate lineage
(car2_racingline_ppo_model.zip) - the observation shape matches the
existing vec+curvature policy (7 dims), but the reward landscape differs
enough that this needs its own checkpoint, not a continuation.

Usage: python.bat train_car2_racingline.py [total_timesteps]
"""

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from car_track_racingline_vec_env import CarTrackRacinglineVecEnv, NUM_ENVS

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Racing_Line_Variant/car2_racingline_ppo_model"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Racing_Line_Variant/car2_racingline_train_monitor.csv"

env = VecMonitor(CarTrackRacinglineVecEnv(num_envs=NUM_ENVS, headless=True), filename=LOG_PATH)

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

with open("C:/Users/sanja/Desktop/thesis/train_car2_racingline_done.txt", "w") as f:
    f.write(f"trained {TOTAL_TIMESTEPS} timesteps, saved to {MODEL_PATH}.zip\n")

env.close()
