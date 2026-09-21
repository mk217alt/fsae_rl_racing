"""Trains SimpleCar2 to drive around the track via PPO (stable-baselines3),
learning purely by trial and error (no human demonstrations).

Usage: python.bat train_car2.py [total_timesteps]
"""

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from car_track_env import CarTrackEnv

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Single_Car_Baseline/car2_ppo_model"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/Single_Car_Baseline/car2_train_monitor.csv"

env = Monitor(CarTrackEnv(headless=True), filename=LOG_PATH)

if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, env=env, device="cpu")
    print(f"Resumed training from existing checkpoint at {MODEL_PATH}.zip")
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        device="cpu",
    )

model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
model.save(MODEL_PATH)

with open("C:/Users/sanja/Desktop/thesis/train_car2_done.txt", "w") as f:
    f.write(f"trained {TOTAL_TIMESTEPS} timesteps, saved to {MODEL_PATH}.zip\n")

env.close()
