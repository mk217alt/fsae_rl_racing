"""Second attempt at RL fine-tuning from a behavioral-cloning start point.

v1 (finetune_bc.py) fine-tuned directly from car2_bc_pretrained.zip and
degraded over 50k steps (episode length 137 -> 81, reward never above -37)
because the critic had never seen any real returns and its bad early
advantage estimates dragged the good BC actor down before it could catch
up. This version starts from car2_bc_warmed.zip (produced by
warmup_critic.py), which has the same BC actor plus a critic pre-regressed
on real rollout returns, so PPO's first updates should be far less
destructive.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b_single_car"))
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from car_track_env import CarTrackEnv

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
WARMED_MODEL_PATH = f"{ROOT}/3_trained_models/car2_bc_warmed"
OUT_MODEL_PATH = f"{ROOT}/3_trained_models/car2_bc_finetuned_v2"
LOG_PATH = f"{ROOT}/6_training_logs/car2_bc_finetune_v2_monitor.csv"

env = Monitor(CarTrackEnv(headless=True), filename=LOG_PATH)

if os.path.exists(OUT_MODEL_PATH + ".zip"):
    model = PPO.load(OUT_MODEL_PATH, env=env, device="cpu")
    print(f"Resumed fine-tuning from {OUT_MODEL_PATH}.zip")
else:
    model = PPO.load(WARMED_MODEL_PATH, env=env, device="cpu")
    print(f"Starting fine-tuning from critic-warmed {WARMED_MODEL_PATH}.zip")

model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
model.save(OUT_MODEL_PATH)

with open(f"{ROOT}/run_output/finetune_bc_v2_done.txt", "w") as f:
    f.write(f"fine-tuned {TOTAL_TIMESTEPS} timesteps, saved to {OUT_MODEL_PATH}.zip\n")

env.close()
