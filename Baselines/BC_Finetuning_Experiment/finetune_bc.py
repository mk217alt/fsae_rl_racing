"""Continues RL training from the behavioral-cloning-pretrained policy
(car2_bc_pretrained.zip), so it keeps self-correcting via trial and error but
starts from the recorded driving style instead of scratch. Saves to a new
lineage (car2_bc_finetuned.zip) - the pure-RL car2_ppo_model.zip is untouched.
"""


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Baselines/Single_Car_Baseline"))
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from car_track_env import CarTrackEnv

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
BC_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/car2_bc_pretrained"
OUT_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/car2_bc_finetuned"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/car2_bc_finetune_monitor.csv"

env = Monitor(CarTrackEnv(headless=True), filename=LOG_PATH)

if os.path.exists(OUT_MODEL_PATH + ".zip"):
    model = PPO.load(OUT_MODEL_PATH, env=env, device="cpu")
    print(f"Resumed fine-tuning from {OUT_MODEL_PATH}.zip")
else:
    model = PPO.load(BC_MODEL_PATH, env=env, device="cpu")
    print(f"Starting fine-tuning from BC-pretrained {BC_MODEL_PATH}.zip")

model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
model.save(OUT_MODEL_PATH)

with open("C:/Users/sanja/Desktop/thesis/finetune_bc_done.txt", "w") as f:
    f.write(f"fine-tuned {TOTAL_TIMESTEPS} timesteps, saved to {OUT_MODEL_PATH}.zip\n")

env.close()
