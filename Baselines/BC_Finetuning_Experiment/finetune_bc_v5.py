"""Third round of the BC+RL pipeline, on a third independent driving
recording (1422 samples, deliberately varied speed/line this time - see
rl_journal.html §3 for rounds 1-2, both of which plateaued at the same
ceiling regardless of driving style). Same three fixes as before, applied
from the start: log_std=0.15 (pretrain_bc.py), critic warmup
(warmup_critic.py), and a 6x-lower PPO learning rate here.
"""


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Baselines/Single_Car_Baseline"))
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from car_track_env import CarTrackEnv

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
WARMED_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/car2_bc_warmed"
OUT_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/car2_bc_finetuned_v5"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/car2_bc_finetune_v5_monitor.csv"
LEARNING_RATE = 5e-5

env = Monitor(CarTrackEnv(headless=True), filename=LOG_PATH)

if os.path.exists(OUT_MODEL_PATH + ".zip"):
    model = PPO.load(OUT_MODEL_PATH, env=env, device="cpu", learning_rate=LEARNING_RATE)
    print(f"Resumed fine-tuning from {OUT_MODEL_PATH}.zip")
else:
    model = PPO.load(WARMED_MODEL_PATH, env=env, device="cpu", learning_rate=LEARNING_RATE)
    print(f"Starting fine-tuning from critic-warmed {WARMED_MODEL_PATH}.zip at lr={LEARNING_RATE}")

model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
model.save(OUT_MODEL_PATH)

with open("C:/Users/sanja/Desktop/thesis/finetune_bc_v5_done.txt", "w") as f:
    f.write(f"fine-tuned {TOTAL_TIMESTEPS} timesteps at lr={LEARNING_RATE}, saved to {OUT_MODEL_PATH}.zip\n")

env.close()
