"""Trains a single shared policy for two-car self-play racing
(car_race_vec_env.py) - every car slot, in every racing pair, feeds
experience into the same network. Separate lineage from both the
single-env pure-RL policy (car2_ppo_model.zip) and the single-car vec+
curvature policy (car2_vec_ppo_model.zip): a fresh 10-dim-observation
network, since neither prior checkpoint's input layer matches.

Usage: python.bat train_race_vec.py [total_timesteps]
"""

import os
import shutil
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from car_race_vec_env import RaceVecEnv, NUM_PAIRS

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
RUN_LABEL = sys.argv[2] if len(sys.argv) > 2 else None
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/car_race_ppo_model"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/car_race_train_monitor.csv"
BACKUP_DIR = "C:/Users/sanja/Desktop/thesis/checkpoint_backups"

env = VecMonitor(RaceVecEnv(num_pairs=NUM_PAIRS, headless=True), filename=LOG_PATH)

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

# 2026-09-20: versioned backup so a bad round can be reverted without losing
# the previous checkpoint - car_race_ppo_model.zip is otherwise unconditionally
# overwritten every run with no history (this cost us the pre-run15 checkpoint
# permanently - see car_racing_status.md). Pass a run label as the 2nd CLI arg
# (e.g. "run29") to name the backup meaningfully; falls back to the cumulative
# step count if omitted, so backups still happen even if the label is forgotten.
# To revert: copy the desired checkpoint_backups/car_race_ppo_model_<label>.zip
# back over car_race_ppo_model.zip before launching the next training round.
os.makedirs(BACKUP_DIR, exist_ok=True)
label = RUN_LABEL if RUN_LABEL else f"step{model.num_timesteps}"
backup_path = os.path.join(BACKUP_DIR, f"car_race_ppo_model_{label}.zip")
shutil.copyfile(MODEL_PATH + ".zip", backup_path)
print(f"Backed up checkpoint to {backup_path}")

with open("C:/Users/sanja/Desktop/thesis/train_race_vec_done.txt", "w") as f:
    f.write(f"trained {TOTAL_TIMESTEPS} timesteps, saved to {MODEL_PATH}.zip, backup at {backup_path}\n")

env.close()
