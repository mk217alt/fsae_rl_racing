"""Continues training the two-car self-play policy (car_race_ppo_model.zip,
currently run49's weights) on the UNKNOWN track (unknown_track_train.usd,
built by build_unknown_track_train.py: 40m long straight, 8m short
straight, 6m corner radius - vs the original lineage's trained-on 16m/16m/
10m symmetric square).

Started after the zero-shot generalization test showed the policy's
survival behavior transfers almost perfectly to this track (full-length
rate unchanged, 93.3% both) but its reward/driving-line quality collapses
(mean reward +141.7 -> -155.5) - the corner curvature and straight length
are far enough outside the training distribution that the learned racing
line doesn't transfer. This script lets the SAME shared checkpoint adapt to
the new geometry directly, rather than starting a separate lineage from
scratch - run49's weights (and the original track's whole 40M-step lineage)
are preserved in checkpoint_backups/, so this is fully reversible.

Usage: python.bat train_unknown_track.py [total_timesteps] [run_label] [stagger_prob]
       stagger_prob (default 0 = always side by side): fraction of resets where the
       trailing car starts 1.5-8 m behind the leader (see car_unknown_track_env.py).
"""

import os
import shutil
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from car_unknown_track_env import UnknownTrackVecEnv, NUM_PAIRS

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
RUN_LABEL = sys.argv[2] if len(sys.argv) > 2 else None
STAGGER_PROB = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0  # fraction of resets with a staggered start
MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/car_race_ppo_model"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Two_Car_Self_Play_Racing/car_unknown_track_train_monitor.csv"
BACKUP_DIR = "C:/Users/sanja/Desktop/thesis/checkpoint_backups"

env = VecMonitor(UnknownTrackVecEnv(num_pairs=NUM_PAIRS, headless=True, stagger_prob=STAGGER_PROB), filename=LOG_PATH)

if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, env=env, device="cpu")
    print(f"Resumed training from existing checkpoint at {MODEL_PATH}.zip (now training on the unknown track)")
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

os.makedirs(BACKUP_DIR, exist_ok=True)
label = RUN_LABEL if RUN_LABEL else f"newtrack_step{model.num_timesteps}"
backup_path = os.path.join(BACKUP_DIR, f"car_race_ppo_model_{label}.zip")
shutil.copyfile(MODEL_PATH + ".zip", backup_path)
print(f"Backed up checkpoint to {backup_path}")

with open("C:/Users/sanja/Desktop/thesis/train_unknown_track_done.txt", "w") as f:
    f.write(f"trained {TOTAL_TIMESTEPS} timesteps on the unknown track, saved to {MODEL_PATH}.zip, backup at {backup_path}\n")

env.close()
