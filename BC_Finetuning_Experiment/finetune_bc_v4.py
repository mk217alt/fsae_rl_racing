"""Second attempt at the full BC+RL pipeline, this time with new driving
data recorded deliberately hugging the track centerline, and all three
fixes discovered during the first attempt (see rl_journal.html §3) applied
from the start instead of rediscovered one at a time:

  1. log_std tightened to std=0.15 during BC pretraining (pretrain_bc.py)
     instead of left at SB3's noisy default.
  2. Critic pre-regressed on real rollout returns (warmup_critic.py)
     before PPO ever touches the actor, instead of starting cold.
  3. A 6x-lower PPO learning rate (5e-5) so fine-tuning nudges the policy
     gently instead of overwriting it.

If this still plateaus the way v1-v3 did on the first recording, that's
much stronger evidence the driving-style/reward-function mismatch (not any
of the three training-mechanics issues) is the real limiting factor.
"""


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Single_Car_Baseline"))
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from car_track_env import CarTrackEnv

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
WARMED_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/BC_Finetuning_Experiment/car2_bc_warmed"
OUT_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/car2_bc_finetuned_v4"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/BC_Finetuning_Experiment/car2_bc_finetune_v4_monitor.csv"
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

with open("C:/Users/sanja/Desktop/thesis/finetune_bc_v4_done.txt", "w") as f:
    f.write(f"fine-tuned {TOTAL_TIMESTEPS} timesteps at lr={LEARNING_RATE}, saved to {OUT_MODEL_PATH}.zip\n")

env.close()
