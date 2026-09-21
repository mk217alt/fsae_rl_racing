"""Third attempt at RL fine-tuning from a behavioral-cloning start point.

v1 (no critic warmup): degraded badly - cold critic gave garbage early
advantage estimates that dragged the good BC actor down over the whole run
(episode length 137 -> 81, reward never above -37).

v2 (critic pre-warmed on 40 real rollout episodes' Monte-Carlo returns,
default PPO learning_rate=3e-4): fixed the catastrophic noise, but the run
plateaued rather than recovering - reward oscillated between -83 and -117
across all four quartiles of the run with no clear upward trend, and the
fraction of episodes reaching the full 500-step length actually fell
(13 -> 7 comparing first/last quartile). Likely cause: 3e-4 is a fairly
large step size for nudging an already-good policy - each PPO update was
probably knocking the actor back out of the good BC basin about as fast as
it could recover.

v3 starts from the SAME critic-warmed checkpoint (not the already-degraded
v2 one) with a 6x lower learning rate, so updates should adapt the policy
gently instead of overwriting it, while still building on a calibrated
critic. Confirmed via a standalone check that PPO.load(..., learning_rate=X)
actually overrides the loaded schedule before committing to a full run.
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
OUT_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/car2_bc_finetuned_v3"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/car2_bc_finetune_v3_monitor.csv"
LEARNING_RATE = 5e-5

env = Monitor(CarTrackEnv(headless=True), filename=LOG_PATH)

if os.path.exists(OUT_MODEL_PATH + ".zip"):
    model = PPO.load(OUT_MODEL_PATH, env=env, device="cpu", learning_rate=LEARNING_RATE)
    print(f"Resumed fine-tuning from {OUT_MODEL_PATH}.zip at lr={LEARNING_RATE}")
else:
    model = PPO.load(WARMED_MODEL_PATH, env=env, device="cpu", learning_rate=LEARNING_RATE)
    print(f"Starting fine-tuning from critic-warmed {WARMED_MODEL_PATH}.zip at lr={LEARNING_RATE}")

model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
model.save(OUT_MODEL_PATH)

with open("C:/Users/sanja/Desktop/thesis/finetune_bc_v3_done.txt", "w") as f:
    f.write(f"fine-tuned {TOTAL_TIMESTEPS} timesteps at lr={LEARNING_RATE}, saved to {OUT_MODEL_PATH}.zip\n")

env.close()
