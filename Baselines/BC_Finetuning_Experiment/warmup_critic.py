"""Warms up the value network (critic) on real rollout returns from the
BC-pretrained actor, BEFORE handing the model to PPO fine-tuning.

Root cause of the first fine-tune failing to improve: BC pretraining only
fit the actor (mean output + tightened log_std). The critic was never
trained on anything, so at the start of PPO fine-tuning its advantage
estimates were essentially noise. PPO's first policy updates trusted those
bad advantages and visibly degraded the actor over the whole 50k-step run
(episode length fell from ~137 to ~81 steps average, reward never rose
above -37) instead of building on the good BC starting point.

This script collects real rollouts under the (frozen) BC actor, computes
Monte-Carlo discounted returns for each transition, and regresses ONLY the
critic pathway (mlp_extractor.value_net + the outer value_net) toward
those returns via MSE - mirroring how pretrain_bc.py trained only the actor
pathway and left the critic alone. The actor (mlp_extractor.policy_net,
action_net, log_std) is never touched here. Output is a new checkpoint
with a good actor AND a roughly-calibrated critic, ready for real PPO
fine-tuning to pick up from.
"""


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Baselines/Single_Car_Baseline"))
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from car_track_env import CarTrackEnv

BC_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/car2_bc_pretrained"
OUT_MODEL_PATH = "C:/Users/sanja/Desktop/thesis/Baselines/BC_Finetuning_Experiment/car2_bc_warmed"
LOG_PATH = "C:/Users/sanja/Desktop/thesis/warmup_critic_log.txt"

WARMUP_EPISODES = 40
CRITIC_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3

env = CarTrackEnv(headless=True)
model = PPO.load(BC_MODEL_PATH, device="cpu")
policy = model.policy
gamma = model.gamma

log_f = open(LOG_PATH, "w")
log_f.write(f"gamma={gamma}\n")

# --- 1. collect real rollouts under the (unmodified) BC actor ---------------
all_obs, all_returns = [], []
episode_rewards, episode_lengths = [], []

for ep in range(WARMUP_EPISODES):
    obs, _ = env.reset()
    ep_obs, ep_rewards = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        ep_obs.append(obs)
        ep_rewards.append(reward)
        obs = next_obs
        done = terminated or truncated

    # Monte-Carlo discounted return-to-go within this (complete) episode
    returns = [0.0] * len(ep_rewards)
    running = 0.0
    for t in reversed(range(len(ep_rewards))):
        running = ep_rewards[t] + gamma * running
        returns[t] = running

    all_obs.extend(ep_obs)
    all_returns.extend(returns)
    episode_rewards.append(sum(ep_rewards))
    episode_lengths.append(len(ep_rewards))

    log_f.write(
        f"rollout ep {ep}: reward={episode_rewards[-1]:.2f} length={episode_lengths[-1]}\n"
    )
    log_f.flush()

log_f.write(
    f"collected {len(all_obs)} transitions from {WARMUP_EPISODES} episodes, "
    f"avg reward={np.mean(episode_rewards):.2f} avg length={np.mean(episode_lengths):.2f}\n"
)
log_f.flush()

# NOTE: do NOT close the Isaac Sim env here - env.close() tears down the
# whole SimulationApp/process. Data collection is done, but the critic
# regression below is pure PyTorch and doesn't need the sim; closing early
# killed the process before it ever reached this code the first time.

# --- 2. regress the critic pathway only, actor stays frozen -----------------
obs_tensor = torch.as_tensor(np.array(all_obs), dtype=torch.float32)
returns_tensor = torch.as_tensor(np.array(all_returns), dtype=torch.float32).unsqueeze(1)

critic_params = list(policy.mlp_extractor.value_net.parameters()) + list(policy.value_net.parameters())
optimizer = torch.optim.Adam(critic_params, lr=LR)


def critic_forward(obs_batch):
    features = policy.extract_features(obs_batch)
    latent_vf = policy.mlp_extractor.forward_critic(features)
    return policy.value_net(latent_vf)


n = len(obs_tensor)
n_val = max(1, int(0.1 * n))
perm = torch.randperm(n)
val_idx, train_idx = perm[:n_val], perm[n_val:]

for epoch in range(CRITIC_EPOCHS):
    policy.train()
    epoch_perm = train_idx[torch.randperm(len(train_idx))]
    total_loss = 0.0
    for start in range(0, len(epoch_perm), BATCH_SIZE):
        batch_idx = epoch_perm[start:start + BATCH_SIZE]
        pred = critic_forward(obs_tensor[batch_idx])
        loss = F.mse_loss(pred, returns_tensor[batch_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_idx)
    train_loss = total_loss / len(train_idx)

    policy.eval()
    with torch.no_grad():
        val_loss = F.mse_loss(critic_forward(obs_tensor[val_idx]), returns_tensor[val_idx]).item()

    if epoch % 10 == 0 or epoch == CRITIC_EPOCHS - 1:
        log_f.write(f"critic epoch {epoch}: train_loss={train_loss:.3f} val_loss={val_loss:.3f}\n")
        log_f.flush()

model.save(OUT_MODEL_PATH)
log_f.write(f"saved warmed-up model to {OUT_MODEL_PATH}.zip\n")
log_f.close()

with open("C:/Users/sanja/Desktop/thesis/warmup_critic_done.txt", "w") as f:
    f.write(f"warmed critic on {len(all_obs)} transitions, saved to {OUT_MODEL_PATH}.zip\n")

env.close()
