"""Pretrains car2's PPO policy network via behavioral cloning on the
human_driving_data.npz recording, so RL fine-tuning starts from the
recorded driving style instead of a random initialization.

Pure supervised learning on already-recorded data - no Isaac Sim needed,
just a dummy env with matching observation/action spaces so PPO builds the
right-shaped network.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

DATA_PATH = f"{ROOT}/5_data/human_driving_data.npz"
OUT_MODEL_PATH = f"{ROOT}/3_trained_models/car2_bc_pretrained"
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3


class DummyCarEnv(gym.Env):
    """Matches CarTrackEnv's spaces exactly, without needing Isaac Sim."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(5, dtype=np.float32), 0.0, False, True, {}


data = np.load(DATA_PATH)
obs_data, act_data = data["obs"], data["act"]

model = PPO("MlpPolicy", DummyCarEnv(), device="cpu")
policy = model.policy

obs_tensor = torch.as_tensor(obs_data, dtype=torch.float32)
act_tensor = torch.as_tensor(act_data, dtype=torch.float32)

n = len(obs_tensor)
n_val = max(1, int(0.1 * n))
perm = torch.randperm(n)
val_idx, train_idx = perm[:n_val], perm[n_val:]

# Only the actor path (shared feature extractor + policy_net + action_net) -
# leave value_net untouched, RL fine-tuning learns its own value estimates.
params = list(policy.mlp_extractor.parameters()) + list(policy.action_net.parameters())
optimizer = torch.optim.Adam(params, lr=LR)


def actor_forward(obs_batch):
    features = policy.extract_features(obs_batch)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_pi)


with open(f"{ROOT}/run_output/pretrain_bc_log.txt", "w") as log_f:
    log_f.write(f"{n} samples ({len(train_idx)} train / {len(val_idx)} val)\n")
    for epoch in range(EPOCHS):
        policy.train()
        epoch_perm = train_idx[torch.randperm(len(train_idx))]
        total_loss = 0.0
        for start in range(0, len(epoch_perm), BATCH_SIZE):
            batch_idx = epoch_perm[start:start + BATCH_SIZE]
            pred = actor_forward(obs_tensor[batch_idx])
            loss = F.mse_loss(pred, act_tensor[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        train_loss = total_loss / len(train_idx)

        policy.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(actor_forward(obs_tensor[val_idx]), act_tensor[val_idx]).item()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            log_f.write(f"epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}\n")
            log_f.flush()

    # BC only fit the mean (action_net); log_std keeps SB3's default init
    # (0.0 -> std=1.0), which is huge next to a [-1, 1] action space. Left
    # alone, PPO samples rollout actions almost entirely from that noise,
    # burying the well-fit mean and starting fine-tuning from something
    # that behaves nothing like the recorded driving. Shrink it to a
    # confident-but-still-explorable std before saving.
    import math
    TARGET_STD = 0.15
    with torch.no_grad():
        policy.log_std.fill_(math.log(TARGET_STD))
    log_f.write(f"set log_std to {math.log(TARGET_STD):.5f} (std={TARGET_STD})\n")

model.save(OUT_MODEL_PATH)
with open(f"{ROOT}/run_output/pretrain_bc_done.txt", "w") as f:
    f.write(f"pretrained on {n} samples, saved to {OUT_MODEL_PATH}.zip\n")
