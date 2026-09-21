
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Single_Car_Baseline"))
import numpy as np

from car_track_env import CarTrackEnv

env = CarTrackEnv(headless=True)

with open("C:/Users/sanja/Desktop/thesis/test_car_track_env_result.txt", "w") as f:
    obs, info = env.reset()
    f.write(f"reset obs: {obs}\n")
    f.write(f"observation_space: {env.observation_space}\n")
    f.write(f"action_space: {env.action_space}\n")

    total_reward = 0.0
    for i in range(30):
        action = np.array([0.5, 0.0], dtype=np.float32)  # constant mild throttle, straight
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        f.write(f"step {i}: obs={obs}, reward={reward:.4f}, terminated={terminated}, truncated={truncated}\n")
        if terminated or truncated:
            f.write("episode ended early\n")
            break

    f.write(f"total_reward over run: {total_reward:.4f}\n")
    f.flush()

env.close()
