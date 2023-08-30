import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from ppoImproved import PPO
# from ppoo import PPO

env = gym.make("HalfCheetah-v4")
# from ppoImproved import PPO
model = PPO(env, seed=888)

model.learn(1e6)

model.save("PPOv2", "improvedSeed888")

env = gym.make("HalfCheetah-v4", render_mode="human")
env.reset()

print("hello")
while True:
    done = truncated = False
    obs, info = env.reset()
    rew = 0
    t=0
    while not done and t < 1000:
        t+=1
        action, _ = model.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore
        done = terminated or truncated
        rew += reward

    print(f"epoch reward = {rew}") 