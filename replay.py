import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from ppo import PPO

env = gym.make("HalfCheetah-v4", render_mode="human")
env.reset()

model = PPO(env)
model.load("PPOv1", "basicSeed522")

while True:
    done = truncated = False
    obs, info = env.reset()
    rew = 0
    t=0
    while not done and t < 1000:
        t+=1
        # with torch.no_grad():
        #     action = model.actor(obs).detach().numpy()
        # obs, reward, done, truncated, info = env.step(action)
        action, _ = model.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore
        done = terminated or truncated
        rew += reward
    print(f"epoch reward = {rew}") 
