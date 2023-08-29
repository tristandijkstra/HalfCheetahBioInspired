import gymnasium as gym
import torch
import matplotlib.pyplot as plt
# from ppoo import PPO

env = gym.make("HalfCheetah-v4")
# env = gym.make("HalfCheetah-v4", render_mode="human")
# env = gym.make("Pendulum-v1")
# from ppo import PPO
from ppoImproved import PPO
model = PPO(env)
# from ppo4 import PPO
# from network import FeedForwardNN
# model = PPO(FeedForwardNN, env)
# # 
model.learn(1e6)
# model.learn(5e4)
# model.learn(1e4)
# model.learn(1e4)
# model.learn(5e5)
model.save("PPOv2", "GaeEnt_lowlr")

# fig, ax = plt.subplots(1, 1)

# ax.plot(model.plotTimestep, model.plotRewards)
# plt.show()
# Load and test saved model
# env = gym.make("HalfCheetah-v4", render_mode="human")
env = gym.make("HalfCheetah-v4", render_mode="human")
env.reset()
# env = gym.make("racetrack-fast-v0", render_mode="rgb_array")
# model = PPO.load("HalfCheetah/model")
# model = PPO(env)
# model.load("PPOv2")
# model.env = env
# model.actor.eval()
print("hello")
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

  # env.render()