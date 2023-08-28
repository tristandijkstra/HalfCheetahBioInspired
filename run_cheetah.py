import gymnasium as gym
from ppo import PPO

env = gym.make("HalfCheetah-v4")
model = PPO(env)

model.learn(1e4)

