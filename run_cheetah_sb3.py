import gymnasium as gym
from stable_baselines3 import PPO
import torch
from torch import nn

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

save = "SB3/final"
print(torch.cuda.is_available())
print(torch.cuda.device(0))

env = gym.make("HalfCheetah-v4")

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
    clip_range=0.2,
    ent_coef=0.0004,
    gae_lambda=0.92,
    gamma=0.98,
    learning_rate=2.5e-5,
    max_grad_norm=0.8,
    n_steps=int(512*4),
    verbose=1,
    n_epochs=20,
    # batch_size=64
    # device="cuda",
    device="cpu",
    tensorboard_log=save,

    vf_coef=0.5,

    # seed
    seed=42
)

model.learn(int(1e6), log_interval=10, progress_bar=True)
model.save(save)

# Load and test saved model
env = gym.make("HalfCheetah-v4", render_mode="human")
env.reset()
# env = gym.make("racetrack-fast-v0", render_mode="rgb_array")
# model = PPO.load(save)
print("hello")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
  # env.render()