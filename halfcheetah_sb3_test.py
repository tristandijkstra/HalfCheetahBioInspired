import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import torch
from torch import nn
from config import racewayConfig

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


print(torch.cuda.is_available())
print(torch.cuda.device(0))

env = gym.make("HalfCheetah-v4")
# env = gym.make("HalfCheetah-v2")

# config = env.configure(racewayConfig)

# model = PPO('MlpPolicy', env,
#               policy_kwargs=dict(net_arch=[256, 256]),
#               learning_rate=5e-4,
#             #   buffer_size=15000,
#               # learning_starts=200,
#               batch_size=32,
#               gamma=0.8,
#             #   train_freq=1,
#             #   gradient_steps=1,
#             #   target_update_interval=50,
#               verbose=1,
#               # device="cuda",
#               tensorboard_log="HalfCheetah/")

# model = PPO(
#     "MlpPolicy",
#     env,
#     #   policy_kwargs=dict(net_arch=[256, 256]),
#     policy_kwargs=dict(
#         log_std_init=-2,
#         ortho_init=False,
#         activation_fn=nn.ReLU,
#         net_arch=[dict(pi=[256, 256], vf=[256, 256])],
#     ),
#     clip_range=0.4,
#     ent_coef=0.0,
#     gae_lambda=0.9,
#     gamma=0.99,
#     learning_rate=3e-5,
#     max_grad_norm=0.8,
#     n_epochs=20,
#     batch_size=128,
#     n_steps=512,
#     vf_coef=0.5,
#     verbose=1,
#     # use_sde=True,
#     # sde_sample_freq=4,
#     tensorboard_log="HalfCheetah/",

#     seed=42
# )

model = PPO(
    "MlpPolicy",
    env,
    #   policy_kwargs=dict(net_arch=[256, 256]),
    policy_kwargs=dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    ),
    clip_range=0.1,
    # clip_range=0.2,
    # ent_coef=0.0004,
    ent_coef=0.001,
    # ent_coef=0.000,
    # ent_coef=0.000401762,
    gae_lambda=0.92,
    # gamma=0.99,
    gamma=0.98,
    # gamma=0.9,
    # learning_rate=5e-4,
    learning_rate=2.5e-5,
    # learning_rate=2.0e-5,
    # learning_rate=2.5e-5,
    # max_grad_norm=0.5,
    max_grad_norm=0.8,
    # n_envs=1,
    n_epochs=20,
    n_steps=512,
    # n_timesteps=1e6,
    # normalize=True,
    vf_coef=0.58096,
    # normalize_kwargs={"norm_obs": True, "norm_reward": False},
    #   buffer_size=15000,
    #   learning_starts=200,
    #   train_freq=1,
    #   gradient_steps=1,
    #   target_update_interval=50,
    verbose=1,
    # device="cuda",
    # use_sde=True,
    # sde_sample_freq=4,
    tensorboard_log="HalfCheetah/",

    seed=41
)
# model = PPO.load("HalfCheetah/model", env=env)

# model.use_sde = True
# model.set_parameters(dict(use_sde=True, sde_sample_freq=4))
# sde_sample_freq=4

model.learn(int(1e5), log_interval=10, progress_bar=True)
model.save("HalfCheetah/model")

# Load and test saved model
env = gym.make("HalfCheetah-v4", render_mode="human")
env.reset()
# env = gym.make("racetrack-fast-v0", render_mode="rgb_array")
model = PPO.load("HalfCheetah/model")
print("hello")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
  # env.render()