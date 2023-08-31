import gymnasium as gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO


name = "final"
saveFile = "recordings/" + name

env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
env.reset()

model = PPO.load("SB3/final.zip")

data = []
done = truncated = False
obs, info = env.reset()
# env = gym.wrappers.RecordVideo(env=env, video_folder=saveFile)
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore
    done = terminated or truncated
    # data.append(action)
    data.append(obs[2:8])

names = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

P = pd.DataFrame(np.array(data), columns = names)

P.to_csv(saveFile + ".csv")

done = truncated = False
obs, info = env.reset()
env = gym.wrappers.RecordVideo(env=env, video_folder=saveFile)
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore
    done = terminated or truncated
