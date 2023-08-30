import matplotlib.pyplot as plt
import pandas as pd

file = "PPOv2/GaeEnt_2.csv"


fig, ax = plt.subplots(1,1)

# file1 = "PPOv2/improvedSeed520ortho.csv"
# file2 = "PPOv2/improvedSeed421.csv"
# file3 = "PPOv2/improvedSeed422.csv"
file1 = "PPOv1/basicSeed520.csv"
file2 = "PPOv1/basicSeed521.csv"
file3 = "PPOv1/basicSeed522.csv"

P1 = pd.read_csv(file1, index_col=0, header=0).iloc[::1]
P2 = pd.read_csv(file2, index_col=0, header=0).iloc[::1]
P3 = pd.read_csv(file3, index_col=0, header=0).iloc[::1]

ax.plot(P1.timestep, P1.reward)
ax.plot(P2.timestep, P2.reward)
ax.plot(P3.timestep, P3.reward)

ax.set_ylim(-100, 2000+100)
plt.show()