# Half Cheetah - RL for legged robots
In this repository, I control the Half Cheetah Reinforcement Learning benchmark with various PPO models.

This project was done as an assignment for the AE4350 Bio-Inspired Learning course at TU Delft.

The final model has a reward score of 3387.

#### Installing dependencies
The dependencies can be installed using conda /mamba using the following command.
```py
conda env create -f environment.yml
```
## Important files

#### Models from Scratch
- **Basic model**: ppo.py
- **Improved model**: ppoImproved.py
#### Run files
- **Train basic model**: run_cheetah.py
- **Train improved model**: run_cheetah_improved.py
- **Train stable baselines3 model**: run_cheetah_sb3.py

#### Final Model
The final Stable Baselines model can be loading using: SB3/final.zip

#### Other files
Plotting files are prefaced with "plot". Videos are made using record files. 
## Videos
#### Doing Nothing


https://github.com/tristandijkstra/HalfCheetahBioInspired/assets/13809731/c4bfe82d-4fc8-4800-946f-d1af6763e5c3



#### Intentional flip and moving on back

https://github.com/tristandijkstra/HalfCheetahBioInspired/assets/13809731/ae8d1b87-3119-423b-b22f-3eea3a28e270

#### Walking on face

https://github.com/tristandijkstra/HalfCheetahBioInspired/assets/13809731/55553fa9-1c74-4ce2-8354-7947d00c956a

#### Okay Gait

https://github.com/tristandijkstra/HalfCheetahBioInspired/assets/13809731/1c66a19c-2500-4457-b296-c5063c98aa83

#### Better Gait --- Final model

https://github.com/tristandijkstra/HalfCheetahBioInspired/assets/13809731/db3226f0-7794-441a-839b-75eb1d3911e1
