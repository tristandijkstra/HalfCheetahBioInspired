import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import gymnasium
from tqdm import tqdm
from torch.distributions import MultivariateNormal
import os
import pandas as pd
# torch.set_default_device("cuda")

# TODO gsde

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """Basic FeedForward Neural Network that serves as our Actor/Critic Models

        Args:
            input_dim (int): Dimension of out observation space
            output_dim (int): Dimension of our action space
            hidden_dim (int, optional): _description_. Defaults to 64.
        """
        super(FeedForwardNN, self).__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        # self.l2a = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

        # torch.nn.init.orthogonal_(self.l1.weight, gain=1.0)
        # torch.nn.init.orthogonal_(self.l2.weight, gain=1.0)
        # torch.nn.init.orthogonal_(self.l3.weight, gain=1.0)
        # torch.nn.init.normal_(self.l1.bias, mean=0, std= 1)
        # torch.nn.init.normal_(self.l2.bias, mean=0, std= 1)
        # torch.nn.init.normal_(self.l3.bias, mean=0, std= 1)

    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float)
        # obs -> l1 -> relu1 -> l2 -> relu2 -> l3 -> out
        activation1 = F.relu(self.l1(observation))
        # activation2 = F.relu(self.l2a(activation1))
        activation3 = F.relu(self.l2(activation1))
        out = self.l3(activation3)
        return out


class PPO:
    def __init__(self, env: gymnasium.Env, hidden_dim=256, seed=420) -> None:
        torch.manual_seed(seed)
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # pseudocode step 1: input
        self.actor = FeedForwardNN(self.observation_dim, self.action_dim, hidden_dim)
        self.critic = FeedForwardNN(self.observation_dim, 1, hidden_dim)

        self._init_hyperparams()

        # cov matrix
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # optimiser (gradient descent)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # saving
        self.trained = False
        self.timestepGlobal = 0
        self.plotRewards = []
        self.plotTimestep = []

        self.record_every = int(512*8)
        self.timestep_of_last_record = -(self.record_every+1)


    def _init_hyperparams(self):
        self.timesteps_per_batch = int(8*1024)
        self.max_timesteps_per_episode = 512
        self.gamma = 0.99
        self.n_updates_per_iteration = 20
        self.clip = 0.1
        # self.lr = 2e-3
        # self.lr = 9e-4
        self.lr = 2.5e-4
        # self.lr = 2.5e-5
        # self.lr = 5e-5

        self.max_grad_norm = 0.8
        # self.timesteps_per_batch = 4800
        # self.max_timesteps_per_episode = 1600
        # self.gamma = 0.98
        # self.n_updates_per_iteration = 5
        # self.clip = 0.2
        # self.lr = 0.005

    def learn(self, total_timesteps):
        timestep = 0
        self.trained = True

        # step 2
        # progress_bar = tqdm(total=total_timesteps)
        with tqdm(total=total_timesteps) as progress_bar:
            while timestep < total_timesteps:
                (
                    batch_obs,
                    batch_acts,
                    batch_log_probs,
                    batch_rtgs,
                    batch_lens,
                    batch_rewards
                ) = self.rollout()


                V, _ = self.evaluate(batch_obs, batch_acts)

                # STEP 5
                A_k = batch_rtgs - V.detach()
                # Advantage normalisation
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                for _ in range(self.n_updates_per_iteration):
                    # Calculate pi_theta(a_t | s_t) (upper part of quotient)
                    # lower part is batch_log_probs
                    _, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # since they are both log we can take diff to get the ratio
                ratios = torch.exp(curr_log_probs-batch_log_probs) #type: ignore

                # surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1- self.clip, 1 + self.clip) * A_k

                # mean loss of the actor
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Backpropagation
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()


                # STEP 7: critic
                # Calculate V_phi and pi_theta(a_t | s_t)    
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                timesteps_this_batch:int = sum(batch_lens)
                timestep += timesteps_this_batch
                self.timestepGlobal += timesteps_this_batch

                progress_bar.update(timesteps_this_batch)

                # mean_rew = round(np.array(batch_rewards).sum(axis=1).mean(), 2)
                # progress_bar.set_description(f"batch rew: {mean_rew}")
                progress_bar.set_description(f"rew: {round(self.plotRewards[-1], 3)} | act loss: {actor_loss}")



    def evaluate(self, batch_obs, batch_acts):
        # V and log prob of batch actions
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs
    

    def get_action(self, obs:torch.Tensor, deterministic=False):
        # get a mean action
        # obs = torch.tensor(obs,dtype=torch.float)
        mean = self.actor(obs)

        distribution = MultivariateNormal(mean, self.cov_mat)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        if deterministic:
            with torch.no_grad():
                action = mean.detach().numpy()
            return action, 1

        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rewards):
        batch_rtgs = []

        # nasty double for loop here
        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0

            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def rollout(self):
        # comments indicate dimensions
        batch_observations = []  # [n_timesteps_p_batch | dim_observation]
        # batch_observations = np.zeros((self.timesteps_per_batch, self.observation_dim))
        batch_actions = []  # [n_timesteps_p_batch | dim_actions]
        batch_log_probabilities = []  # [n_timesteps_p_batch | dim_observation]
        batch_rewards = []  # [n_episodes | n_timesteps_p_episode]
        # batch_rewards_to_go = [] # [n_timesteps_p_batch]
        batch_lengths = []  # [n_episodes]

        t = 0

        while t < self.timesteps_per_batch:
            episode_rewards = []
            observations, _ = self.env.reset()
            done = False

            for episode in range(self.max_timesteps_per_episode):
                t += 1

                # collect obs and step
                batch_observations.append(observations)
                action, log_prob = self.get_action(observations)  # type: ignore
                observations, reward, terminated, truncated, _ = self.env.step(action)  # type: ignore
                done = terminated or truncated

                # save rewards action and log_prob
                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probabilities.append(log_prob)

                if done:
                    break

            batch_lengths.append(episode + 1)  # type: ignore
            # print(episode_rewards)
            batch_rewards.append(episode_rewards)

        
        if (self.timestepGlobal + t) - self.timestep_of_last_record > self.record_every:
            # log for plot
            obs_, _ = self.env.reset()
            rew_ = 0
            t_=0
            done_ = False
            while not done_ and t_ < 1000:
                t+=1
                action_, _ = self.get_action(obs_, deterministic=True)
                obs_, reward_, terminated_, truncated_, _ = self.env.step(action_)  # type: ignore
                done_ = terminated_ or truncated_
                rew_ += reward_

            self.plotRewards.append(rew_)
            self.plotTimestep.append(self.timestepGlobal + t)

            self.timestep_of_last_record = self.timestepGlobal + t

        # reshape to tensors
        batch_observations = torch.tensor(np.array(batch_observations), dtype=torch.float)
        # batch_observations = torch.tensor(batch_observations, dtype=torch.float)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float)
        # batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probabilities = torch.tensor(
            batch_log_probabilities, dtype=torch.float
        )

        batch_rewards_to_go = self.compute_rtgs(batch_rewards)

        return (
            batch_observations,
            batch_actions,
            batch_log_probabilities,
            batch_rewards_to_go,
            batch_lengths,
            batch_rewards
        )


    def save(self, path:str, name=None):
        print("Saving")

        if name is None:
            full = path
            if not os.path.exists(path):
                os.mkdir(path)
        else:
            full = path + "/" + name
            if not os.path.exists(full):
                os.mkdir(full)

        torch.save(self.actor.state_dict(), f'{full}/ppo_actor.pth')
        torch.save(self.critic.state_dict(), f'{full}/ppo_critic.pth')

        csv = pd.DataFrame(np.array([self.plotTimestep, self.plotRewards]).T, columns=["timestep", "reward"])
        csv.to_csv(f"{path}/{name}.csv")

    def load(self, path:str, name=None):
        if name is None:
            full = path
            if not os.path.exists(path):
                os.mkdir(path)
        else:
            full = path + "/" + name
            if not os.path.exists(full):
                os.mkdir(full)
        self.actor.load_state_dict(torch.load(f'{full}/ppo_actor.pth'))
        # self.critic.load_state_dict(f'.{path}/ppo_critic.pth')

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)