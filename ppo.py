import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import gymnasium
from tqdm import tqdm
from torch.distributions import MultivariateNormal

torch.set_default_device("cuda")

# TODO gsde

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        """Basic FeedForward Neural Network that serves as our Actor/Critic Models

        Args:
            input_dim (int): Dimension of out observation space
            output_dim (int): Dimension of our action space
            hidden_dim (int, optional): _description_. Defaults to 64.
        """
        super(FeedForwardNN, self).__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(input_dim, output_dim)

    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float)
        activation1 = F.relu(self.l1(observation))
        activation2 = F.relu(self.l2(activation1))
        out = self.l3(activation2)
        return out


class PPO:
    def __init__(self, env: gymnasium.Env) -> None:
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # pseudocode step 1: input
        self.actor = FeedForwardNN(self.observation_dim, self.action_dim)
        self.critic = FeedForwardNN(self.observation_dim, 1)

        self._init_hyperparams()

        # cov matrix
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # optimiser (gradient descent)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparams(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.98
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def learn(self, total_timesteps):
        timestep = 0

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
                self.actor_optim.step()


                # STEP 7: critic
                # Calculate V_phi and pi_theta(a_t | s_t)    
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()

                timesteps_this_batch:int = np.sum(batch_lens, dtype=np.int)
                timestep += timesteps_this_batch

                progress_bar.update(timesteps_this_batch)



    def evaluate(self, batch_obs, batch_acts):
        # V and log prob of batch actions
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs
    

    def get_action(self, obs):
        # get a mean action
        mean = self.actor(obs)

        distribution = MultivariateNormal(mean, self.cov_mat)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

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
        batch_actions = []  # [n_timesteps_p_batch | dim_actions]
        batch_log_probabilities = []  # [n_timesteps_p_batch | dim_observation]
        batch_rewards = []  # [n_episodes | n_timesteps_p_episode]
        # batch_rewards_to_go = [] # [n_timesteps_p_batch]
        batch_lengths = []  # [n_episodes]

        t = 0

        while t < self.timesteps_per_batch:
            episode_rewards = []
            observations = self.env.reset()
            done = False

            for episode in range(self.max_timesteps_per_episode):
                t += 1

                # collect obs and step
                batch_observations.append(observations)
                action, log_prob = self.get_action(observations)  # type: ignore
                observations, reward, done, _ = self.env.step(action)  # type: ignore

                # save rewards action and log_prob
                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probabilities.append(log_prob)

                if done:
                    break

            batch_lengths.append(episode + 1)  # type: ignore
            batch_rewards.append(episode_rewards)

        # reshape to tensors

        batch_observations = torch.tensor(batch_observations, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
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
        )
