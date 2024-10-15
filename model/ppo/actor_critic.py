import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # actor
        self.actor_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(128, self.action_dim),
        )
        self.actor_sigma = nn.Sequential(
            nn.Linear(128, self.action_dim),
            nn.Softplus(),
        )

    def forward(self, obs):
        x = self.actor_encoder(obs)
        mean = self.actor_mean(x)
        sigma = self.actor_sigma(x)
        return mean, sigma


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # actor
        self.actor = Actor(self.obs_dim, self.action_dim)

        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError

    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        return self.critic(obs)

    def get_action_and_log_prob(self, obs):
        mean, sigma = self.actor(obs)

        cov_mat = torch.diag(sigma ** 2).to(device)
        dist = MultivariateNormal(mean, cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return mean.detach(), sigma.detach(), action.detach(), log_prob.detach()

    def evaluate(self, obs, action):
        mean, sigma = self.actor(obs)

        cov_mat = torch.diag_embed(sigma ** 2).to(device)
        dist = MultivariateNormal(mean, cov_mat)

        log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        values = self.critic(obs)

        return log_probs, values, dist_entropy
