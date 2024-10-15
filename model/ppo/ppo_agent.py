import numpy as np

import torch
import torch.nn as nn

from model.ppo.actor_critic import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent(object):
    def __init__(self, obs_dim=100, ppo_train_config=None):

        self.obs_dim = obs_dim
        self.action_dim = 2
        self.clip = ppo_train_config.POLICY_CLIP
        self.gamma = ppo_train_config.GAMMA
        self.n_updates_per_iteration = ppo_train_config.N_UPDATES_PER_ITERATION
        self.lr = ppo_train_config.PPO_LEARNING_RATE
        self.memory = Buffer()
        self.checkpoint_file_no = 0

        self.policy = ActorCritic(self.obs_dim, self.action_dim).to(device)
        self.optimizer1 = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr)
        self.optimizer_td = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr)

        self.old_policy = ActorCritic(self.obs_dim, self.action_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def get_action(self, obs):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)
            action_mean, action_sigma, action, logprob = self.old_policy.get_action_and_log_prob(obs.to(device))
        if is_train:
            self.memory.observation.append(obs.to(device))
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)

        return action_mean.cpu().numpy().flatten(), action_sigma.cpu().numpy().flatten(), action.detach().cpu().numpy().flatten()

    def learn(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # shape [batchsize, ]
        # print("rewards: ", rewards)
        # print(rewards.shape)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(
            device)  # shape [batchsize, latent_dim]
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)  # shape [batchsize, 2]
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(
            device)  # shape [batchsize, ]

        total_epoch_loss = 0
        # Optimize policy for K epochs
        for _ in range(self.n_updates_per_iteration):
            # Evaluating old actions and values
            # print(old_states.device)
            # print(old_actions.device)
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)  # [b, ], [b, 1], [b, ]

            # print("logprobs:", logprobs.shape)  # shape [batchsize, ]
            # print("values:", values.shape)  # shape [batchsize, 1]
            # print("dist_entropy:", dist_entropy.shape)  # shape [batchsize, ]
            # exit()

            # match values tensor dimensions with rewards tensor
            values = torch.squeeze(values)  # shape [batchsize, 1] ->  [batchsize, ]

            # Finding the ratio (pi_theta / pi_theta__old)
            # Calculate ratio:
            # r_t(θ) = exp( logs   π(a_t | s_t; θ) - logs π(a_t | s_t; θ_old)   )
            # r_t(θ) = exp( logs ( π(a_t | s_t; θ) /     π(a_t | s_t; θ_old) ) )
            # r_t(θ) = π(a_t | s_t; θ) / π(a_t | s_t; θ_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - values.detach()  # 减掉baseline
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            # final loss of clipped objective PPO
            # # 源代码
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * dist_entropy
            #
            # total_epoch_loss += loss.mean().item()
            # # take gradient step
            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimizer.step()
            loss1 = -torch.min(surr1, surr2)
            loss1 = loss1.mean()
            loss_td = self.MseLoss(values, rewards)

            total_epoch_loss += loss1.item()
            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()

            self.optimizer_td.zero_grad()
            loss_td.backward()
            self.optimizer_td.step()
        self.old_policy.load_state_dict(self.policy.state_dict())

        average_train_loss = total_epoch_loss / self.n_updates_per_iteration
        print('Train: ',
              'Experience Number: {}'.format(len(self.memory.rewards)),
              ', Learn times: {}'.format(self.n_updates_per_iteration),
              ', Average Train Loss:  {:.2f}'.format(average_train_loss))

        self.memory.clear()

    def save(self, filename):
        torch.save(self.old_policy.state_dict(), filename)
        print("successfully save " + filename)

    def load(self, checkpoint_file):
        self.old_policy.load_state_dict(torch.load(checkpoint_file))
        self.policy.load_state_dict(torch.load(checkpoint_file))
        print("load " + checkpoint_file + " successfully")


class Buffer:
    def __init__(self):
        # Batch data
        self.observation = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.observation[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
