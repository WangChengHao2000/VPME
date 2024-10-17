import torch
from torch import nn

from model.behavior_clone.expert import Expert
from model.vae.vae_cnn import CNNVariationalAutoencoder


class VPMEPolicyNet(nn.Module):
    def __init__(self):
        super(VPMEPolicyNet, self).__init__()


class VPMEValueNet(nn.Module):
    def __init__(self):
        super(VPMEValueNet, self).__init__()


class VPMEAgent(object):
    def __init__(self, actor_learning_rate, critic_learning_rate, gamma, lmbda):
        self.vae = CNNVariationalAutoencoder()
        self.expert_list = list()
        for i in range(8):
            expert = Expert(obs_dim=100, action_dim=2)
            expert.load("")
            self.expert_list.append(expert)

        self.actor = VPMEPolicyNet()
        self.critic = VPMEValueNet()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.gamma = gamma
        self.lmbda = lmbda

    def take_action(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state)
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action
