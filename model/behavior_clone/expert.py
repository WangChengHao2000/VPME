import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Expert, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.encoder_layer = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.mu = nn.Sequential(
            nn.Linear(128, self.action_dim),
        )
        self.sigma = nn.Sequential(
            nn.Linear(128, self.action_dim),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.encoder_layer(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

    def save(self, filename: str):
        filename = "checkpoints/expert/" + filename
        torch.save(self.state_dict(), filename)
        print("successfully save " + filename)

    def load(self, filename: str):
        filename = "checkpoints/expert/" + filename
        self.load_state_dict(torch.load(filename))
        print("successfully load " + filename)


if __name__ == '__main__':
    import numpy as np

    model = Expert(100, 3)
    obs = np.zeros((1, 100))
    obs = torch.tensor(obs).to(torch.float32)
    action = model(obs)
    print(action)
