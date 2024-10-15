import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class CNNVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims=100):
        super(CNNVariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims).to(device)
        self.decoder = Decoder(latent_dims).to(device)

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        return self.decoder(z), mu, sigma

    def save(self, filename: str):
        filename = "checkpoints/vae/" + filename
        torch.save(self.state_dict(), filename)
        print("successfully save " + filename)

    def load(self, filename: str):
        filename = "checkpoints/vae/" + filename
        self.load_state_dict(torch.load(filename))
        print("successfully load " + filename)


class Encoder(nn.Module):
    def __init__(self, latent_dims=100):
        super(Encoder, self).__init__()
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.LeakyReLU())

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2),
            nn.LeakyReLU())

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.encoder_layer5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.linear = nn.Sequential(
            nn.Linear(128 * 6 * 6, 768),
            nn.LeakyReLU())

        self.mu = nn.Linear(768, latent_dims)
        self.sigma = nn.Linear(768, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

    def forward(self, x):
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.encoder_layer5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        return z, mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 128 * 6 * 6),
            nn.LeakyReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 6, 6))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Sigmoid())

    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    vae = CNNVariationalAutoencoder()
    image = np.zeros((3, 100, 100))
    result = vae(image)
