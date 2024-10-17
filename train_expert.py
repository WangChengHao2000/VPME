import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
from tqdm import tqdm

from model.behavior_clone.expert import Expert
from model.behavior_clone.expert_folder import ExpertImageFolder
from model.vae.vae_cnn import CNNVariationalAutoencoder

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TARIN_EPOCHS = 5000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LATENT_SPACE = 128
ACTION_SPACE = 2
EPS = 1e-6


def main():
    vae = CNNVariationalAutoencoder(latent_dims=LATENT_SPACE)
    vae.load("best.pth")
    vae.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = ExpertImageFolder("data/expert/ParkingExit", transform=transform)

    m = len(train_data)
    train_data, valid_data = random_split(train_data, [int(m - m * 0.2), int(m * 0.2)])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    model = Expert(obs_dim=LATENT_SPACE, action_dim=ACTION_SPACE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.L1Loss()

    min_valid_total_loss = 10000

    for epoch in tqdm(range(TARIN_EPOCHS)):
        model.train()
        train_total_loss = 0
        for x, _, y in train_loader:
            x = x.to(device)
            y = y.to(torch.float32).to(device)
            with torch.no_grad():
                latent, mu, sigma = vae.encoder(x)
            mean, std = model(latent)
            action = torch.normal(mean=mean, std=std)
            loss = loss_fn(action, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_total_loss += loss.item()

        train_total_loss = train_total_loss / len(train_loader.dataset)
        tqdm.write("EPOCH:{}   train_total_loss:{:.4f}".format(epoch, train_total_loss))

        if (epoch + 1) % 25 == 0:
            model.eval()
            valid_total_loss = 0.0
            with torch.no_grad():
                for x, _, y in valid_loader:
                    x = x.to(device)
                    y = y.to(device)
                    latent, mu, sigma = vae.encoder(x)

                    mean, std = model(latent)
                    action = torch.normal(mean=mean, std=std)
                    loss = loss_fn(action, y)

                    valid_total_loss += loss.item()
                valid_total_loss = valid_total_loss / len(valid_loader.dataset)
            tqdm.write("==================================================")
            tqdm.write(
                "EPOCH:{}   valid_total_loss:{:.4f}".format(epoch, valid_total_loss)
            )
            if valid_total_loss < min_valid_total_loss:
                min_valid_total_loss = valid_total_loss
                model.save("expert_epoch_%d.pth" % (epoch + 1))
                tqdm.write("save model...")
            tqdm.write("==================================================")


if __name__ == '__main__':
    main()
