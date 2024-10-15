import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from tqdm import tqdm

from model.vae.vae_cnn import CNNVariationalAutoencoder

NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LATENT_SPACE = 128

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main():
    data_dir = 'data/vae_images'
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_data = datasets.ImageFolder(data_dir + '/train', transform=transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=transforms)

    m = len(train_data)
    train_data, val_data = random_split(train_data, [int(m - m * 0.2), int(m * 0.2)])

    # Data Loading

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = CNNVariationalAutoencoder(latent_dims=LATENT_SPACE)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    min_val_loss = 5000000

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss = train(model, train_loader, optim)
        val_loss = test(model, valid_loader)
        tqdm.write(
            'EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS, train_loss, val_loss))

        if (epoch + 1) % 10 == 0 and val_loss < min_val_loss:
            min_val_loss = val_loss
            model.save("best.pth")
            tqdm.write("save model...")


def train(model, loader, optim):
    model.train()
    train_loss = 0.0
    for (x, _) in loader:
        x = x.to(device)
        x_recon, mu, sigma = model(x)
        kl = 0.5 * (sigma + mu ** 2 - torch.log(sigma) - 1).sum()
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        loss = recon_loss + kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
    return train_loss / len(loader.dataset)


def test(model, loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, mu, sigma = model(x)
            kl = 0.5 * (sigma + mu ** 2 - torch.log(sigma) - 1).sum()
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
            loss = recon_loss + kl
            val_loss += loss.item()
    return val_loss / len(loader.dataset)


if __name__ == '__main__':
    main()
