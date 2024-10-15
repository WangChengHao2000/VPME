import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.vae.vae_cnn import CNNVariationalAutoencoder

BATCH_SIZE = 1
LATENT_SPACE = 128

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main():
    data_dir = 'data/vae_images/'

    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.ImageFolder(data_dir + 'test', transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = CNNVariationalAutoencoder(latent_dims=LATENT_SPACE)
    model.load("best.pth")

    count = 1
    with torch.no_grad():
        for x, _ in tqdm(test_loader):
            x = x.to(device)
            x_hat = model(x)[0]
            x = x.cpu()
            x = x.squeeze(0)
            x_hat = x_hat.cpu()
            x_hat = x_hat.squeeze(0)
            transform = transforms.ToPILImage()
            img = torch.cat((x, x_hat), dim=2)

            img = transform(img)

            image_filename = str(count) + '.png'
            img.save('checkpoints/reconstructed/' + image_filename)
            count += 1


if __name__ == "__main__":
    main()
