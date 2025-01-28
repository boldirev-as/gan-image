import platform
import os
from PIL import Image
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CustomDataset
from generator import Generator
from discriminator import Discriminator


def infer_gan(generator, num_samples, latent_dim, device, output_dir='generated_images'):
    generator.eval()

    z = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        generated_images = generator(z)

    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        img = generated_images[i].cpu().detach().numpy()
        img = (img * 0.5 + 0.5) * 255
        img = img.astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))

        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f'generated_{i}.png'))

    return generated_images


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def train_gan(data_loader, generator, discriminator, g_optimizer, d_optimizer, criterion, latent_dim, epochs):
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for i, real_images in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            z = torch.randn(batch_size, latent_dim, device=device)
            generated_images = generator(z)

            d_real = discriminator(real_images)
            d_real_loss = criterion(d_real, valid)

            d_fake = discriminator(generated_images.detach())
            d_fake_loss = criterion(d_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            g_loss = criterion(discriminator(generated_images), valid)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 15 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(data_loader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        infer_gan(generator, 1, latent_dim, device, f'generated_images_2/epoch_{epoch}/')

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')


latent_dim = 128
batch_size = 64
num_epochs = 300
learning_rate = 0.0002

if platform.system() == 'Darwin':
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomDataset(root_dir='afhq/train/dog/', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load('discriminator.pth', map_location=device))
discriminator.eval()

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

criterion = torch.nn.BCELoss()

generator.apply(weights_init)
discriminator.apply(weights_init)

train_gan(dataloader, generator, discriminator, optimizer_G, optimizer_D, criterion, latent_dim, num_epochs)

num_samples = 40
generated_images = infer_gan(generator, num_samples, latent_dim, device)

print(f"Generated images shape: {generated_images.size()}")
