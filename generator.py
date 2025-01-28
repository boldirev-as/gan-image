import numpy as np
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_shape=1024, img_shape=128):
        super(Generator, self).__init__()

        self.hidden_shape = hidden_shape
        self.img_shape = img_shape

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, hidden_shape * 8 * 8),
            nn.BatchNorm1d(hidden_shape * 8 * 8),
            nn.ReLU()
        )

        self.generator = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=hidden_shape, out_channels=512, kernel_size=5, stride=1, padding=2),  # 'same' padding
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5, stride=1, padding=2),  # Выход RGB
            nn.Tanh()
        )

    def forward(self, z):
        x = self.upsampler(z)

        x = x.view(-1, self.hidden_shape, 8, 8)

        return self.generator(x)


if __name__ == '__main__':
    latent_dim = 100
    generator = Generator(latent_dim)
    output = generator(Tensor(np.random.randn(8, 100)))
    print(output.size())
