import torch
from torch import nn
from torch.nn import functional as F
from utils.class_registry import ClassRegistry

gens_registry = ClassRegistry()
discs_registry = ClassRegistry()


@gens_registry.add_to_registry(name="base_gen")
class VerySimpleGenarator(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.z_dim = model_config["z_dim"]
        self.hidden_dim = model_config["hidden_dim"]
        self.blocks_num = model_config["blocks_num"]

        self.z_to_x = nn.Linear(self.z_dim, self.hidden_dim * 4 * 4)

        self.blocks = nn.ModuleList([])
        for i in range(self.blocks_num):
            self.blocks.append(VerySimpleBlock(self.hidden_dim, self.hidden_dim))

        self.to_rgb_block = VerySimpleBlock(self.hidden_dim, 3, use_batchnorm=False)

    def forward(self, z):
        x = self.z_to_x(z).reshape(-1, self.hidden_dim, 4, 4)

        for block in self.blocks:
            x = block(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.to_rgb_block(x)
        return x


@discs_registry.add_to_registry(name="base_disc")
class VerySimpleDiscriminator(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config["hidden_dim"]
        self.blocks_num = model_config["blocks_num"]

        self.from_rgb_block = VerySimpleBlock(3, self.hidden_dim, use_batchnorm=False)

        self.blocks = nn.ModuleList([])
        for i in range(self.blocks_num):
            self.blocks.append(VerySimpleBlock(self.hidden_dim, self.hidden_dim))

        self.to_label = nn.Conv2d(self.hidden_dim, 1, kernel_size=4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.from_rgb_block(x)

        for block in self.blocks:
            x = block(x)
            x = F.interpolate(x, scale_factor=0.5, mode="nearest")
        x = self.to_label(x)
        x = self.sigmoid(x)
        return x


class VerySimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Применяем BatchNorm
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)  # Применяем BatchNorm
        x = self.activation(x)
        return x
