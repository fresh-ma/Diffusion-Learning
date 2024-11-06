import numpy as np
import torch
from torch import nn

class Conv3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int) -> None:
        super(Unet, self).__init__()
        
        self.down0 = nn.Sequential(
            Conv3(in_channels, n_feat),
            nn.MaxPool2d(n_feat),
        )
        self.down1 = nn.Sequential(
            Conv3(n_feat, n_feat * 2),
            nn.MaxPool2d(n_feat * 2),
        )
        self.down2 = nn.Sequential(
            Conv3(n_feat * 2, n_feat * 4),
            nn.MaxPool2d(n_feat * 4),
        )
        self.down3 = nn.Sequential(
            Conv3(n_feat * 4, n_feat * 8),
            nn.MaxPool2d(n_feat * 8),
        )
        self.time_emb = nn.Sequential(
            nn.Linear(1, 2 * n_feat),
            torch.sin(),
            nn.Linear(n_feat * 2, n_feat * 2),
        )
        self.up3 = nn.Sequential(
            Conv3(n_feat * 8, n_feat * 8),
            nn.ConvTranspose2d(n_feat * 8, n_feat * 4)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x