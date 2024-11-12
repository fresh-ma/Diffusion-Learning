import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
        self.n_feat = n_feat
        
        self.down0 = nn.Sequential(
            Conv3(in_channels, n_feat),
            nn.MaxPool2d(2),
        )
        self.down1 = nn.Sequential(
            Conv3(n_feat, n_feat * 2),
            nn.MaxPool2d(2),
        )
        self.down2 = nn.Sequential(
            Conv3(n_feat * 2, n_feat * 4),
            nn.MaxPool2d(2),
        )
        # add time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, 4 * n_feat),
            nn.Sigmoid(),
            nn.Linear(n_feat * 4, n_feat * 4),
        )
        # Unet bottom block
        self.bottom = nn.Sequential(
            Conv3(n_feat * 4, n_feat * 8),
            nn.ConvTranspose2d(n_feat * 8, n_feat * 4, kernel_size=2, stride=2),
        )
        self.up2 = nn.Sequential(
            Conv3(n_feat * 8, n_feat * 4),
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=2, stride=2),
        )
        self.up1 = nn.Sequential(
            Conv3(n_feat * 4, n_feat * 2),
            nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=2, stride=2),
        )
        self.up0 = nn.Sequential(
            Conv3(n_feat * 2, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, out_channels, kernel_size=1, stride=1),
        )

    def corp_tensor(self, x: torch.Tensor, y: torch.Tensor):
        # 中心裁剪/复制x使其shape与y相同
        x_len = x.size()[2]
        y_len = y.size()[2]
        if x_len >= y_len:
            delta = (x_len - y_len) // 2
            if (x_len - y_len) % 2 == 1:
                return x[:, :, delta: x_len - delta - 1, delta: x_len - delta - 1]
            else:
                return x[:, :, delta: x_len - delta, delta: x_len - delta]
        else:
            delta = (y_len - x_len) // 2
            padding = (delta, delta, delta, delta) if (y_len - x_len) % 2 == 0 else (delta, delta + 1, delta, delta + 1)
            return F.pad(x, padding, "constant", 0)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        print("x.shape: ", x.shape)
        down0 = self.down0(x)
        print("down0.shape: ", down0.shape)
        down1 = self.down1(down0)
        print("down1.shape", down1.shape)
        down2 = self.down2(down1)
        print("down2.shape", down2.shape)
        
        t = self.time_emb(torch.tensor([float(t)])).view(-1, self.n_feat * 4, 1, 1)
        print("t.shape", t.shape)
        
        up2 = self.bottom(down2 + t)
        print("up2.shape", up2.shape)       # torch.Size([16, 256, 34, 34])
        
        
        down2 = self.corp_tensor(down2, up2)
        print("corped_down2.shape", down2.shape)
        up1 = self.up2(torch.cat((up2 + t, down2), 1))
        print("up1.shape", up1.shape)
        down1 = self.corp_tensor(down1, up1)
        print("corped_down1.shape", down1.shape)
        up0 = self.up1(torch.cat((up1, down1), 1))
        print("up0.shape", up0.shape)
        down0 = self.corp_tensor(down0, up0)
        print("corped_down0.shape", down0.shape)
        output = self.up0(torch.cat((up0, down0), 1))
        
        return output