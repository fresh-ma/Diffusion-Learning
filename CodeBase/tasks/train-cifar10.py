from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import os
import sys
sys.path.append("../")
from models.Unet import Unet
from models.DDPM import DDPM



def train_cifar10(
    n_epoch: int = 100, device: str = "cuda:0", load_path: str = None
) -> None:

    ddpm = DDPM(betas=[1e-4, 0.02], noise_predictor=Unet(3, 3, n_feat=64), step_T=1000)

    if load_path != None and os.path.exists(load_path):
        ddpm.load_state_dict(torch.load(load_path))

    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 32, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"../outputs/task-1/ddpm_sample_cifar{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"../outputs/task-1/ddpm_cifar.pth")


if __name__ == "__main__":
    train_cifar10()