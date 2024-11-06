import torch
from torch import nn

import numpy as np


class DDPM(nn.Module):
    def __init__(self, betas: list, step_T: int,
                 noise_predictor: nn.Module,
                 criterion: nn.Module = nn.MSELoss()) -> None:
        super().__init__()
        
        self.noise_predictor = noise_predictor
        self.criterion = criterion
        self.T = step_T
        
        assert step_T > 0, "must meet: step_T > 0"
        assert 0 < betas[0] < betas[1] < 1, "must meet: 0 < beta1 < beta2 < 1"
        
        self.beta_t = (betas[0] - betas[1]) * [i / step_T for i in range(1, step_T + 1)] + betas[0]
        self.alpha_t = 1 - self.beta_t
        self.alphabar_t = np.cumprod(self.alpha_t)
        
    def forward(self, x) -> torch.Tensor:
        noise = torch.randn_like(x).to(x.device)
        ts = torch.randint(0, self.T + 1, size = (x.shape[0], ))
        
        add_noise_x = torch.sqrt(self.alphabar_t[ts]) * x + torch.sqrt(1 - self.alphabar_t[ts]) * noise
        predict = self.noise_predictor(add_noise_x, ts / self.T)
        
        return self.criterion(noise, predict)
    
    def sample(self, n, size, device = 'cuda') -> torch.Tensor:
        x = torch.randn_like(n, *size).to(device)
        
        for t in range(self.T, 0, -1):
            z = torch.randn_like(n, *size).to(device) if t > 1 else 0
            noise = self.noise_predictor(x, torch.tensor(t / self.T).to(device).repeat(n, 1))
            x = (x - (1 - self.alpha_t) / torch.sqrt(1 - self.alphabar_t) * noise) / torch.sqrt(self.alpha_t) +  torch.sqrt(self.beta_t) * z
            
        return x
        
        
        