from typing import Tuple
from torch import nn, Tensor
import torch

class ScalarHead(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(feature_dim, 1))
    
    def forward(self, features:Tensor) -> Tensor:
        return self.model(features)
    

class VectorHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, action_dim))
            # nn.Tanh())


    def forward(self, features:Tensor) -> Tensor:
        return self.model(features)
    

class IndependentStdHead(VectorHead):
    def __init__(self, feature_dim: int, action_dim: int, initial_log_std: float):
        super().__init__(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * initial_log_std)


    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        mu = super().forward(features)
        std = self.log_std.expand_as(mu).exp()

        return mu, std
    

class StateDependentGaussianHead(VectorHead):
    def __init__(self, feature_dim: int, action_dim: int, log_std_min: float, log_std_max: float):
        super().__init__(feature_dim, action_dim*2)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min


    def forward(self, features: Tensor) -> Tensor:
        output = super().forward(features)
        mu, log_std = output.chunk(2, dim=-1)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return mu, std
    

class SACHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.fc = nn.Linear(feature_dim, action_dim * 2)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        output = self.fc(features)
        mu, log_std = output.chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        std = log_std.exp()

        return mu, std