from torch import nn, Tensor


class ValueHead(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.model = nn.Linear(feature_dim, 1)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)