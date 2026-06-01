import torch
import torch.nn as nn

from src.constants import RANDOM_STATE

HIDDEN_DIM = 512
DROPOUT = 0.3

torch.manual_seed(RANDOM_STATE)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
