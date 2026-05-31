import torch
import torch.nn as nn

from src.constants import RANDOM_STATE

DROPOUT = 0.3

torch.manual_seed(RANDOM_STATE)


class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim) → (batch, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return self.classifier(x).squeeze(1)
