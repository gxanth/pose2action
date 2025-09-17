import torch.nn as nn


class MLPTemporal(nn.Module):
    def __init__(self, T=32, feats=66, num_classes=3):
        super().__init__()
        d = T * feats
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # x: (B, T, feats)
        return self.net(x)
