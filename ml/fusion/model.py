import torch
import torch.nn as nn


class CardioFusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(128 + 768 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, ecg, nlp, cv):

        x = torch.cat(
            [ecg, nlp, cv],
            dim=1
        )

        return self.network(x)