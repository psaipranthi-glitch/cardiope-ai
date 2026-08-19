import torch
import torch.nn as nn
from torchvision import models


class VisionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.model.fc = nn.Identity()

    def forward(self, x):

        return self.model(x)