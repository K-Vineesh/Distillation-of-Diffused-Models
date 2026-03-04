# diffusion/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.label_emb = nn.Embedding(num_classes, 32)

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, labels):

        emb = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)

        emb = emb.expand(-1, -1, x.shape[2], x.shape[3])

        x = torch.cat([x, emb[:, :1]], dim=1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.up(x)

        x = F.relu(self.conv3(x))

        x = self.conv4(x)

        return x