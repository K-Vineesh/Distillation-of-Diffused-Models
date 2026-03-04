import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Encodes the diffusion timestep t as a vector.
    Without this, the model doesn't know how noisy the input is.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class SimpleUNet(nn.Module):
    """
    UNet with:
    - Sinusoidal timestep embeddings (so model knows where in diffusion process it is)
    - Label embeddings with null label support for CFG
    """

    def __init__(self, num_classes=10, time_emb_dim=64):
        super().__init__()

        self.num_classes = num_classes
        self.null_label = num_classes  # label 10 = unconditional token for MNIST

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Label embedding: num_classes + 1 to include null label
        self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        # Encoder
        # Input: 1 channel (grayscale MNIST)
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        # Decoder
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

        # Time + label conditioning injected into channel dim
        self.time_label_proj1 = nn.Linear(time_emb_dim, 64)
        self.time_label_proj2 = nn.Linear(time_emb_dim, 128)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, labels, t):
        """
        x:      (B, 1, H, W)  noisy image
        labels: (B,)           class labels (use null_label for unconditional)
        t:      (B,)           timestep indices
        """
        # Compute combined time + label embedding
        t_emb = self.time_mlp(t)                    # (B, time_emb_dim)
        l_emb = self.label_emb(labels)              # (B, time_emb_dim)
        cond = t_emb + l_emb                        # (B, time_emb_dim)

        # Encoder
        h1 = F.silu(self.conv1(x))                 # (B, 64, H, W)
        h1 = h1 + self.time_label_proj1(cond).unsqueeze(-1).unsqueeze(-1)
        h1_pooled = self.pool(h1)                   # (B, 64, H/2, W/2)

        h2 = F.silu(self.conv2(h1_pooled))         # (B, 128, H/2, W/2)
        h2 = h2 + self.time_label_proj2(cond).unsqueeze(-1).unsqueeze(-1)

        # Decoder
        h3 = self.up(h2)                            # (B, 128, H, W)
        h3 = F.silu(self.conv3(h3))                 # (B, 64, H, W)

        out = self.conv4(h3)                        # (B, 1, H, W)

        return out