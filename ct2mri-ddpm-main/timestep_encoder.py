"""
Timestep encoder: sinusoidal position embedding of the diffusion step followed
by a small MLP, producing the vector each UNet residual block is conditioned on.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t, dim, max_period=10000):
    """
    Transformer-style sinusoidal embedding.
    t: (B,) integer or float timesteps -> (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEncoder(nn.Module):
    def __init__(self, base_dim, emb_dim):
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t):
        return self.mlp(sinusoidal_embedding(t, self.base_dim))
