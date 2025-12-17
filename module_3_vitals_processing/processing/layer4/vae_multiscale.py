"""Multi-Scale Conv1D VAE for vital trajectory embeddings.

Four parallel convolutional branches capture patterns at different temporal scales:
- Local (k=3,5): Beat-to-beat variability
- Hourly (k=15,31): Hour-scale patterns
- Daily (k=63,127): Daily trends
- Multi-day (k=255): Multi-day trajectories

Forces latent to encode multi-resolution information, preventing posterior collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Conv1D + BatchNorm + LeakyReLU + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.conv(x))))


class MultiScaleEncoder(nn.Module):
    """Multi-scale convolutional encoder with 4 parallel branches."""

    def __init__(
        self,
        input_dim: int = 7,
        seq_len: int = 745,
        latent_dim: int = 32,
        base_channels: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        c = base_channels

        # Branch 1: Local patterns (k=3, 5)
        self.branch_local = nn.Sequential(
            ConvBlock(input_dim, c // 2, kernel_size=3, stride=2, dropout=dropout),
            ConvBlock(c // 2, c, kernel_size=5, stride=2, dropout=dropout),
            ConvBlock(c, c, kernel_size=5, stride=2, dropout=dropout),
            nn.AdaptiveAvgPool1d(1)
        )

        # Branch 2: Hourly patterns (k=15, 31)
        self.branch_hourly = nn.Sequential(
            ConvBlock(input_dim, c // 2, kernel_size=15, stride=4, dropout=dropout),
            ConvBlock(c // 2, c, kernel_size=31, stride=4, dropout=dropout),
            nn.AdaptiveAvgPool1d(1)
        )

        # Branch 3: Daily patterns (k=63, 127)
        self.branch_daily = nn.Sequential(
            ConvBlock(input_dim, c // 2, kernel_size=63, stride=8, dropout=dropout),
            ConvBlock(c // 2, c, kernel_size=127, stride=8, dropout=dropout),
            nn.AdaptiveAvgPool1d(1)
        )

        # Branch 4: Multi-day patterns (k=255)
        self.branch_multiday = nn.Sequential(
            ConvBlock(input_dim, c, kernel_size=255, stride=16, dropout=dropout),
            nn.AdaptiveAvgPool1d(1)
        )

        # Merge and project to latent
        self.fc_merge = nn.Linear(c * 4, c * 2)
        self.fc_mu = nn.Linear(c * 2, latent_dim)
        self.fc_logvar = nn.Linear(c * 2, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution.

        Args:
            x: (batch, seq_len, input_dim)
            mask: Optional observation mask (unused)

        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Transpose to (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Extract multi-scale features
        h_local = self.branch_local(x).squeeze(-1)      # (batch, c)
        h_hourly = self.branch_hourly(x).squeeze(-1)    # (batch, c)
        h_daily = self.branch_daily(x).squeeze(-1)      # (batch, c)
        h_multiday = self.branch_multiday(x).squeeze(-1) # (batch, c)

        # Concatenate
        h = torch.cat([h_local, h_hourly, h_daily, h_multiday], dim=1)  # (batch, c*4)

        # Project to latent
        h = F.leaky_relu(self.fc_merge(h), 0.2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
