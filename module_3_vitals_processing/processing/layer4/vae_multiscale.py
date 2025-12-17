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


class TransposeConvBlock(nn.Module):
    """ConvTranspose1D + BatchNorm + LeakyReLU + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        padding = kernel_size // 2
        output_padding = stride - 1
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.conv(x))))


class MultiScaleDecoder(nn.Module):
    """Multi-scale convolutional decoder with 4 parallel branches."""

    def __init__(
        self,
        output_dim: int = 7,
        seq_len: int = 745,
        latent_dim: int = 32,
        base_channels: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        self.output_dim = output_dim
        self.seq_len = seq_len
        c = base_channels
        self.c = c

        # Project latent to initial representation
        self.fc = nn.Linear(latent_dim, c * 4 * 8)  # 4 branches, 8 initial length

        # Branch 1: Local reconstruction
        self.branch_local = nn.Sequential(
            TransposeConvBlock(c, c, kernel_size=5, stride=2, dropout=dropout),
            TransposeConvBlock(c, c // 2, kernel_size=5, stride=2, dropout=dropout),
            TransposeConvBlock(c // 2, c // 2, kernel_size=5, stride=2, dropout=dropout),
            TransposeConvBlock(c // 2, c // 4, kernel_size=5, stride=2, dropout=dropout),
            TransposeConvBlock(c // 4, c // 4, kernel_size=5, stride=2, dropout=dropout),
            TransposeConvBlock(c // 4, c // 4, kernel_size=5, stride=2, dropout=dropout),
        )
        self.out_local = nn.Conv1d(c // 4, output_dim, kernel_size=3, padding=1)

        # Branch 2: Hourly reconstruction
        self.branch_hourly = nn.Sequential(
            TransposeConvBlock(c, c, kernel_size=15, stride=4, dropout=dropout),
            TransposeConvBlock(c, c // 2, kernel_size=15, stride=4, dropout=dropout),
            TransposeConvBlock(c // 2, c // 4, kernel_size=15, stride=4, dropout=dropout),
        )
        self.out_hourly = nn.Conv1d(c // 4, output_dim, kernel_size=3, padding=1)

        # Branch 3: Daily reconstruction
        self.branch_daily = nn.Sequential(
            TransposeConvBlock(c, c, kernel_size=31, stride=8, dropout=dropout),
            TransposeConvBlock(c, c // 2, kernel_size=31, stride=8, dropout=dropout),
        )
        self.out_daily = nn.Conv1d(c // 2, output_dim, kernel_size=3, padding=1)

        # Branch 4: Multi-day reconstruction
        self.branch_multiday = nn.Sequential(
            TransposeConvBlock(c, c // 2, kernel_size=63, stride=16, dropout=dropout),
            TransposeConvBlock(c // 2, c // 4, kernel_size=63, stride=8, dropout=dropout),
        )
        self.out_multiday = nn.Conv1d(c // 4, output_dim, kernel_size=3, padding=1)

        # Final merge
        self.merge = nn.Conv1d(output_dim * 4, output_dim, kernel_size=1)

    def _adjust_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Adjust tensor length to target via trimming or padding."""
        current_len = x.shape[2]
        if current_len > target_len:
            return x[:, :, :target_len]
        elif current_len < target_len:
            return F.pad(x, (0, target_len - current_len), mode='replicate')
        return x

    def forward(
        self,
        z: torch.Tensor,
        return_branches: bool = False
    ) -> torch.Tensor:
        """Decode latent to reconstruction.

        Args:
            z: (batch, latent_dim)
            return_branches: If True, also return per-branch outputs

        Returns:
            recon: (batch, seq_len, output_dim)
            branch_outputs: Optional list of per-branch reconstructions
        """
        batch_size = z.shape[0]

        # Project to initial representation
        h = self.fc(z)
        h = h.view(batch_size, self.c * 4, 8)  # (batch, c*4, 8)

        # Split for branches
        h_local = h[:, :self.c, :]
        h_hourly = h[:, self.c:self.c*2, :]
        h_daily = h[:, self.c*2:self.c*3, :]
        h_multiday = h[:, self.c*3:, :]

        # Decode each branch
        out_local = self.out_local(self.branch_local(h_local))
        out_hourly = self.out_hourly(self.branch_hourly(h_hourly))
        out_daily = self.out_daily(self.branch_daily(h_daily))
        out_multiday = self.out_multiday(self.branch_multiday(h_multiday))

        # Adjust lengths
        out_local = self._adjust_length(out_local, self.seq_len)
        out_hourly = self._adjust_length(out_hourly, self.seq_len)
        out_daily = self._adjust_length(out_daily, self.seq_len)
        out_multiday = self._adjust_length(out_multiday, self.seq_len)

        # Merge
        merged = torch.cat([out_local, out_hourly, out_daily, out_multiday], dim=1)
        recon = self.merge(merged)  # (batch, output_dim, seq_len)

        # Transpose to (batch, seq_len, output_dim)
        recon = recon.transpose(1, 2)

        if return_branches:
            branches = [
                out_local.transpose(1, 2),
                out_hourly.transpose(1, 2),
                out_daily.transpose(1, 2),
                out_multiday.transpose(1, 2)
            ]
            return recon, branches

        return recon


class MultiScaleConv1DVAE(nn.Module):
    """Multi-Scale 1D Convolutional VAE.

    Prevents posterior collapse through:
    - Multi-scale feature extraction (local to multi-day)
    - Per-branch reconstruction loss
    - Symmetric encoder-decoder design
    """

    def __init__(
        self,
        input_dim: int = 7,
        seq_len: int = 745,
        latent_dim: int = 32,
        base_channels: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.encoder = MultiScaleEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            base_channels=base_channels,
            dropout=dropout
        )

        self.decoder = MultiScaleDecoder(
            output_dim=input_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            base_channels=base_channels,
            dropout=dropout
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_branches: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim)
            mask: Optional observation mask
            return_branches: If True, also return per-branch reconstructions

        Returns:
            recon: (batch, seq_len, input_dim)
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        mu, logvar = self.encoder(x, mask)
        z = self.reparameterize(mu, logvar)

        if return_branches:
            recon, branches = self.decoder(z, return_branches=True)
            return recon, mu, logvar, branches

        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode only (for inference)."""
        mu, logvar = self.encoder(x, mask)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
