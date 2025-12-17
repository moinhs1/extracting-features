# Multi-Scale Conv1D VAE Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace collapsed LSTM-VAE with Multi-Scale Conv1D VAE that captures local, hourly, daily, and multi-day temporal patterns.

**Architecture:** Four parallel convolutional branches with different kernel sizes extract multi-scale features, concatenated before latent bottleneck. Symmetric decoder with per-branch reconstruction loss forces latent to encode all scales.

**Tech Stack:** PyTorch, numpy, h5py. Builds on existing Layer 4 infrastructure.

---

## Task 1: Create Multi-Scale Encoder

**Files:**
- Create: `processing/layer4/vae_multiscale.py`
- Test: `tests/test_layer4/test_vae_multiscale.py`

**Step 1: Write failing test for encoder output shapes**

Create `tests/test_layer4/test_vae_multiscale.py`:

```python
"""Tests for Multi-Scale Conv1D VAE."""
import pytest
import torch


class TestMultiScaleEncoder:
    """Test encoder produces correct shapes."""

    def test_encoder_output_shape(self):
        from processing.layer4.vae_multiscale import MultiScaleEncoder

        encoder = MultiScaleEncoder(
            input_dim=7,
            seq_len=745,
            latent_dim=32,
            base_channels=64
        )

        x = torch.randn(4, 745, 7)  # batch=4
        mu, logvar = encoder(x)

        assert mu.shape == (4, 32), f"Expected (4, 32), got {mu.shape}"
        assert logvar.shape == (4, 32), f"Expected (4, 32), got {logvar.shape}"

    def test_encoder_branches_exist(self):
        from processing.layer4.vae_multiscale import MultiScaleEncoder

        encoder = MultiScaleEncoder(
            input_dim=7,
            seq_len=745,
            latent_dim=32,
            base_channels=64
        )

        assert hasattr(encoder, 'branch_local')
        assert hasattr(encoder, 'branch_hourly')
        assert hasattr(encoder, 'branch_daily')
        assert hasattr(encoder, 'branch_multiday')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer4/test_vae_multiscale.py -v`
Expected: FAIL with "No module named 'processing.layer4.vae_multiscale'"

**Step 3: Write encoder implementation**

Create `processing/layer4/vae_multiscale.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_layer4/test_vae_multiscale.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add processing/layer4/vae_multiscale.py tests/test_layer4/test_vae_multiscale.py
git commit -m "feat(vae): add multi-scale encoder with 4 temporal branches"
```

---

## Task 2: Create Multi-Scale Decoder

**Files:**
- Modify: `processing/layer4/vae_multiscale.py`
- Modify: `tests/test_layer4/test_vae_multiscale.py`

**Step 1: Write failing test for decoder**

Add to `tests/test_layer4/test_vae_multiscale.py`:

```python
class TestMultiScaleDecoder:
    """Test decoder reconstructs correct shapes."""

    def test_decoder_output_shape(self):
        from processing.layer4.vae_multiscale import MultiScaleDecoder

        decoder = MultiScaleDecoder(
            output_dim=7,
            seq_len=745,
            latent_dim=32,
            base_channels=64
        )

        z = torch.randn(4, 32)  # batch=4
        recon = decoder(z)

        assert recon.shape == (4, 745, 7), f"Expected (4, 745, 7), got {recon.shape}"

    def test_decoder_branch_outputs(self):
        from processing.layer4.vae_multiscale import MultiScaleDecoder

        decoder = MultiScaleDecoder(
            output_dim=7,
            seq_len=745,
            latent_dim=32,
            base_channels=64
        )

        z = torch.randn(4, 32)
        recon, branch_outputs = decoder(z, return_branches=True)

        assert len(branch_outputs) == 4
        for bo in branch_outputs:
            assert bo.shape[0] == 4  # batch
            assert bo.shape[2] == 7  # output_dim
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer4/test_vae_multiscale.py::TestMultiScaleDecoder -v`
Expected: FAIL with "cannot import name 'MultiScaleDecoder'"

**Step 3: Write decoder implementation**

Add to `processing/layer4/vae_multiscale.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_layer4/test_vae_multiscale.py::TestMultiScaleDecoder -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add processing/layer4/vae_multiscale.py tests/test_layer4/test_vae_multiscale.py
git commit -m "feat(vae): add multi-scale decoder with 4 reconstruction branches"
```

---

## Task 3: Create Full Multi-Scale VAE

**Files:**
- Modify: `processing/layer4/vae_multiscale.py`
- Modify: `tests/test_layer4/test_vae_multiscale.py`

**Step 1: Write failing test for full VAE**

Add to `tests/test_layer4/test_vae_multiscale.py`:

```python
class TestMultiScaleVAE:
    """Test full VAE forward pass."""

    def test_vae_forward(self):
        from processing.layer4.vae_multiscale import MultiScaleConv1DVAE

        vae = MultiScaleConv1DVAE(
            input_dim=7,
            seq_len=745,
            latent_dim=32,
            base_channels=64
        )

        x = torch.randn(4, 745, 7)
        recon, mu, logvar = vae(x)

        assert recon.shape == (4, 745, 7)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_vae_reparameterize(self):
        from processing.layer4.vae_multiscale import MultiScaleConv1DVAE

        vae = MultiScaleConv1DVAE(input_dim=7, seq_len=745, latent_dim=32)

        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)

        z1 = vae.reparameterize(mu, logvar)
        z2 = vae.reparameterize(mu, logvar)

        # Should be different due to sampling
        assert not torch.allclose(z1, z2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer4/test_vae_multiscale.py::TestMultiScaleVAE -v`
Expected: FAIL with "cannot import name 'MultiScaleConv1DVAE'"

**Step 3: Write full VAE implementation**

Add to `processing/layer4/vae_multiscale.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_layer4/test_vae_multiscale.py::TestMultiScaleVAE -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add processing/layer4/vae_multiscale.py tests/test_layer4/test_vae_multiscale.py
git commit -m "feat(vae): add MultiScaleConv1DVAE combining encoder and decoder"
```

---

## Task 4: Add Multi-Scale VAE Loss with Cyclical Annealing

**Files:**
- Modify: `processing/layer4/vae_multiscale.py`
- Modify: `tests/test_layer4/test_vae_multiscale.py`

**Step 1: Write failing test for loss function**

Add to `tests/test_layer4/test_vae_multiscale.py`:

```python
class TestMultiScaleLoss:
    """Test VAE loss with cyclical annealing."""

    def test_loss_basic(self):
        from processing.layer4.vae_multiscale import MultiScaleVAELoss

        loss_fn = MultiScaleVAELoss(beta=1.0, free_bits=0.0)

        recon = torch.randn(4, 745, 7)
        target = torch.randn(4, 745, 7)
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)

        total, recon_loss, kl_loss = loss_fn(recon, target, mu, logvar)

        assert total.ndim == 0  # scalar
        assert recon_loss.ndim == 0
        assert kl_loss.ndim == 0
        assert total >= 0

    def test_loss_free_bits(self):
        from processing.layer4.vae_multiscale import MultiScaleVAELoss

        loss_fn = MultiScaleVAELoss(beta=1.0, free_bits=2.0)

        recon = torch.zeros(4, 745, 7)
        target = torch.zeros(4, 745, 7)
        mu = torch.zeros(4, 32)  # Zero KL
        logvar = torch.zeros(4, 32)

        total, recon_loss, kl_loss = loss_fn(recon, target, mu, logvar)

        # With free_bits=2.0 and zero actual KL, kl_loss should be 0
        assert kl_loss == 0.0

    def test_cyclical_beta(self):
        from processing.layer4.vae_multiscale import MultiScaleVAELoss

        loss_fn = MultiScaleVAELoss(
            beta=0.5,
            free_bits=2.0,
            warmup_epochs=10,
            cycle_epochs=20
        )

        # Epoch 0: beta should be 0
        assert loss_fn.get_beta(0) == 0.0

        # Epoch 5: beta should be 0.25 (halfway through warmup)
        assert abs(loss_fn.get_beta(5) - 0.25) < 0.01

        # Epoch 10: beta should be 0.5 (end of warmup)
        assert abs(loss_fn.get_beta(10) - 0.5) < 0.01

        # Epoch 20: new cycle starts, beta back to 0
        assert loss_fn.get_beta(20) == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer4/test_vae_multiscale.py::TestMultiScaleLoss -v`
Expected: FAIL with "cannot import name 'MultiScaleVAELoss'"

**Step 3: Write loss implementation**

Add to `processing/layer4/vae_multiscale.py`:

```python
class MultiScaleVAELoss(nn.Module):
    """VAE loss with cyclical beta annealing and free bits.

    Features:
    - Per-branch reconstruction loss (optional)
    - Free bits to prevent posterior collapse
    - Cyclical beta annealing to escape local minima
    """

    def __init__(
        self,
        beta: float = 0.5,
        free_bits: float = 2.0,
        warmup_epochs: int = 30,
        cycle_epochs: int = 40
    ):
        super().__init__()
        self.target_beta = beta
        self.free_bits = free_bits
        self.warmup_epochs = warmup_epochs
        self.cycle_epochs = cycle_epochs
        self._current_beta = 0.0

    def get_beta(self, epoch: int) -> float:
        """Get beta value for current epoch with cyclical annealing."""
        # Which cycle are we in?
        cycle_position = epoch % self.cycle_epochs

        # Warmup within cycle
        if cycle_position < self.warmup_epochs:
            beta = self.target_beta * (cycle_position / self.warmup_epochs)
        else:
            beta = self.target_beta

        self._current_beta = beta
        return beta

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        branch_recons: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss.

        Args:
            recon: (batch, seq_len, dim) reconstruction
            target: (batch, seq_len, dim) target
            mu: (batch, latent_dim) mean
            logvar: (batch, latent_dim) log variance
            mask: Optional observation mask
            branch_recons: Optional list of per-branch reconstructions

        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Reconstruction loss
        if mask is not None:
            recon_error = ((recon - target) ** 2) * mask
            recon_loss = recon_error.sum() / (mask.sum() + 1e-8)
        else:
            recon_loss = F.mse_loss(recon, target)

        # Per-branch reconstruction loss (if provided)
        if branch_recons is not None:
            branch_losses = []
            for branch_recon in branch_recons:
                if mask is not None:
                    br_error = ((branch_recon - target) ** 2) * mask
                    br_loss = br_error.sum() / (mask.sum() + 1e-8)
                else:
                    br_loss = F.mse_loss(branch_recon, target)
                branch_losses.append(br_loss)
            recon_loss = (recon_loss + sum(branch_losses)) / (1 + len(branch_losses))

        # KL divergence with free bits
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (batch, latent_dim)

        # Apply free bits: only penalize KL above threshold
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim - self.free_bits, min=0)

        kl_loss = kl_per_dim.mean()

        # Total loss
        total_loss = recon_loss + self._current_beta * kl_loss

        return total_loss, recon_loss, kl_loss
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_layer4/test_vae_multiscale.py::TestMultiScaleLoss -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add processing/layer4/vae_multiscale.py tests/test_layer4/test_vae_multiscale.py
git commit -m "feat(vae): add MultiScaleVAELoss with cyclical annealing and free bits"
```

---

## Task 5: Update VAE Trainer to Support Multi-Scale VAE

**Files:**
- Modify: `processing/layer4/vae_trainer.py`

**Step 1: Write failing test for trainer model selection**

Add to `tests/test_layer4/test_vae_multiscale.py`:

```python
class TestVAETrainerIntegration:
    """Test VAE trainer with multi-scale model."""

    def test_trainer_creates_multiscale_model(self):
        from processing.layer4.vae_trainer import VAEBuilder
        from pathlib import Path
        import tempfile

        # Create dummy data
        with tempfile.TemporaryDirectory() as tmpdir:
            # Skip actual training, just test model creation
            builder = VAEBuilder(
                layer2_tensors_path=Path(tmpdir) / "dummy.h5",
                output_dir=Path(tmpdir),
                model_type='multiscale_conv1d',
                latent_dim=32
            )
            assert builder.model_type == 'multiscale_conv1d'
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer4/test_vae_multiscale.py::TestVAETrainerIntegration -v`
Expected: FAIL with "unexpected keyword argument 'model_type'"

**Step 3: Update VAE trainer**

Modify `processing/layer4/vae_trainer.py`:

At imports section, add:
```python
try:
    from .vae_model import LSTMVAE, VAELoss
    from .vae_multiscale import MultiScaleConv1DVAE, MultiScaleVAELoss
except ImportError:
    from vae_model import LSTMVAE, VAELoss
    from vae_multiscale import MultiScaleConv1DVAE, MultiScaleVAELoss
```

Update `VAEBuilder.__init__`:
```python
def __init__(
    self,
    layer2_tensors_path: Path,
    output_dir: Path,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    n_layers: int = 2,
    beta: float = 0.5,
    beta_warmup_epochs: int = 30,
    free_bits: float = 2.0,
    model_type: str = 'multiscale_conv1d',  # NEW: 'lstm' or 'multiscale_conv1d'
    base_channels: int = 64,                 # NEW: for conv models
    cycle_epochs: int = 40                   # NEW: cyclical annealing period
):
    """Initialize VAE builder.

    Args:
        layer2_tensors_path: Path to hourly_tensors.h5
        output_dir: Output directory
        latent_dim: Latent space dimension
        hidden_dim: LSTM hidden dimension (for LSTM model)
        n_layers: Number of LSTM layers (for LSTM model)
        beta: Target KL weight
        beta_warmup_epochs: Epochs to anneal beta from 0 to target
        free_bits: Minimum KL per dimension
        model_type: 'lstm' or 'multiscale_conv1d'
        base_channels: Base channel count for conv models
        cycle_epochs: Cyclical annealing period
    """
    self.layer2_path = Path(layer2_tensors_path)
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)

    self.latent_dim = latent_dim
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.beta = beta
    self.beta_warmup_epochs = beta_warmup_epochs
    self.free_bits = free_bits
    self.model_type = model_type
    self.base_channels = base_channels
    self.cycle_epochs = cycle_epochs

    self.trainer: Optional[VAETrainer] = None
    self.latents: Optional[Dict] = None
    self.patient_ids: Optional[np.ndarray] = None
```

Update `VAEBuilder.build` method's model creation:
```python
def build(
    self,
    epochs: int = 150,
    batch_size: int = 64,
    patience: int = 30,
    learning_rate: float = 1e-3
) -> Dict[str, np.ndarray]:
    """Train VAE and extract latents."""
    values, masks = self._load_data()

    # Create model based on type
    if self.model_type == 'multiscale_conv1d':
        model = MultiScaleConv1DVAE(
            input_dim=values.shape[2],
            seq_len=values.shape[1],
            latent_dim=self.latent_dim,
            base_channels=self.base_channels,
            dropout=0.2
        )
        loss_fn = MultiScaleVAELoss(
            beta=self.beta,
            free_bits=self.free_bits,
            warmup_epochs=self.beta_warmup_epochs,
            cycle_epochs=self.cycle_epochs
        )
    else:  # lstm
        model = LSTMVAE(
            input_dim=values.shape[2],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            seq_len=values.shape[1],
            n_layers=self.n_layers
        )
        loss_fn = VAELoss(
            beta=0.0 if self.beta_warmup_epochs > 0 else self.beta,
            free_bits=self.free_bits
        )

    # Rest of training logic...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_layer4/test_vae_multiscale.py::TestVAETrainerIntegration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add processing/layer4/vae_trainer.py tests/test_layer4/test_vae_multiscale.py
git commit -m "feat(vae): add model_type selection to VAEBuilder for multi-scale support"
```

---

## Task 6: Update Layer 4 Config

**Files:**
- Modify: `processing/layer4_builder.py`

**Step 1: Update config defaults**

In `processing/layer4_builder.py`, update the VAE config section:

```python
'vae': {
    'model_type': 'multiscale_conv1d',
    'latent_dim': 32,
    'base_channels': 64,
    'hidden_dim': 128,       # For LSTM fallback
    'n_layers': 2,           # For LSTM fallback
    'beta': 0.5,
    'beta_warmup_epochs': 30,
    'cycle_epochs': 40,
    'free_bits': 2.0,
    'epochs': 150,
    'batch_size': 64,
    'patience': 30,
    'learning_rate': 1e-3
},
```

**Step 2: Update VAEBuilder instantiation**

In `Layer4Builder.build_vae()`:

```python
cfg = self.config['vae']
self.vae_builder = VAEBuilder(
    self.layer2_path,
    self.output_dir,
    latent_dim=cfg['latent_dim'],
    hidden_dim=cfg.get('hidden_dim', 128),
    n_layers=cfg.get('n_layers', 2),
    beta=cfg['beta'],
    beta_warmup_epochs=cfg.get('beta_warmup_epochs', 30),
    free_bits=cfg.get('free_bits', 2.0),
    model_type=cfg.get('model_type', 'multiscale_conv1d'),
    base_channels=cfg.get('base_channels', 64),
    cycle_epochs=cfg.get('cycle_epochs', 40)
)
```

**Step 3: Commit**

```bash
git add processing/layer4_builder.py
git commit -m "feat(layer4): update config for multi-scale VAE with cyclical annealing"
```

---

## Task 7: Full Integration Test

**Files:**
- Create: `tests/test_layer4/test_integration_multiscale.py`

**Step 1: Write integration test**

Create `tests/test_layer4/test_integration_multiscale.py`:

```python
"""Integration test for Multi-Scale VAE in Layer 4 pipeline."""
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import h5py


class TestMultiScaleIntegration:
    """Test full pipeline with multi-scale VAE."""

    @pytest.fixture
    def dummy_layer2_data(self):
        """Create dummy Layer 2 data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hourly_tensors.h5"

            n_patients = 100
            seq_len = 745
            n_vitals = 7

            values = np.random.randn(n_patients, seq_len, n_vitals).astype(np.float32)
            masks = (np.random.rand(n_patients, seq_len, n_vitals) > 0.1).astype(np.float32)
            patient_ids = np.array([f"P{i:04d}" for i in range(n_patients)])
            vital_names = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP']

            with h5py.File(path, 'w') as f:
                f.create_dataset('values', data=values)
                f.create_dataset('masks', data=masks)
                f.create_dataset('patient_index', data=patient_ids.astype('S'))
                f.create_dataset('vital_index', data=np.array(vital_names, dtype='S'))

            yield path, tmpdir

    def test_multiscale_vae_produces_nonzero_kl(self, dummy_layer2_data):
        """Verify multi-scale VAE doesn't collapse."""
        from processing.layer4.vae_trainer import VAEBuilder

        layer2_path, tmpdir = dummy_layer2_data

        builder = VAEBuilder(
            layer2_path,
            Path(tmpdir) / "output",
            model_type='multiscale_conv1d',
            latent_dim=32,
            base_channels=32,  # Smaller for test
            beta=0.5,
            free_bits=2.0,
            beta_warmup_epochs=5,
            cycle_epochs=10
        )

        # Train briefly
        latents = builder.build(epochs=15, batch_size=16, patience=10)

        # Check latent statistics
        mu_std = np.std(latents['mu'])
        logvar_mean = np.mean(latents['logvar'])

        # These should indicate non-collapsed latent space
        assert mu_std > 0.01, f"mu std too low: {mu_std}"
        assert logvar_mean > -10, f"logvar mean too low: {logvar_mean}"
```

**Step 2: Run integration test**

Run: `pytest tests/test_layer4/test_integration_multiscale.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_layer4/test_integration_multiscale.py
git commit -m "test(layer4): add integration test for multi-scale VAE"
```

---

## Task 8: Run Full Layer 4 Pipeline

**Step 1: Clear cache and run**

```bash
find processing -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
python -m processing.layer4_builder
```

**Step 2: Verify success criteria**

Check output for:
- KL divergence > 0.01 (not 0.000)
- Latent std reported or check manually
- Training completes without early collapse

**Step 3: Validate latents**

```python
import h5py
import numpy as np

with h5py.File('outputs/layer4/vae_latents.h5', 'r') as f:
    mu = f['mu'][:]
    logvar = f['logvar'][:]

print(f"mu std: {mu.std():.4f}")
print(f"logvar mean: {logvar.mean():.4f}")
print(f"Success: mu_std > 0.1 = {mu.std() > 0.1}")
```

**Step 4: Commit final state**

```bash
git add -A
git commit -m "feat(layer4): complete multi-scale VAE integration"
```
