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
