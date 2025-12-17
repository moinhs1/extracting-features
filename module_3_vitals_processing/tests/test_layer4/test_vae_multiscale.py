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
