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
