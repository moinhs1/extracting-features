"""Test fixes for VAE Trainer code review issues."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from processing.layer4.vae_trainer import VAETrainer
from processing.layer4.vae_model import LSTMVAE, VAELoss
from processing.layer4.vae_conv1d import Conv1DVAE
from processing.layer4.vae_multiscale import MultiScaleConv1DVAE, MultiScaleVAELoss


class TestVAETrainerFixes:
    """Test that code review fixes are working correctly."""

    @pytest.fixture
    def dummy_data(self):
        """Create minimal dummy data for testing."""
        n_patients = 10
        seq_len = 100
        n_vitals = 7
        values = np.random.randn(n_patients, seq_len, n_vitals).astype(np.float32)
        masks = np.ones((n_patients, seq_len, n_vitals), dtype=np.float32)
        return values, masks

    def test_accepts_different_model_types(self, dummy_data):
        """Test that VAETrainer accepts different model types (Fix Issue 3)."""
        values, _ = dummy_data
        seq_len, n_vitals = values.shape[1], values.shape[2]

        # LSTM model
        lstm_model = LSTMVAE(input_dim=n_vitals, hidden_dim=32, latent_dim=8, seq_len=seq_len)
        trainer_lstm = VAETrainer(lstm_model, device='cpu')
        assert trainer_lstm.model is not None

        # Conv1D model
        conv_model = Conv1DVAE(input_dim=n_vitals, seq_len=seq_len, latent_dim=8)
        trainer_conv = VAETrainer(conv_model, device='cpu')
        assert trainer_conv.model is not None

        # MultiScale model
        multiscale_model = MultiScaleConv1DVAE(input_dim=n_vitals, seq_len=seq_len, latent_dim=8)
        trainer_multiscale = VAETrainer(multiscale_model, device='cpu')
        assert trainer_multiscale.model is not None

    def test_accepts_custom_loss_fn(self, dummy_data):
        """Test that VAETrainer accepts and uses custom loss_fn (Fix Issue 2)."""
        values, _ = dummy_data
        seq_len, n_vitals = values.shape[1], values.shape[2]

        model = LSTMVAE(input_dim=n_vitals, hidden_dim=32, latent_dim=8, seq_len=seq_len)
        custom_loss = VAELoss(beta=0.7, free_bits=1.5)

        trainer = VAETrainer(model, device='cpu', loss_fn=custom_loss)

        # Check that the custom loss is used
        assert trainer.loss_fn is custom_loss
        assert trainer.loss_fn.beta == 0.7
        assert trainer.loss_fn.free_bits == 1.5

    def test_uses_loss_fn_get_beta_if_available(self, dummy_data):
        """Test that trainer uses loss_fn.get_beta() when available (Fix Issue 1)."""
        values, masks = dummy_data
        seq_len, n_vitals = values.shape[1], values.shape[2]

        model = MultiScaleConv1DVAE(input_dim=n_vitals, seq_len=seq_len, latent_dim=8)

        # Create MultiScaleVAELoss with cyclical annealing
        loss_fn = MultiScaleVAELoss(
            beta=0.5,
            free_bits=1.0,
            warmup_epochs=10,
            cycle_epochs=20
        )

        trainer = VAETrainer(model, device='cpu', loss_fn=loss_fn)

        # Train for a few epochs and check beta values
        history = trainer.train(
            values, masks,
            epochs=5,
            batch_size=4,
            patience=10
        )

        # Check that beta values are recorded in history
        assert len(history['beta']) == 5

        # Check that beta increases during warmup (first 10 epochs)
        # For epochs 0-4, beta should be increasing from 0 toward 0.5
        expected_betas = [0.5 * (i / 10) for i in range(5)]
        for i, (actual, expected) in enumerate(zip(history['beta'], expected_betas)):
            assert abs(actual - expected) < 0.01, f"Epoch {i}: expected {expected}, got {actual}"

    def test_fallback_to_internal_get_beta(self, dummy_data):
        """Test that trainer falls back to internal _get_beta() for VAELoss."""
        values, masks = dummy_data
        seq_len, n_vitals = values.shape[1], values.shape[2]

        model = LSTMVAE(input_dim=n_vitals, hidden_dim=32, latent_dim=8, seq_len=seq_len)

        # VAELoss doesn't have get_beta(), so trainer should use _get_beta()
        trainer = VAETrainer(
            model,
            device='cpu',
            beta=0.5,
            beta_warmup_epochs=10
        )

        # Train for a few epochs
        history = trainer.train(
            values, masks,
            epochs=5,
            batch_size=4,
            patience=10
        )

        # Check that beta increases linearly during warmup
        expected_betas = [0.5 * (i / 10) for i in range(5)]
        for i, (actual, expected) in enumerate(zip(history['beta'], expected_betas)):
            assert abs(actual - expected) < 0.01, f"Epoch {i}: expected {expected}, got {actual}"

    def test_cyclical_annealing_cycles(self, dummy_data):
        """Test that cyclical annealing actually cycles."""
        values, masks = dummy_data
        seq_len, n_vitals = values.shape[1], values.shape[2]

        model = MultiScaleConv1DVAE(input_dim=n_vitals, seq_len=seq_len, latent_dim=8)

        # Create loss with short cycle for testing
        loss_fn = MultiScaleVAELoss(
            beta=0.5,
            free_bits=1.0,
            warmup_epochs=5,
            cycle_epochs=10  # Cycle every 10 epochs
        )

        trainer = VAETrainer(model, device='cpu', loss_fn=loss_fn)

        # Train for 25 epochs to see multiple cycles
        history = trainer.train(
            values, masks,
            epochs=25,
            batch_size=4,
            patience=30  # High patience to complete all epochs
        )

        betas = history['beta']

        # Check that beta resets at cycle boundaries (epochs 10, 20)
        # Epoch 0-4: warmup (0 -> 0.5)
        # Epoch 5-9: steady (0.5)
        # Epoch 10-14: warmup again (0 -> 0.5)
        # Epoch 15-19: steady (0.5)
        # Epoch 20-24: warmup again (0 -> 0.5)

        # Check cycle 1 warmup
        assert betas[0] < betas[4], "Beta should increase during first warmup"

        # Check cycle 2 starts from low value
        if len(betas) > 10:
            assert betas[10] < betas[9], "Beta should reset at cycle boundary (epoch 10)"

        # Check cycle 3 starts from low value
        if len(betas) > 20:
            assert betas[20] < betas[19], "Beta should reset at cycle boundary (epoch 20)"
