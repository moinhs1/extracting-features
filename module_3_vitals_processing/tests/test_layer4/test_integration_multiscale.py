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
