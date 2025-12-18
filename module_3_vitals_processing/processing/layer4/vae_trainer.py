"""VAE Trainer - Training loop and inference for LSTM-VAE.

Handles:
- Training with early stopping
- Learning rate scheduling
- Latent extraction for all patients
- Model checkpointing
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from .vae_model import LSTMVAE, VAELoss
    from .vae_conv1d import Conv1DVAE
    from .vae_multiscale import MultiScaleConv1DVAE, MultiScaleVAELoss
except ImportError:
    from vae_model import LSTMVAE, VAELoss
    from vae_conv1d import Conv1DVAE
    from vae_multiscale import MultiScaleConv1DVAE, MultiScaleVAELoss

logger = logging.getLogger(__name__)


class VAETrainer:
    """Trainer for LSTM-VAE model."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'auto',
        beta: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        beta_warmup_epochs: int = 0,
        free_bits: float = 0.0,
        loss_fn: Optional[torch.nn.Module] = None
    ):
        """Initialize trainer.

        Args:
            model: VAE model to train (LSTMVAE, MultiScaleConv1DVAE, etc.)
            device: 'cuda', 'cpu', or 'auto'
            beta: Target KL weight for β-VAE (reached after warmup)
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            beta_warmup_epochs: Epochs to linearly anneal β from 0 to target (prevents posterior collapse)
            free_bits: Minimum KL per dimension before penalty applies
            loss_fn: Optional loss function (if None, creates VAELoss with given params)
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model = model.to(device)
        self.target_beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.free_bits = free_bits

        # Use provided loss_fn or create default
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = VAELoss(beta=0.0 if beta_warmup_epochs > 0 else beta, free_bits=free_bits)

        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = None

        self.history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_loss': [], 'val_recon': [], 'val_kl': [],
            'beta': []
        }
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def _get_beta(self, epoch: int) -> float:
        """Get β value for current epoch (annealing schedule)."""
        if self.beta_warmup_epochs <= 0:
            return self.target_beta
        if epoch >= self.beta_warmup_epochs:
            return self.target_beta
        # Linear warmup from 0 to target_beta
        return self.target_beta * (epoch / self.beta_warmup_epochs)

    def _create_dataloaders(
        self,
        values: np.ndarray,
        masks: np.ndarray,
        batch_size: int,
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders.

        Args:
            values: (n_patients, seq_len, n_vitals) vital values
            masks: (n_patients, seq_len, n_vitals) observation masks
            batch_size: Batch size
            val_split: Fraction for validation

        Returns:
            Tuple of (train_loader, val_loader)
        """
        n_patients = values.shape[0]
        n_val = int(n_patients * val_split)

        # Shuffle indices
        indices = np.random.permutation(n_patients)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        # Create tensors
        def make_loader(idx, shuffle):
            v = torch.FloatTensor(values[idx])
            m = torch.FloatTensor(masks[idx])
            dataset = TensorDataset(v, m)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        train_loader = make_loader(train_idx, shuffle=True)
        val_loader = make_loader(val_idx, shuffle=False)

        logger.info(f"Train: {len(train_idx)} patients, Val: {len(val_idx)} patients")
        return train_loader, val_loader

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for values, masks in loader:
            values = values.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            recon, mu, logvar = self.model(values, masks)
            loss, recon_loss, kl_loss = self.loss_fn(recon, values, mu, logvar, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Tuple[float, float, float]:
        """Run validation."""
        self.model.eval()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for values, masks in loader:
            values = values.to(self.device)
            masks = masks.to(self.device)

            recon, mu, logvar = self.model(values, masks)
            loss, recon_loss, kl_loss = self.loss_fn(recon, values, mu, logvar, masks)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    def train(
        self,
        values: np.ndarray,
        masks: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        val_split: float = 0.2,
        patience: int = 15,
        min_delta: float = 1e-4
    ) -> Dict:
        """Train the VAE model.

        Args:
            values: (n_patients, seq_len, n_vitals)
            masks: (n_patients, seq_len, n_vitals)
            epochs: Maximum epochs
            batch_size: Batch size
            val_split: Validation split fraction
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping

        Returns:
            Training history dict
        """
        # Get initial beta for logging
        initial_beta = getattr(self.loss_fn, 'beta', None) or getattr(self.loss_fn, 'target_beta', self.target_beta)
        logger.info(f"Training VAE: {epochs} epochs, batch_size={batch_size}, β={initial_beta}")

        train_loader, val_loader = self._create_dataloaders(
            values, masks, batch_size, val_split
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        no_improve = 0

        for epoch in range(epochs):
            # Update beta for this epoch (annealing)
            # Use loss_fn's get_beta() if available, otherwise use local _get_beta()
            if hasattr(self.loss_fn, 'get_beta'):
                current_beta = self.loss_fn.get_beta(epoch)
            else:
                current_beta = self._get_beta(epoch)
                self.loss_fn.beta = current_beta

            # Train
            train_loss, train_recon, train_kl = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_recon'].append(train_recon)
            self.history['train_kl'].append(train_kl)
            self.history['beta'].append(current_beta)

            # Validate
            val_loss, val_recon, val_kl = self._validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_recon'].append(val_recon)
            self.history['val_kl'].append(val_kl)

            self.scheduler.step()

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                beta_str = f", β={current_beta:.3f}" if self.beta_warmup_epochs > 0 else ""
                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"train_loss={train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}){beta_str}, "
                    f"val_loss={val_loss:.4f}"
                )

            # Early stopping
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model (val_loss={self.best_val_loss:.4f})")

        return self.history

    @torch.no_grad()
    def extract_latents(
        self,
        values: np.ndarray,
        masks: np.ndarray,
        batch_size: int = 128
    ) -> Dict[str, np.ndarray]:
        """Extract latent representations for all patients.

        Args:
            values: (n_patients, seq_len, n_vitals)
            masks: (n_patients, seq_len, n_vitals)
            batch_size: Batch size for inference

        Returns:
            Dict with 'mu', 'logvar', 'recon_error' arrays
        """
        self.model.eval()

        values_t = torch.FloatTensor(values)
        masks_t = torch.FloatTensor(masks)

        dataset = TensorDataset(values_t, masks_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_mu = []
        all_logvar = []
        all_recon_error = []

        for v_batch, m_batch in loader:
            v_batch = v_batch.to(self.device)
            m_batch = m_batch.to(self.device)

            # Encode
            recon, mu, logvar = self.model(v_batch, m_batch)

            # Compute per-patient reconstruction error
            recon_error = ((recon - v_batch) ** 2 * m_batch).sum(dim=(1, 2))
            n_obs = m_batch.sum(dim=(1, 2))
            recon_error = recon_error / (n_obs + 1e-8)

            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_recon_error.append(recon_error.cpu().numpy())

        return {
            'mu': np.concatenate(all_mu, axis=0),
            'logvar': np.concatenate(all_logvar, axis=0),
            'recon_error': np.concatenate(all_recon_error, axis=0)
        }

    def save_model(self, path: Path):
        """Save model weights."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'input_dim': self.model.encoder.input_proj.in_features,
                'hidden_dim': self.model.encoder.lstm.hidden_size,
                'latent_dim': self.model.latent_dim,
                'seq_len': self.model.decoder.seq_len,
                'n_layers': self.model.encoder.lstm.num_layers
            }
        }, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load_model(cls, path: Path, device: str = 'auto') -> 'VAETrainer':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']

        model = LSTMVAE(**config)
        model.load_state_dict(checkpoint['model_state'])

        trainer = cls(model, device=device)
        return trainer


class VAEBuilder:
    """High-level builder for VAE training and latent extraction."""

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
        model_type: str = 'multiscale_conv1d',
        base_channels: int = 64,
        cycle_epochs: int = 40
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

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from Layer 2."""
        logger.info(f"Loading data from {self.layer2_path}")

        with h5py.File(self.layer2_path, 'r') as f:
            values = f['values'][:]
            masks = f['masks'][:]
            self.patient_ids = f['patient_index'][:].astype(str)

        # Normalize values to [0, 1] range per vital
        # Store normalization params for potential inverse transform
        self.vital_mins = values.min(axis=(0, 1))
        self.vital_maxs = values.max(axis=(0, 1))
        self.vital_range = self.vital_maxs - self.vital_mins
        self.vital_range[self.vital_range == 0] = 1  # Avoid division by zero

        values_norm = (values - self.vital_mins) / self.vital_range

        logger.info(f"Loaded {values.shape[0]} patients, normalized to [0,1]")
        return values_norm, masks

    def build(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        learning_rate: float = 1e-3
    ) -> Dict[str, np.ndarray]:
        """Train VAE and extract latents.

        Args:
            epochs: Maximum training epochs
            batch_size: Batch size
            patience: Early stopping patience
            learning_rate: Initial learning rate

        Returns:
            Dict with latent arrays
        """
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

        # Create trainer with β-annealing to prevent posterior collapse
        # Pass the loss_fn so trainer uses it instead of creating its own
        self.trainer = VAETrainer(
            model,
            device='auto',
            beta=self.beta,
            learning_rate=learning_rate,
            beta_warmup_epochs=self.beta_warmup_epochs,
            free_bits=self.free_bits,
            loss_fn=loss_fn
        )

        # Train
        history = self.trainer.train(
            values, masks,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )

        # Extract latents
        logger.info("Extracting latent representations...")
        self.latents = self.trainer.extract_latents(values, masks)

        return self.latents

    def save(self) -> Dict[str, Path]:
        """Save VAE outputs."""
        if self.latents is None:
            raise RuntimeError("Must call build() before save()")

        paths = {}

        # Save latents as HDF5
        latents_path = self.output_dir / 'vae_latents.h5'
        with h5py.File(latents_path, 'w') as f:
            f.create_dataset('mu', data=self.latents['mu'])
            f.create_dataset('logvar', data=self.latents['logvar'])
            f.create_dataset('recon_error', data=self.latents['recon_error'])
            f.create_dataset('patient_index', data=self.patient_ids.astype('S'))

            # Store normalization params
            f.attrs['vital_mins'] = self.vital_mins
            f.attrs['vital_maxs'] = self.vital_maxs
            f.attrs['latent_dim'] = self.latent_dim

        paths['latents'] = latents_path
        logger.info(f"Saved latents to {latents_path}")

        # Save model
        model_path = self.output_dir / 'vae_model.pt'
        self.trainer.save_model(model_path)
        paths['model'] = model_path

        # Save training history
        import json
        history_path = self.output_dir / 'vae_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.trainer.history, f)
        paths['history'] = history_path

        return paths

    def get_summary(self) -> Dict:
        """Get training summary."""
        if self.latents is None:
            return {'status': 'not built'}

        return {
            'n_patients': len(self.latents['mu']),
            'latent_dim': self.latent_dim,
            'final_val_loss': self.trainer.history['val_loss'][-1] if self.trainer else None,
            'best_val_loss': self.trainer.best_val_loss if self.trainer else None,
            'mean_recon_error': float(self.latents['recon_error'].mean()),
            'epochs_trained': len(self.trainer.history['train_loss']) if self.trainer else 0
        }


def main():
    """Run VAE builder as standalone script."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    base_dir = Path(__file__).parent.parent.parent
    layer2_path = base_dir / 'outputs' / 'layer2' / 'hourly_tensors.h5'
    output_dir = base_dir / 'outputs' / 'layer4'

    if not layer2_path.exists():
        logger.error(f"Layer 2 tensors not found: {layer2_path}")
        sys.exit(1)

    builder = VAEBuilder(
        layer2_path,
        output_dir,
        latent_dim=32,
        hidden_dim=128,
        n_layers=2,
        beta=1.0
    )

    latents = builder.build(
        epochs=100,
        batch_size=64,
        patience=15
    )

    paths = builder.save()

    summary = builder.get_summary()
    print("\n=== VAE Summary ===")
    print(f"Patients: {summary['n_patients']}")
    print(f"Latent dim: {summary['latent_dim']}")
    print(f"Epochs trained: {summary['epochs_trained']}")
    print(f"Best val loss: {summary['best_val_loss']:.4f}")
    print(f"Mean recon error: {summary['mean_recon_error']:.4f}")


if __name__ == '__main__':
    main()
