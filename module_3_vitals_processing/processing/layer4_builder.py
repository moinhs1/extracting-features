"""Layer 4 Builder - Orchestrates all Layer 4 components.

Coordinates:
1. FPCA - Functional Principal Component Analysis
2. VAE - LSTM Variational Autoencoder
3. DTW Clustering - Validation baseline
4. HDBSCAN Clustering - Primary phenotype identification

All components can be run independently or as a coordinated pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import json
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Layer4Builder:
    """Orchestrates all Layer 4 embedding and clustering components."""

    def __init__(
        self,
        layer2_tensors_path: Path,
        output_dir: Path,
        config: Optional[Dict] = None
    ):
        """Initialize Layer 4 builder.

        Args:
            layer2_tensors_path: Path to hourly_tensors.h5 from Layer 2
            output_dir: Output directory for Layer 4 outputs
            config: Optional configuration overrides
        """
        self.layer2_path = Path(layer2_tensors_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        self.config = {
            'fpca': {
                'n_components': 10
            },
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
            'dtw': {
                'vitals': ['HR', 'MAP'],
                'n_clusters': 5,
                'max_iter': 50
            },
            'hdbscan': {
                'min_cluster_size': 50,      # Larger for more robust clusters
                'min_samples': 15,           # More strict core point threshold
                'cluster_selection_method': 'leaf',  # Finer-grained clusters
                'use_fpca': True,
                'use_vae': True,
                'use_umap': True,
                'umap_n_components': 15,
                'umap_n_neighbors': 30,
                'umap_min_dist': 0.0
            }
        }

        # Override with provided config
        if config:
            for section, params in config.items():
                if section in self.config:
                    self.config[section].update(params)

        # Component instances
        self.fpca_builder = None
        self.vae_builder = None
        self.dtw_clusterer = None
        self.embedding_clusterer = None

        # Results
        self.results = {}

    def build_fpca(self) -> Dict:
        """Build FPCA features.

        Returns:
            Summary dict
        """
        from .layer4.fpca_builder import FPCABuilder

        logger.info("=== Building FPCA Features ===")

        self.fpca_builder = FPCABuilder(
            self.layer2_path,
            self.output_dir,
            n_components=self.config['fpca']['n_components']
        )

        scores = self.fpca_builder.build()
        paths = self.fpca_builder.save()
        summary = self.fpca_builder.get_summary()

        self.results['fpca'] = {
            'summary': summary,
            'paths': {k: str(v) for k, v in paths.items()}
        }

        return summary

    def build_vae(self) -> Dict:
        """Build VAE latents.

        Returns:
            Summary dict
        """
        from .layer4.vae_trainer import VAEBuilder

        logger.info("=== Building VAE Latents ===")

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

        latents = self.vae_builder.build(
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            patience=cfg['patience'],
            learning_rate=cfg['learning_rate']
        )

        paths = self.vae_builder.save()
        summary = self.vae_builder.get_summary()

        self.results['vae'] = {
            'summary': summary,
            'paths': {k: str(v) for k, v in paths.items()}
        }

        return summary

    def build_dtw_clusters(self) -> Dict:
        """Build DTW-based clusters (validation).

        Returns:
            Summary dict
        """
        from .layer4.clustering_dtw import DTWClusterer

        logger.info("=== Building DTW Clusters ===")

        cfg = self.config['dtw']
        self.dtw_clusterer = DTWClusterer(
            self.layer2_path,
            self.output_dir,
            vitals=cfg['vitals'],
            n_clusters=cfg['n_clusters'],
            max_iter=cfg['max_iter']
        )

        clusters = self.dtw_clusterer.build()
        paths = self.dtw_clusterer.save()
        summary = self.dtw_clusterer.get_summary()

        self.results['dtw'] = {
            'summary': summary,
            'paths': {k: str(v) for k, v in paths.items()}
        }

        return summary

    def build_embedding_clusters(self) -> Dict:
        """Build HDBSCAN clusters on embeddings.

        Requires FPCA and/or VAE to be built first.

        Returns:
            Summary dict
        """
        from .layer4.clustering_embedding import EmbeddingClusterer

        logger.info("=== Building Embedding Clusters ===")

        fpca_path = self.output_dir / 'fpca_scores.parquet'
        vae_path = self.output_dir / 'vae_latents.h5'

        cfg = self.config['hdbscan']

        if cfg['use_fpca'] and not fpca_path.exists():
            raise RuntimeError("FPCA scores not found. Run build_fpca() first.")

        if cfg['use_vae'] and not vae_path.exists():
            raise RuntimeError("VAE latents not found. Run build_vae() first.")

        self.embedding_clusterer = EmbeddingClusterer(
            fpca_path,
            vae_path,
            self.output_dir,
            min_cluster_size=cfg['min_cluster_size'],
            min_samples=cfg['min_samples'],
            cluster_selection_method=cfg.get('cluster_selection_method', 'eom'),
            use_umap=cfg.get('use_umap', True),
            umap_n_components=cfg.get('umap_n_components', 15),
            umap_n_neighbors=cfg.get('umap_n_neighbors', 30),
            umap_min_dist=cfg.get('umap_min_dist', 0.0)
        )

        clusters = self.embedding_clusterer.build(
            use_fpca=cfg['use_fpca'],
            use_vae=cfg['use_vae']
        )

        paths = self.embedding_clusterer.save()
        summary = self.embedding_clusterer.get_summary()

        self.results['hdbscan'] = {
            'summary': summary,
            'paths': {k: str(v) for k, v in paths.items()}
        }

        return summary

    def build_all(
        self,
        skip_vae: bool = False,
        skip_dtw: bool = False
    ) -> Dict:
        """Build all Layer 4 components.

        Args:
            skip_vae: Skip VAE training (use existing if available)
            skip_dtw: Skip DTW clustering

        Returns:
            Combined results dict
        """
        logger.info("Building all Layer 4 components...")

        # 1. FPCA - always required
        self.build_fpca()

        # 2. VAE
        vae_path = self.output_dir / 'vae_latents.h5'
        if skip_vae and vae_path.exists():
            logger.info("Skipping VAE training (existing latents found)")
        else:
            self.build_vae()

        # 3. DTW clustering (validation)
        if not skip_dtw:
            self.build_dtw_clusters()

        # 4. Embedding clustering (primary)
        self.build_embedding_clusters()

        # Save combined config and results
        self._save_metadata()

        return self.results

    def _save_metadata(self):
        """Save configuration and results metadata."""
        metadata = {
            'config': self.config,
            'results': self.results
        }

        metadata_path = self.output_dir / 'layer4_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Saved metadata to {metadata_path}")

    def get_summary(self) -> Dict:
        """Get combined summary of all components."""
        return {
            'output_dir': str(self.output_dir),
            'components': list(self.results.keys()),
            'fpca': self.results.get('fpca', {}).get('summary', {}),
            'vae': self.results.get('vae', {}).get('summary', {}),
            'dtw': self.results.get('dtw', {}).get('summary', {}),
            'hdbscan': self.results.get('hdbscan', {}).get('summary', {})
        }


def main():
    """Run Layer 4 builder as standalone script."""
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Build Layer 4 embeddings and clusters')
    parser.add_argument('--skip-vae', action='store_true', help='Skip VAE training')
    parser.add_argument('--skip-dtw', action='store_true', help='Skip DTW clustering')
    parser.add_argument('--fpca-only', action='store_true', help='Only build FPCA')
    parser.add_argument('--vae-only', action='store_true', help='Only build VAE')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    layer2_path = base_dir / 'outputs' / 'layer2' / 'hourly_tensors.h5'
    output_dir = base_dir / 'outputs' / 'layer4'

    if not layer2_path.exists():
        logger.error(f"Layer 2 tensors not found: {layer2_path}")
        sys.exit(1)

    builder = Layer4Builder(layer2_path, output_dir)

    if args.fpca_only:
        builder.build_fpca()
    elif args.vae_only:
        builder.build_vae()
    else:
        builder.build_all(
            skip_vae=args.skip_vae,
            skip_dtw=args.skip_dtw
        )

    summary = builder.get_summary()
    print("\n=== Layer 4 Summary ===")
    print(f"Output: {summary['output_dir']}")
    print(f"Components built: {summary['components']}")

    if summary.get('fpca'):
        print(f"\nFPCA:")
        print(f"  Features: {summary['fpca'].get('n_features', 'N/A')}")
        var = summary['fpca'].get('explained_variance', {})
        if var:
            avg_var = sum(v['total'] for v in var.values()) / len(var)
            print(f"  Avg explained variance: {avg_var:.1%}")

    if summary.get('vae'):
        print(f"\nVAE:")
        print(f"  Latent dim: {summary['vae'].get('latent_dim', 'N/A')}")
        print(f"  Best val loss: {summary['vae'].get('best_val_loss', 'N/A'):.4f}")

    if summary.get('hdbscan'):
        print(f"\nHDBSCAN:")
        print(f"  Clusters: {summary['hdbscan'].get('n_clusters', 'N/A')}")
        print(f"  Outliers: {summary['hdbscan'].get('outlier_pct', 'N/A'):.1f}%")


if __name__ == '__main__':
    main()
