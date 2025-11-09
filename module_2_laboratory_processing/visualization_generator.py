"""
Visualization generation for harmonization review.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from pathlib import Path
from typing import Optional


def generate_static_dendrogram(
    linkage_matrix: np.ndarray,
    test_names: list,
    output_path: Path,
    title: str = "Hierarchical Clustering Dendrogram"
):
    """
    Generate static dendrogram PNG.

    Args:
        linkage_matrix: Scipy linkage matrix
        test_names: List of test names for labels
        output_path: Path to save PNG
        title: Plot title
    """
    if linkage_matrix is None or len(test_names) == 0:
        print("  No clustering data to visualize (skipping dendrogram)")
        return

    print(f"  Generating static dendrogram...")

    # Create figure
    plt.figure(figsize=(20, 10))

    # Generate dendrogram
    dendro = dendrogram(
        linkage_matrix,
        labels=test_names,
        leaf_rotation=90,
        leaf_font_size=8
    )

    plt.title(title, fontsize=16)
    plt.xlabel('Test Name', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved static dendrogram to: {output_path}")
