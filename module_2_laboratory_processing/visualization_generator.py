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


def generate_interactive_dendrogram(
    linkage_matrix: np.ndarray,
    test_names: list,
    output_path: Path,
    title: str = "Interactive Hierarchical Clustering Dendrogram"
):
    """
    Generate interactive dendrogram HTML with plotly.

    Args:
        linkage_matrix: Scipy linkage matrix
        test_names: List of test names for labels
        output_path: Path to save HTML
        title: Plot title
    """
    if linkage_matrix is None or len(test_names) == 0:
        print("  No clustering data to visualize (skipping interactive dendrogram)")
        return

    print(f"  Generating interactive dendrogram...")

    try:
        import plotly.graph_objects as go
        import plotly.figure_factory as ff
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

        # Create dendrogram figure
        fig = ff.create_dendrogram(
            linkage_matrix,
            labels=test_names,
            orientation='bottom',
            linkagefun=lambda x: linkage_matrix
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Test Name',
            yaxis_title='Distance',
            height=800,
            width=1400,
            hovermode='closest'
        )

        # Rotate x-axis labels
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=10))

        # Save as HTML
        fig.write_html(
            output_path,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )

        print(f"  Saved interactive dendrogram to: {output_path}")

    except Exception as e:
        print(f"  Warning: Could not generate interactive dendrogram: {e}")
