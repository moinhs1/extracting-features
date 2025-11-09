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


def generate_harmonization_explorer(
    harmonization_map,
    output_path: Path
):
    """
    Generate interactive harmonization explorer dashboard.

    Args:
        harmonization_map: Harmonization map DataFrame
        output_path: Path to save HTML
    """
    print(f"  Generating harmonization explorer dashboard...")

    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplots: 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Coverage by Tier',
                'Groups Needing Review',
                'Patient Coverage Distribution',
                'Test Count per Group'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'histogram'}]
            ]
        )

        # Plot 1: Coverage by tier (pie chart)
        tier_counts = harmonization_map['tier'].value_counts().sort_index()
        fig.add_trace(
            go.Pie(
                labels=[f'Tier {t}' for t in tier_counts.index],
                values=tier_counts.values,
                name='Tier Coverage',
                hovertemplate='<b>%{label}</b><br>Groups: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot 2: Review status (bar chart)
        review_counts = harmonization_map['needs_review'].value_counts()
        fig.add_trace(
            go.Bar(
                x=['Approved', 'Needs Review'],
                y=[review_counts.get(False, 0), review_counts.get(True, 0)],
                name='Review Status',
                marker_color=['green', 'orange'],
                hovertemplate='<b>%{x}</b><br>Groups: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # Plot 3: Patient coverage distribution (histogram)
        fig.add_trace(
            go.Histogram(
                x=harmonization_map['patient_count'],
                name='Patient Coverage',
                nbinsx=20,
                hovertemplate='Patient Count: %{x}<br>Groups: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # Plot 4: Test count per group (histogram)
        test_counts = harmonization_map['matched_tests'].str.split('|').apply(len)
        fig.add_trace(
            go.Histogram(
                x=test_counts,
                name='Tests per Group',
                nbinsx=20,
                hovertemplate='Tests in Group: %{x}<br>Groups: %{y}<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Harmonization Map Explorer Dashboard",
            height=900,
            showlegend=False
        )

        # Save
        fig.write_html(
            output_path,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )

        print(f"  Saved harmonization explorer to: {output_path}")

    except Exception as e:
        print(f"  Warning: Could not generate explorer dashboard: {e}")
