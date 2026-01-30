"""
Visualization functions for scVAE-Annotator.
"""

from typing import List, Tuple, Any
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc

from .config import Config, logger


def create_visualizations(adata: ad.AnnData, config: Config) -> None:
    """Create comprehensive visualizations with reproducible UMAP."""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    sc.tl.umap(adata, random_state=config.random_state)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_configs = [
        ('cell_type_ground_truth', 'Ground Truth'),
        ('autoencoder_predictions', 'Predictions'),
        ('leiden_labels', 'Leiden Clusters'),
        ('autoencoder_confidence', 'Prediction Confidence')
    ]

    for i, (col, title) in enumerate(plot_configs):
        if col in adata.obs.columns:
            sc.pl.umap(adata, color=col, ax=axes[i//2, i%2], show=False, frameon=False)
            axes[i//2, i%2].set_title(title)

    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/umap_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    if 'autoencoder_confidence' in adata.obs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        adata.obs['autoencoder_confidence'].hist(bins=50, ax=ax1)
        ax1.axvline(config.confidence_threshold, color='red', linestyle='--',
                   label=f'Threshold: {config.confidence_threshold:.3f}')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Number of Cells')
        ax1.set_title('Confidence Score Distribution')
        ax1.legend()

        if 'cell_type_ground_truth' in adata.obs:
            conf_by_type = adata.obs.groupby('cell_type_ground_truth')['autoencoder_confidence'].mean()
            conf_by_type.plot(kind='bar', ax=ax2, rot=45)
            ax2.set_xlabel('Cell Type')
            ax2.set_ylabel('Average Confidence')
            ax2.set_title('Average Confidence by Cell Type')

        plt.tight_layout()
        plt.savefig(f"{config.output_dir}/confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
