"""
Visualization utilities for scVAE-Annotator.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from typing import Optional
import pandas as pd


def plot_umap(
    adata: "anndata.AnnData",
    annotations: Optional[pd.DataFrame] = None,
    save: Optional[str] = None
):
    """
    Create UMAP visualization.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with UMAP coordinates
    annotations : DataFrame, optional
        Cell type annotations
    save : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if annotations is not None and 'cell_type' in annotations.columns:
        # Color by cell type
        if 'cell_id' in annotations.columns:
            cell_type_map = annotations.set_index('cell_id')['cell_type']
            adata.obs['cell_type'] = adata.obs_names.map(cell_type_map)
        elif len(annotations) == adata.n_obs:
            adata.obs['cell_type'] = annotations['cell_type'].values
        sc.pl.umap(adata, color='cell_type', ax=ax, show=False)
    else:
        # Color by cluster
        sc.pl.umap(adata, color='leiden', ax=ax, show=False)
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_marker_genes(
    adata: "anndata.AnnData",
    marker_genes: list,
    groupby: str = 'leiden',
    save: Optional[str] = None
):
    """
    Plot marker gene expression.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    marker_genes : list
        List of marker genes to plot
    groupby : str, default='leiden'
        Column to group by
    save : str, optional
        Path to save figure
    """
    sc.pl.dotplot(adata, marker_genes, groupby=groupby, save=save)


def plot_training_history(history: dict, save: Optional[str] = None):
    """
    Plot training loss history.
    
    Parameters
    ----------
    history : dict
        Training history
    save : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history['loss'], label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()
