"""
Data preprocessing utilities for scVAE-Annotator.
"""

import scanpy as sc
from typing import Optional


def preprocess_data(
    adata: "anndata.AnnData",
    min_genes: int = 200,
    min_cells: int = 3,
    target_sum: Optional[float] = 1e4,
    n_top_genes: int = 2000,
    log_transform: bool = True
) -> "anndata.AnnData":
    """
    Preprocess single-cell RNA-seq data.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    min_genes : int, default=200
        Minimum number of genes per cell
    min_cells : int, default=3
        Minimum number of cells per gene
    target_sum : float, optional
        Target sum for normalization
    n_top_genes : int, default=2000
        Number of highly variable genes to keep
    log_transform : bool, default=True
        Whether to apply log transformation
        
    Returns
    -------
    AnnData
        Preprocessed AnnData object
    """
    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Normalize
    if target_sum is not None:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Log transform
    if log_transform:
        sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    
    # Scale
    sc.pp.scale(adata, max_value=10)
    
    return adata
