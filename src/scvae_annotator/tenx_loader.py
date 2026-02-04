"""
10x Genomics data loader and utilities for scVAE-Annotator.

This module provides native support for 10x Genomics single-cell data formats
including Cell Ranger outputs (MTX, H5) with metadata preservation.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


def load_10x_data(
    path: str,
    genome: Optional[str] = None,
    var_names: str = 'gene_symbols',
    cache: bool = False,
    validate_metadata: bool = True
) -> ad.AnnData:
    """
    Load 10x Genomics data with automatic format detection.
    
    Supports:
    - Cell Ranger MTX output (filtered_feature_bc_matrix/)
    - Cell Ranger H5 output (.h5)
    - Pre-processed H5AD files (.h5ad)
    
    Parameters
    ----------
    path : str
        Path to 10x data directory or file
    genome : Optional[str]
        Genome name for legacy 10x H5 files (e.g., 'GRCh38')
    var_names : str
        Attribute to use as variable names ('gene_symbols' or 'gene_ids')
    cache : bool
        Whether to cache the result
    validate_metadata : bool
        Whether to validate 10x-specific metadata
        
    Returns
    -------
    ad.AnnData
        Loaded and validated AnnData object
        
    Examples
    --------
    >>> # Load from MTX directory
    >>> adata = load_10x_data('filtered_feature_bc_matrix/')
    
    >>> # Load from H5 file
    >>> adata = load_10x_data('filtered_feature_bc_matrix.h5')
    
    >>> # Load with specific genome
    >>> adata = load_10x_data('data.h5', genome='GRCh38')
    """
    path_obj = Path(path)
    
    # Detect format
    if path_obj.suffix == '.h5ad':
        logger.info(f"Loading H5AD format from {path}")
        adata = sc.read_h5ad(path)
        
    elif path_obj.suffix == '.h5':
        logger.info(f"Loading 10x H5 format from {path}")
        adata = sc.read_10x_h5(path, genome=genome)
        
    elif path_obj.is_dir() or 'matrix.mtx' in str(path):
        logger.info(f"Loading 10x MTX format from {path}")
        adata = sc.read_10x_mtx(path, var_names=var_names, cache=cache)
        
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    # Make gene names unique (critical for 10x data)
    adata.var_names_make_unique()
    
    # Validate 10x metadata if requested
    if validate_metadata:
        _validate_10x_metadata(adata)
    
    logger.info(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def _validate_10x_metadata(adata: ad.AnnData) -> None:
    """Validate and log 10x-specific metadata."""
    if 'gene_ids' in adata.var.columns:
        logger.info("✓ 10x gene_ids found (Ensembl IDs)")
    
    # Check for common 10x metadata
    expected_var_cols = ['gene_ids', 'feature_types', 'genome']
    present_cols = [col for col in expected_var_cols if col in adata.var.columns]
    
    if present_cols:
        logger.info(f"✓ 10x metadata columns: {', '.join(present_cols)}")


def detect_10x_chemistry(adata: ad.AnnData) -> Optional[str]:
    """
    Detect 10x Genomics chemistry version from metadata.
    
    Returns chemistry version if detectable: 'v2', 'v3', 'v3.1', etc.
    """
    # Check uns for chemistry info
    if 'chemistry_description' in adata.uns:
        desc = adata.uns['chemistry_description']
        if 'v3' in desc.lower():
            return 'v3'
        elif 'v2' in desc.lower():
            return 'v2'
    
    # Heuristic based on gene count:
    # - v3 chemistry typically detects >30,000 features (genes)
    # - v2 chemistry typically detects 20,000-30,000 features
    # Reference: 10x Genomics technical specifications
    if adata.n_vars > 30000:
        return 'v3 (inferred)'
    elif adata.n_vars > 20000:
        return 'v2 (inferred)'
    
    return None


def get_10x_metadata_summary(adata: ad.AnnData) -> Dict[str, Any]:
    """
    Extract summary of 10x-specific metadata.
    
    Returns
    -------
    dict
        Summary including chemistry, feature types, genome, etc.
    """
    summary = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'chemistry': detect_10x_chemistry(adata),
        'has_gene_ids': 'gene_ids' in adata.var.columns,
        'has_feature_types': 'feature_types' in adata.var.columns,
    }
    
    # Feature types (for multimodal data)
    if 'feature_types' in adata.var.columns:
        summary['feature_types'] = adata.var['feature_types'].value_counts().to_dict()
    
    # Genome
    if 'genome' in adata.var.columns:
        summary['genome'] = adata.var['genome'].iloc[0] if len(adata.var) > 0 else None
    
    return summary
