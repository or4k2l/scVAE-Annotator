"""
Tests for Preprocessing Functions.
"""

import pytest
import numpy as np
import scanpy as sc
import pandas as pd


def test_filter_cells():
    """Test cell filtering."""
    # Create synthetic data
    np.random.seed(42)
    adata = sc.AnnData(np.random.poisson(2, size=(100, 50)))
    adata.var_names = [f"gene_{i}" for i in range(50)]
    
    # Filter
    initial_cells = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=5)
    
    assert adata.n_obs <= initial_cells


def test_filter_genes():
    """Test gene filtering."""
    # Create synthetic data with some zero genes
    np.random.seed(42)
    counts = np.random.poisson(2, size=(100, 50))
    counts[:, :10] = 0  # Make first 10 genes all zeros
    
    adata = sc.AnnData(counts)
    adata.var_names = [f"gene_{i}" for i in range(50)]
    
    # Filter
    sc.pp.filter_genes(adata, min_cells=1)
    
    # Should have removed the zero genes
    assert adata.n_vars < 50


def test_normalization():
    """Test normalization."""
    np.random.seed(42)
    adata = sc.AnnData(np.random.poisson(5, size=(100, 50)))
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Check sums
    sums = np.array(adata.X.sum(axis=1)).flatten()
    assert np.allclose(sums, 1e4, rtol=1e-2)


def test_log_transformation():
    """Test log transformation."""
    np.random.seed(42)
    adata = sc.AnnData(np.random.poisson(5, size=(100, 50)))
    
    # Normalize and log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Check that values are positive (after log)
    assert np.all(adata.X >= 0)


def test_highly_variable_genes():
    """Test HVG selection."""
    np.random.seed(42)
    adata = sc.AnnData(np.random.poisson(5, size=(100, 100)))
    adata.var_names = [f"gene_{i}" for i in range(100)]
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=50)
    
    assert 'highly_variable' in adata.var.columns
    assert adata.var['highly_variable'].sum() == 50


def test_scaling():
    """Test data scaling."""
    np.random.seed(42)
    adata = sc.AnnData(np.random.randn(100, 50) * 5 + 10)
    
    # Scale
    sc.pp.scale(adata, max_value=10)
    
    # Check that data is centered and scaled
    means = np.array(adata.X.mean(axis=0)).flatten()
    assert np.allclose(means, 0, atol=1e-10)


def test_pca():
    """Test PCA computation."""
    np.random.seed(42)
    adata = sc.AnnData(np.random.randn(100, 50))
    
    # Compute PCA
    sc.pp.pca(adata, n_comps=20)
    
    assert 'X_pca' in adata.obsm
    assert adata.obsm['X_pca'].shape == (100, 20)


def test_neighbors_graph():
    """Test neighbor graph construction."""
    np.random.seed(42)
    adata = sc.AnnData(np.random.randn(100, 50))
    sc.pp.pca(adata, n_comps=20)
    
    # Build neighbors
    sc.pp.neighbors(adata, n_neighbors=15)
    
    assert 'neighbors' in adata.uns
    assert 'connectivities' in adata.obsp
    assert 'distances' in adata.obsp


def test_umap():
    """Test UMAP computation."""
    np.random.seed(42)
    adata = sc.AnnData(np.random.randn(100, 50))
    sc.pp.pca(adata, n_comps=20)
    sc.pp.neighbors(adata, n_neighbors=15)
    
    # Compute UMAP
    sc.tl.umap(adata)
    
    assert 'X_umap' in adata.obsm
    assert adata.obsm['X_umap'].shape == (100, 2)


def test_preprocessing_chain():
    """Test full preprocessing chain."""
    np.random.seed(42)
    adata = sc.AnnData(np.random.poisson(5, size=(100, 100)))
    adata.var_names = [f"gene_{i}" for i in range(100)]
    adata.obs_names = [f"cell_{i}" for i in range(100)]
    
    # Full chain
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=50)
    
    # Subset to HVGs
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    sc.pp.scale(adata_hvg, max_value=10)
    sc.pp.pca(adata_hvg, n_comps=20)
    sc.pp.neighbors(adata_hvg, n_neighbors=15)
    sc.tl.umap(adata_hvg)
    
    # Verify all steps completed
    assert adata_hvg.n_obs > 0
    assert adata_hvg.n_vars == 50
    assert 'X_pca' in adata_hvg.obsm
    assert 'X_umap' in adata_hvg.obsm


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
