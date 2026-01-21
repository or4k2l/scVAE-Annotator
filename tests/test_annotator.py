"""
Tests for scVAE-Annotator Pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from scvae_annotator import Config, create_optimized_config


def test_config_initialization():
    """Test Config initialization with defaults."""
    config = Config()
    
    assert config.target_genes == 2000
    assert config.n_neighbors == 30
    assert config.output_dir == 'results'
    assert config.min_cells == 3
    assert config.min_genes == 200


def test_config_custom_values():
    """Test Config initialization with custom values."""
    config = Config(
        target_genes=3000,
        n_neighbors=15,
        output_dir='custom_results'
    )
    
    assert config.target_genes == 3000
    assert config.n_neighbors == 15
    assert config.output_dir == 'custom_results'


def test_create_optimized_config():
    """Test creating optimized configuration."""
    config = create_optimized_config()
    
    # Check that optimized values are set
    assert config.target_genes == 2000
    assert config.n_neighbors == 30
    assert config.leiden_resolution == 0.4
    assert config.latent_dim == 32
    assert config.vae_epochs == 100
    assert config.early_stopping_patience == 10


def test_synthetic_data_preprocessing():
    """Test preprocessing on synthetic data."""
    # Create synthetic data
    np.random.seed(42)
    n_cells = 100
    n_genes = 50
    
    counts = np.random.poisson(5, size=(n_cells, n_genes))
    adata = sc.AnnData(counts)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    
    # Basic preprocessing
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    assert adata.n_obs > 0
    assert adata.n_vars > 0
    assert 'X_norm' not in adata.layers  # log1p operates on .X


def test_config_with_data_path():
    """Test Config with data path."""
    config = Config(data_path='test_data.h5ad')
    
    assert config.data_path == 'test_data.h5ad'

    np.random.seed(42)
    data = pd.DataFrame(
        np.random.poisson(5, size=(100, 50)),
        columns=[f"gene_{i}" for i in range(50)]
    )
    
    annotator = Annotator()
    annotator.load_data(data)
    
    with pytest.raises(ValueError, match="Model not trained"):
        annotator.annotate()
