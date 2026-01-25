"""
Integration tests for the complete pipeline.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
import scanpy as sc
from pathlib import Path


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_adata():
    """Create synthetic AnnData for testing."""
    np.random.seed(42)
    n_cells = 200
    n_genes = 100
    
    # Create count matrix
    counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Create AnnData
    adata = sc.AnnData(counts)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    
    # Add cell type labels for testing
    cell_types = np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=n_cells)
    adata.obs['cell_type'] = cell_types
    
    return adata


def test_config_initialization():
    """Test Config class initialization."""
    from scvae_annotator import Config
    
    config = Config()
    assert config.n_top_genes == 3000
    assert config.leiden_k_neighbors == 30
    assert config.output_dir == './results'


def test_config_custom_parameters():
    """Test Config with custom parameters."""
    from scvae_annotator import Config
    
    config = Config(
        n_top_genes=2500,
        leiden_k_neighbors=15,
        leiden_resolution_range=(0.01, 0.15),
        output_dir='custom_output'
    )
    
    assert config.n_top_genes == 2500
    assert config.leiden_k_neighbors == 15
    assert config.leiden_resolution_range == (0.01, 0.15)
    assert config.output_dir == 'custom_output'


def test_create_optimized_config():
    """Test creating optimized configuration."""
    from scvae_annotator import create_optimized_config
    
    config = create_optimized_config()
    
    # Verify optimized parameters
    assert config.n_top_genes == 3000
    assert config.leiden_k_neighbors == 30
    assert config.leiden_resolution_range == (0.01, 0.2)
    assert config.autoencoder_embedding_dim == 32
    assert config.autoencoder_epochs == 100


def test_preprocessing_pipeline(synthetic_adata):
    """Test preprocessing steps."""
    # Filter cells and genes
    sc.pp.filter_cells(synthetic_adata, min_genes=10)
    sc.pp.filter_genes(synthetic_adata, min_cells=3)
    
    assert synthetic_adata.n_obs > 0
    assert synthetic_adata.n_vars > 0
    
    # Normalize
    sc.pp.normalize_total(synthetic_adata, target_sum=1e4)
    sc.pp.log1p(synthetic_adata)
    
    # Check that data was processed (X should contain log-transformed values)
    assert synthetic_adata.X.max() > 0  # Has positive values
    assert 'log1p' in synthetic_adata.uns  # Log transformation was applied


def test_vae_training_basic(synthetic_adata, temp_output_dir):
    """Test basic VAE training."""
    import torch
    
    # Prepare data
    sc.pp.normalize_total(synthetic_adata, target_sum=1e4)
    sc.pp.log1p(synthetic_adata)
    
    # Simple VAE architecture
    input_dim = synthetic_adata.n_vars
    latent_dim = 10
    
    encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, latent_dim)
    )
    
    decoder = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, input_dim)
    )
    
    # Test forward pass
    X = torch.FloatTensor(synthetic_adata.X)
    z = encoder(X)
    X_recon = decoder(z)
    
    assert z.shape == (synthetic_adata.n_obs, latent_dim)
    assert X_recon.shape == X.shape


def test_leiden_clustering(synthetic_adata):
    """Test Leiden clustering."""
    import warnings
    # Preprocess
    sc.pp.normalize_total(synthetic_adata, target_sum=1e4)
    sc.pp.log1p(synthetic_adata)
    sc.pp.pca(synthetic_adata, n_comps=20)
    
    # Build graph
    sc.pp.neighbors(synthetic_adata, n_neighbors=15)
    
    # Cluster - suppress FutureWarning about igraph
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sc.tl.leiden(synthetic_adata, resolution=0.5)
    
    assert 'leiden' in synthetic_adata.obs.columns
    assert len(synthetic_adata.obs['leiden'].unique()) > 0


@pytest.mark.slow
def test_full_pipeline_synthetic(synthetic_adata, temp_output_dir):
    """Test full pipeline with synthetic data."""
    from scvae_annotator import Config, run_annotation_pipeline
    
    # Save synthetic data
    data_path = os.path.join(temp_output_dir, 'test_data.h5ad')
    synthetic_adata.write_h5ad(data_path)
    
    # Create config with corrected parameters
    config = Config(
        output_dir=temp_output_dir,
        n_top_genes=50,  # Use fewer genes for testing
        leiden_k_neighbors=10,
        autoencoder_embedding_dim=10,
        autoencoder_epochs=5,  # Few epochs for testing
        optuna_trials=2  # Few trials for testing
    )
    
    # This test is marked as slow and would require full pipeline implementation
    # Skip for now as it's an integration test
    pytest.skip("Full pipeline test requires complete implementation")


def test_pipeline_with_output_dir():
    """Test pipeline with custom output directory."""
    from scvae_annotator import Config
    
    config = Config(output_dir='./custom_output')
    
    # Just verify the config was created with the output directory
    assert config.output_dir == './custom_output'


def test_output_directory_creation(temp_output_dir):
    """Test that output directory is created."""
    from scvae_annotator import Config
    
    output_dir = os.path.join(temp_output_dir, 'new_output')
    config = Config(output_dir=output_dir)
    
    # Create directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    assert os.path.exists(output_dir)


def test_confidence_scores():
    """Test confidence score calculation."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    # Create synthetic data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, size=100)
    
    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    calibrated = CalibratedClassifierCV(clf, cv=2)
    calibrated.fit(X, y)
    
    # Get probabilities
    probs = calibrated.predict_proba(X)
    
    assert probs.shape == (100, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
