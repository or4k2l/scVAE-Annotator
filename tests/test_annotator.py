"""
Tests for scVAE-Annotator.
"""

import pytest
import numpy as np
import pandas as pd
from scvae_annotator import Annotator


def test_annotator_initialization():
    """Test Annotator initialization."""
    annotator = Annotator(n_latent=10, hidden_dims=[128, 64])
    
    assert annotator.n_latent == 10
    assert annotator.hidden_dims == [128, 64]
    assert annotator.model is None
    assert annotator.adata is None
    assert not annotator.is_trained


def test_load_data_from_dataframe():
    """Test loading data from pandas DataFrame."""
    # Create synthetic data
    np.random.seed(42)
    n_cells = 100
    n_genes = 50
    data = pd.DataFrame(
        np.random.poisson(5, size=(n_cells, n_genes)),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"cell_{i}" for i in range(n_cells)]
    )
    
    annotator = Annotator()
    annotator.load_data(data)
    
    assert annotator.adata is not None
    assert annotator.adata.n_obs == n_cells


def test_training_requires_data():
    """Test that training requires loaded data."""
    annotator = Annotator()
    
    with pytest.raises(ValueError, match="No data loaded"):
        annotator.train(epochs=1)


def test_annotation_requires_training():
    """Test that annotation requires trained model."""
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.poisson(5, size=(100, 50)),
        columns=[f"gene_{i}" for i in range(50)]
    )
    
    annotator = Annotator()
    annotator.load_data(data)
    
    with pytest.raises(ValueError, match="Model not trained"):
        annotator.annotate()
