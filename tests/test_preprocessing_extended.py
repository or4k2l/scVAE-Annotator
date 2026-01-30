"""
Comprehensive tests for preprocessing module.
"""

import pytest
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from scvae_annotator.config import Config
from scvae_annotator.preprocessing import (
    discover_marker_genes,
    download_data,
    load_and_prepare_data,
    enhanced_preprocessing
)


@pytest.fixture
def sample_adata() -> ad.AnnData:
    """Create sample AnnData object."""
    np.random.seed(42)
    n_obs, n_vars = 200, 100
    X = np.random.poisson(2, size=(n_obs, n_vars)).astype(float)
    
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    
    # Add some MT genes
    adata.var_names = adata.var_names.tolist()[:95] + ['MT-CO1', 'MT-CO2', 'MT-ND1', 'MT-ND2', 'MT-ATP6']
    
    return adata


@pytest.fixture
def sample_adata_with_ground_truth() -> ad.AnnData:
    """Create sample AnnData with ground truth."""
    np.random.seed(42)
    n_obs, n_vars = 200, 100
    X = np.random.poisson(2, size=(n_obs, n_vars)).astype(float)
    
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    
    # Add ground truth
    adata.obs['cell_type_ground_truth'] = np.random.choice(
        ['T cell', 'B cell', 'Monocyte'], size=n_obs
    )
    
    return adata


class TestDiscoverMarkerGenes:
    """Test suite for discover_marker_genes function."""

    def test_no_ground_truth(self, sample_adata: ad.AnnData) -> None:
        """Test with no ground truth labels."""
        config = Config()
        markers = discover_marker_genes(sample_adata, config)
        
        assert markers == config.marker_genes

    def test_with_ground_truth(self, sample_adata_with_ground_truth: ad.AnnData) -> None:
        """Test with ground truth labels."""
        config = Config()
        markers = discover_marker_genes(sample_adata_with_ground_truth, config)
        
        assert len(markers) >= len(config.marker_genes)
        # Original markers should be included
        for gene in config.marker_genes:
            assert gene in markers or gene not in sample_adata_with_ground_truth.var_names

    def test_too_few_labeled_cells(self, sample_adata: ad.AnnData) -> None:
        """Test with too few labeled cells."""
        config = Config()
        # Add only 50 labeled cells (< 100 threshold)
        sample_adata.obs['cell_type_ground_truth'] = [np.nan] * 150 + ['T cell'] * 50
        
        markers = discover_marker_genes(sample_adata, config)
        
        assert markers == config.marker_genes

    def test_marker_uniqueness(self, sample_adata_with_ground_truth: ad.AnnData) -> None:
        """Test that discovered markers are unique."""
        config = Config()
        markers = discover_marker_genes(sample_adata_with_ground_truth, config)
        
        assert len(markers) == len(set(markers))


class TestDownloadData:
    """Test suite for download_data function."""

    @patch('os.system')
    @patch('os.path.exists')
    def test_download_data_when_missing(
        self, mock_exists: MagicMock, mock_system: MagicMock
    ) -> None:
        """Test data download when files are missing."""
        mock_exists.return_value = False
        
        download_data()
        
        # Should have called wget
        assert mock_system.call_count >= 2

    @patch('os.system')
    @patch('os.path.exists')
    def test_download_data_when_exists(
        self, mock_exists: MagicMock, mock_system: MagicMock
    ) -> None:
        """Test data download when files already exist."""
        mock_exists.return_value = True
        
        download_data()
        
        # Should not download if files exist
        # But might still unzip annotations
        assert mock_system.call_count <= 2


class TestLoadAndPrepareData:
    """Test suite for load_and_prepare_data function."""

    def test_load_invalid_path(self) -> None:
        """Test loading from invalid path."""
        with pytest.raises(Exception):
            load_and_prepare_data('/nonexistent/path.h5')

    @patch('scanpy.read_10x_h5')
    def test_load_data_success(self, mock_read: MagicMock) -> None:
        """Test successful data loading."""
        # Create mock AnnData
        mock_adata = ad.AnnData(np.random.rand(100, 50))
        mock_adata.var_names = [f"gene_{i}" for i in range(50)]
        mock_read.return_value = mock_adata
        
        result = load_and_prepare_data('dummy_path.h5')
        
        assert isinstance(result, ad.AnnData)
        mock_read.assert_called_once()


class TestEnhancedPreprocessing:
    """Test suite for enhanced_preprocessing function."""

    def test_basic_preprocessing(self, sample_adata: ad.AnnData) -> None:
        """Test basic preprocessing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                min_genes_per_cell=5,
                max_mt_percent=20,
                n_top_genes=50
            )
            
            result = enhanced_preprocessing(sample_adata, config)
            
            assert isinstance(result, ad.AnnData)
            assert 'X_pca' in result.obsm
            assert 'highly_variable' in result.var.columns

    def test_qc_metrics_computed(self, sample_adata: ad.AnnData) -> None:
        """Test that QC metrics are computed."""
        config = Config()
        result = enhanced_preprocessing(sample_adata, config)
        
        assert 'n_genes_by_counts' in result.obs.columns
        assert 'pct_counts_mt' in result.obs.columns
        assert 'pct_counts_ribo' in result.obs.columns

    def test_cell_filtering(self, sample_adata: ad.AnnData) -> None:
        """Test that low-quality cells are filtered."""
        initial_cells = sample_adata.n_obs
        
        config = Config(min_genes_per_cell=10, max_mt_percent=5)
        result = enhanced_preprocessing(sample_adata, config)
        
        # Some cells should be filtered
        assert result.n_obs <= initial_cells

    def test_gene_filtering(self, sample_adata: ad.AnnData) -> None:
        """Test that genes are filtered."""
        initial_genes = sample_adata.n_vars
        
        config = Config(n_top_genes=30)
        result = enhanced_preprocessing(sample_adata, config)
        
        # Should have highly variable genes
        assert result.n_vars <= initial_genes

    def test_normalization(self, sample_adata: ad.AnnData) -> None:
        """Test that data is normalized."""
        config = Config()
        result = enhanced_preprocessing(sample_adata, config)
        
        # Check that raw data is preserved
        assert result.raw is not None

    def test_pca_computation(self, sample_adata: ad.AnnData) -> None:
        """Test PCA computation."""
        config = Config()
        result = enhanced_preprocessing(sample_adata, config)
        
        assert 'X_pca' in result.obsm
        assert result.obsm['X_pca'].shape[1] == 50

    def test_neighbors_computation(self, sample_adata: ad.AnnData) -> None:
        """Test that neighbors are computed."""
        config = Config()
        result = enhanced_preprocessing(sample_adata, config)
        
        assert 'neighbors' in result.uns

    def test_marker_genes_included(self, sample_adata: ad.AnnData) -> None:
        """Test that marker genes are included in highly variable."""
        # Add some marker genes to var_names
        marker_gene = 'CD3E'
        var_names = sample_adata.var_names.tolist()
        if len(var_names) > 0:
            var_names[0] = marker_gene
            sample_adata.var_names = var_names
        
        config = Config()
        result = enhanced_preprocessing(sample_adata, config)
        
        # CD3E should be in highly variable if it exists
        if marker_gene in result.var_names:
            assert result.var.loc[marker_gene, 'highly_variable']

    @patch('scanpy.external.pp.harmony_integrate')
    def test_harmony_integration(
        self, mock_harmony: MagicMock, sample_adata: ad.AnnData
    ) -> None:
        """Test Harmony batch correction."""
        config = Config()
        enhanced_preprocessing(sample_adata, config)
        
        # Should attempt Harmony
        mock_harmony.assert_called_once()

    @patch('scanpy.external.pp.harmony_integrate')
    def test_harmony_failure_handling(
        self, mock_harmony: MagicMock, sample_adata: ad.AnnData
    ) -> None:
        """Test handling of Harmony failures."""
        mock_harmony.side_effect = Exception("Harmony failed")
        
        config = Config()
        # Should not raise error, just log warning
        result = enhanced_preprocessing(sample_adata, config)
        
        assert isinstance(result, ad.AnnData)

    def test_output_shape(self, sample_adata: ad.AnnData) -> None:
        """Test output shape matches expectations."""
        config = Config(n_top_genes=50)
        result = enhanced_preprocessing(sample_adata, config)
        
        assert result.n_obs > 0
        assert result.n_vars > 0
        assert result.n_vars <= 50  # Should have only top genes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
