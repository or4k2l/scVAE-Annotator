"""
Comprehensive tests for clustering module.
"""

import pytest
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
import tempfile

from scvae_annotator.config import Config
from scvae_annotator.clustering import optimized_leiden_clustering


@pytest.fixture
def sample_adata() -> ad.AnnData:
    """Create a sample AnnData object for testing."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50
    
    # Create expression matrix
    X = np.random.rand(n_obs, n_vars)
    
    # Create AnnData
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    
    # Add PCA (required for clustering)
    sc.tl.pca(adata, n_comps=20)
    
    # Add neighbors (required for Leiden)
    sc.pp.neighbors(adata, n_neighbors=15)
    
    return adata


@pytest.fixture
def sample_adata_with_ground_truth() -> ad.AnnData:
    """Create a sample AnnData object with ground truth labels."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50
    
    X = np.random.rand(n_obs, n_vars)
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    
    # Add ground truth
    adata.obs['cell_type_ground_truth'] = np.random.choice(
        ['TypeA', 'TypeB', 'TypeC'], size=n_obs
    )
    
    sc.tl.pca(adata, n_comps=20)
    sc.pp.neighbors(adata, n_neighbors=15)
    
    return adata


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            output_dir=tmpdir,
            leiden_resolution_range=(0.1, 0.5),
            leiden_resolution_steps=3,
            random_state=42
        )
        return config


class TestOptimizedLeidenClustering:
    """Test suite for optimized_leiden_clustering function."""

    def test_basic_clustering(self, sample_adata: ad.AnnData, test_config: Config) -> None:
        """Test basic clustering without ground truth."""
        adata_result, n_clusters = optimized_leiden_clustering(sample_adata, test_config)
        
        assert 'leiden' in adata_result.obs.columns
        assert 'leiden_labels' in adata_result.obs.columns
        assert n_clusters > 0
        assert isinstance(n_clusters, int)

    def test_clustering_with_ground_truth(
        self, sample_adata_with_ground_truth: ad.AnnData, test_config: Config
    ) -> None:
        """Test clustering with ground truth labels."""
        adata_result, n_clusters = optimized_leiden_clustering(
            sample_adata_with_ground_truth, test_config
        )
        
        assert 'leiden' in adata_result.obs.columns
        assert 'leiden_labels' in adata_result.obs.columns
        assert n_clusters > 0

    def test_clustering_saves_metrics(
        self, sample_adata: ad.AnnData, test_config: Config
    ) -> None:
        """Test that clustering metrics are saved."""
        optimized_leiden_clustering(sample_adata, test_config)
        
        metrics_file = Path(test_config.output_dir) / "clustering_metrics.csv"
        assert metrics_file.exists()

    def test_clustering_resolution_range(
        self, sample_adata: ad.AnnData, test_config: Config
    ) -> None:
        """Test that clustering tries multiple resolutions."""
        _, n_clusters = optimized_leiden_clustering(sample_adata, test_config)
        
        # Should have tried multiple resolutions
        assert n_clusters >= 1

    def test_clustering_reproducibility(
        self, sample_adata: ad.AnnData, test_config: Config
    ) -> None:
        """Test that clustering is reproducible with same random state."""
        adata1 = sample_adata.copy()
        adata2 = sample_adata.copy()
        
        result1, n1 = optimized_leiden_clustering(adata1, test_config)
        result2, n2 = optimized_leiden_clustering(adata2, test_config)
        
        assert n1 == n2
        assert (result1.obs['leiden'] == result2.obs['leiden']).all()

    def test_clustering_output_structure(
        self, sample_adata: ad.AnnData, test_config: Config
    ) -> None:
        """Test the structure of clustering output."""
        adata_result, n_clusters = optimized_leiden_clustering(sample_adata, test_config)
        
        # Check leiden labels are strings
        assert adata_result.obs['leiden'].dtype == object
        assert adata_result.obs['leiden_labels'].dtype == object
        
        # Check number of unique clusters matches reported number
        unique_clusters = len(adata_result.obs['leiden'].unique())
        assert unique_clusters == n_clusters

    def test_clustering_with_high_coverage_ground_truth(
        self, sample_adata: ad.AnnData, test_config: Config
    ) -> None:
        """Test clustering with high ground truth coverage."""
        # Add ground truth with >80% coverage
        n_labeled = int(len(sample_adata) * 0.9)
        labels = ['TypeA'] * (n_labeled // 2) + ['TypeB'] * (n_labeled - n_labeled // 2)
        labels += [np.nan] * (len(sample_adata) - n_labeled)
        sample_adata.obs['cell_type_ground_truth'] = labels
        
        adata_result, n_clusters = optimized_leiden_clustering(sample_adata, test_config)
        
        assert n_clusters > 0

    def test_clustering_with_low_coverage_ground_truth(
        self, sample_adata: ad.AnnData, test_config: Config
    ) -> None:
        """Test clustering with low ground truth coverage."""
        # Add ground truth with <80% coverage
        n_labeled = int(len(sample_adata) * 0.5)
        labels = ['TypeA'] * n_labeled + [np.nan] * (len(sample_adata) - n_labeled)
        sample_adata.obs['cell_type_ground_truth'] = labels
        
        adata_result, n_clusters = optimized_leiden_clustering(sample_adata, test_config)
        
        assert n_clusters > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
