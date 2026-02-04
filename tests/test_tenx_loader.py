"""Tests for 10x Genomics loader functionality."""

import pytest
import numpy as np
import anndata as ad
import tempfile
from pathlib import Path

from scvae_annotator.tenx_loader import (
    load_10x_data,
    detect_10x_chemistry,
    get_10x_metadata_summary,
    _validate_10x_metadata
)


@pytest.fixture
def mock_10x_adata():
    """Create mock 10x-like AnnData."""
    n_obs, n_vars = 100, 1000
    X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars))
    
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs_names = [f"CELL_{i}" for i in range(n_obs)]
    adata.var_names = [f"GENE_{i}" for i in range(n_vars)]
    
    # Add 10x-specific metadata
    # Format matches Ensembl gene IDs: ENSG followed by 11 digits (zero-padded)
    adata.var['gene_ids'] = [f"ENSG{i:011d}" for i in range(n_vars)]
    adata.var['feature_types'] = ['Gene Expression'] * n_vars
    adata.var['genome'] = ['GRCh38'] * n_vars
    
    return adata


class TestLoad10xData:
    """Tests for load_10x_data function."""
    
    def test_load_h5ad(self, mock_10x_adata):
        """Test loading H5AD format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5ad"
            mock_10x_adata.write_h5ad(path)
            
            adata = load_10x_data(str(path))
            
            assert adata.n_obs == 100
            assert adata.n_vars == 1000
            assert 'gene_ids' in adata.var.columns
    
    def test_metadata_validation(self, mock_10x_adata):
        """Test 10x metadata validation."""
        _validate_10x_metadata(mock_10x_adata)
        # Should not raise
    
    def test_unsupported_format(self):
        """Test error on unsupported format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_10x_data("test.txt")


class TestDetect10xChemistry:
    """Tests for chemistry detection."""
    
    def test_detect_v3(self):
        """Test v3 chemistry detection."""
        # Simulate v3 with many genes
        n_obs, n_vars = 100, 35000  # v3 typically has >30k genes
        X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars))
        adata_v3 = ad.AnnData(X=X.astype(np.float32))
        
        chemistry = detect_10x_chemistry(adata_v3)
        assert chemistry is not None
        assert 'v3' in chemistry.lower()
    
    def test_detect_from_uns(self):
        """Test chemistry detection from uns."""
        adata = ad.AnnData(np.random.randn(10, 10))
        adata.uns['chemistry_description'] = 'Single Cell 3\' v3'
        
        chemistry = detect_10x_chemistry(adata)
        assert chemistry == 'v3'
    
    def test_detect_none(self):
        """Test chemistry detection returns None for small datasets."""
        adata = ad.AnnData(np.random.randn(10, 100))
        
        chemistry = detect_10x_chemistry(adata)
        assert chemistry is None


class TestGet10xMetadataSummary:
    """Tests for metadata summary extraction."""
    
    def test_summary_complete(self, mock_10x_adata):
        """Test complete metadata summary."""
        summary = get_10x_metadata_summary(mock_10x_adata)
        
        assert summary['n_cells'] == 100
        assert summary['n_genes'] == 1000
        assert summary['has_gene_ids'] is True
        assert summary['has_feature_types'] is True
        assert 'Gene Expression' in summary['feature_types']
    
    def test_summary_minimal(self):
        """Test summary with minimal metadata."""
        adata = ad.AnnData(np.random.randn(50, 200))
        
        summary = get_10x_metadata_summary(adata)
        
        assert summary['n_cells'] == 50
        assert summary['n_genes'] == 200
        assert summary['has_gene_ids'] is False
        assert summary['has_feature_types'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
