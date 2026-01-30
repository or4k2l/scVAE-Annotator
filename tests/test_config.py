"""
Comprehensive tests for config module.
"""

import tempfile
from pathlib import Path
import pytest

from scvae_annotator.config import Config, create_optimized_config, logger


class TestConfig:
    """Test suite for Config dataclass."""

    def test_config_default_initialization(self) -> None:
        """Test Config with default values."""
        config = Config()
        
        assert config.leiden_resolution_range == (0.005, 0.1)
        assert config.leiden_resolution_steps == 10
        assert config.leiden_k_neighbors == 30
        assert config.autoencoder_embedding_dim == 32
        assert config.autoencoder_epochs == 50
        assert config.confidence_threshold == 0.7
        assert config.n_top_genes == 3000
        assert config.random_state == 42
        assert config.n_jobs == -1

    def test_config_custom_initialization(self) -> None:
        """Test Config with custom values."""
        config = Config(
            leiden_resolution_range=(0.01, 0.2),
            autoencoder_embedding_dim=64,
            n_top_genes=5000,
            random_state=123
        )
        
        assert config.leiden_resolution_range == (0.01, 0.2)
        assert config.autoencoder_embedding_dim == 64
        assert config.n_top_genes == 5000
        assert config.random_state == 123

    def test_config_output_dir_creation(self) -> None:
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_results"
            config = Config(output_dir=str(output_path))
            
            assert output_path.exists()
            assert output_path.is_dir()

    def test_config_marker_genes(self) -> None:
        """Test marker genes list."""
        config = Config()
        
        assert isinstance(config.marker_genes, list)
        assert len(config.marker_genes) > 0
        assert 'CD3E' in config.marker_genes
        assert 'CD4' in config.marker_genes

    def test_config_autoencoder_params(self) -> None:
        """Test autoencoder parameters."""
        config = Config()
        
        assert isinstance(config.autoencoder_hidden_dims, list)
        assert len(config.autoencoder_hidden_dims) == 3
        assert config.autoencoder_batch_size == 64
        assert config.autoencoder_lr == 0.001
        assert config.autoencoder_dropout == 0.2
        assert config.autoencoder_patience == 7

    def test_config_preprocessing_params(self) -> None:
        """Test preprocessing parameters."""
        config = Config()
        
        assert config.min_genes_per_cell == 200
        assert config.max_mt_percent == 15
        assert config.min_ground_truth_ratio == 0.8

    def test_config_optuna_params(self) -> None:
        """Test Optuna parameters."""
        config = Config()
        
        assert config.use_hyperparameter_optimization is True
        assert config.optuna_trials == 50
        assert config.subsample_optuna_train == 5000

    def test_config_smote_param(self) -> None:
        """Test SMOTE parameter."""
        config = Config()
        assert config.use_smote is True
        
        config_no_smote = Config(use_smote=False)
        assert config_no_smote.use_smote is False


class TestCreateOptimizedConfig:
    """Test suite for create_optimized_config function."""

    def test_create_optimized_config_returns_config(self) -> None:
        """Test that create_optimized_config returns a Config instance."""
        config = create_optimized_config()
        
        assert isinstance(config, Config)

    def test_optimized_config_values(self) -> None:
        """Test optimized config has expected values."""
        config = create_optimized_config()
        
        assert config.leiden_resolution_range == (0.01, 0.2)
        assert config.leiden_resolution_steps == 15
        assert config.autoencoder_embedding_dim == 32
        assert config.autoencoder_hidden_dims == [512, 256, 128, 64]
        assert config.autoencoder_epochs == 100
        assert config.autoencoder_batch_size == 128
        assert config.autoencoder_dropout == 0.1
        assert config.use_hyperparameter_optimization is True
        assert config.optuna_trials == 50

    def test_optimized_config_preprocessing(self) -> None:
        """Test optimized config preprocessing parameters."""
        config = create_optimized_config()
        
        assert config.n_top_genes == 3000
        assert config.min_genes_per_cell == 200
        assert config.max_mt_percent == 20  # Different from default
        assert config.min_ground_truth_ratio == 0.7  # Different from default

    def test_optimized_config_creates_output_dir(self) -> None:
        """Test that optimized config creates output directory."""
        config = create_optimized_config()
        output_path = Path(config.output_dir)
        
        assert output_path.exists()


class TestLogger:
    """Test suite for logger configuration."""

    def test_logger_exists(self) -> None:
        """Test that logger is configured."""
        assert logger is not None
        assert logger.name == "scvae_annotator.config"

    def test_logger_level(self) -> None:
        """Test logger level."""
        assert logger.level == 20  # INFO level


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
