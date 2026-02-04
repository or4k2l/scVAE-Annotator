"""
Comprehensive tests for scientific VAE features.
"""

import pytest
import torch
import numpy as np
import anndata as ad
import tempfile
from pathlib import Path

from scvae_annotator.config import Config, create_scientific_config, create_optimized_config
from scvae_annotator.vae import (
    ImprovedVAE,
    improved_vae_loss,
    train_improved_vae
)


class TestScientificConfig:
    """Test suite for scientific configuration."""

    def test_config_default_likelihood(self) -> None:
        """Test that default config has MSE likelihood."""
        config = Config()
        assert config.likelihood_type == 'mse'

    def test_config_scientific_parameters(self) -> None:
        """Test scientific modeling parameters exist."""
        config = Config()
        
        assert hasattr(config, 'likelihood_type')
        assert hasattr(config, 'warmup_epochs')
        assert hasattr(config, 'kl_warmup_start')
        assert hasattr(config, 'kl_warmup_end')
        assert hasattr(config, 'logvar_clip_min')
        assert hasattr(config, 'logvar_clip_max')
        assert hasattr(config, 'recon_clip_min')
        assert hasattr(config, 'recon_clip_max')

    def test_config_likelihood_validation(self) -> None:
        """Test that invalid likelihood type raises error."""
        with pytest.raises(ValueError, match="likelihood_type must be"):
            Config(likelihood_type='invalid')

    def test_config_warmup_epochs_validation(self) -> None:
        """Test that negative warmup epochs raise error."""
        with pytest.raises(ValueError, match="warmup_epochs must be non-negative"):
            Config(warmup_epochs=-1)

    def test_config_poisson_likelihood(self) -> None:
        """Test config with Poisson likelihood."""
        config = Config(likelihood_type='poisson')
        assert config.likelihood_type == 'poisson'

    def test_create_scientific_config(self) -> None:
        """Test create_scientific_config function."""
        config = create_scientific_config()
        
        assert isinstance(config, Config)
        assert config.likelihood_type == 'poisson'
        assert config.warmup_epochs == 10
        assert config.autoencoder_lr == 5e-4
        assert config.output_dir == './results_scientific'

    def test_create_scientific_config_with_overrides(self) -> None:
        """Test create_scientific_config with custom parameters."""
        config = create_scientific_config(
            warmup_epochs=20,
            autoencoder_epochs=50
        )
        
        assert config.likelihood_type == 'poisson'
        assert config.warmup_epochs == 20
        assert config.autoencoder_epochs == 50

    def test_create_optimized_config_defaults(self) -> None:
        """Test that create_optimized_config includes new defaults."""
        config = create_optimized_config()
        
        assert config.likelihood_type == 'mse'
        assert config.warmup_epochs == 10


class TestScientificVAELoss:
    """Test suite for scientific VAE loss function."""

    def test_loss_mse_mode(self) -> None:
        """Test VAE loss with MSE likelihood."""
        batch_size, input_dim, latent_dim = 32, 50, 10
        recon_x = torch.randn(batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        
        config = Config(likelihood_type='mse')
        loss, metrics = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() > 0
        assert 'recon_loss' in metrics
        assert 'kl_loss' in metrics
        assert 'kl_weight' in metrics
        assert 'total_loss' in metrics

    def test_loss_poisson_mode(self) -> None:
        """Test VAE loss with Poisson likelihood."""
        batch_size, input_dim, latent_dim = 32, 50, 10
        # For Poisson, recon_x should be log-rates and x should be counts
        recon_x = torch.randn(batch_size, input_dim) * 2  # log-rates
        x = torch.abs(torch.randn(batch_size, input_dim)) * 10  # counts (non-negative)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        
        config = Config(likelihood_type='poisson')
        loss, metrics = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() > 0
        assert 'recon_loss' in metrics
        assert 'kl_loss' in metrics

    def test_kl_warmup_progression(self) -> None:
        """Test that KL weight increases during warmup."""
        batch_size, input_dim, latent_dim = 16, 30, 8
        recon_x = torch.randn(batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        
        config = Config(warmup_epochs=10, kl_warmup_start=0.0, kl_warmup_end=1.0)
        
        weights = []
        for epoch in [0, 5, 10, 15]:
            _, metrics = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=epoch)
            weights.append(metrics['kl_weight'])
        
        # Weight should increase during warmup
        assert weights[0] < weights[1] < weights[2]
        # Weight should be constant after warmup
        assert weights[2] == weights[3]
        # First weight should be near start, last should be near end
        assert abs(weights[0] - 0.0) < 0.1
        assert abs(weights[3] - 1.0) < 0.1

    def test_numerical_stability_clamping(self) -> None:
        """Test that extreme values are clamped for stability."""
        batch_size, input_dim, latent_dim = 16, 30, 8
        
        # Create extreme values
        recon_x = torch.ones(batch_size, input_dim) * 100  # Very large
        x = torch.ones(batch_size, input_dim)
        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.ones(batch_size, latent_dim) * 50  # Very large
        
        config = Config(
            likelihood_type='poisson',
            recon_clip_min=-10.0,
            recon_clip_max=15.0,
            logvar_clip_min=-10.0,
            logvar_clip_max=10.0
        )
        
        # Should not raise error due to clamping
        loss, metrics = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=0)
        
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_returns_correct_structure(self) -> None:
        """Test that loss returns tuple with correct structure."""
        batch_size, input_dim, latent_dim = 8, 20, 5
        recon_x = torch.randn(batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        
        config = Config()
        result = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=0)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        loss, metrics = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)


@pytest.fixture
def sample_adata_with_counts() -> ad.AnnData:
    """Create sample AnnData with counts layer."""
    np.random.seed(42)
    n_obs, n_vars = 100, 30
    
    # Create count data (non-negative integers)
    X = np.random.poisson(lam=5, size=(n_obs, n_vars)).astype(np.float32)
    
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    
    # Add raw counts layer
    adata.layers['counts'] = X.copy()
    
    return adata


class TestTrainImprovedVAEWithScientific:
    """Test suite for training VAE with scientific features."""

    def test_train_vae_mse_likelihood(self, sample_adata_with_counts: ad.AnnData) -> None:
        """Test VAE training with MSE likelihood."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                likelihood_type='mse',
                autoencoder_embedding_dim=8,
                autoencoder_hidden_dims=[32, 16],
                autoencoder_epochs=2,
                autoencoder_batch_size=16
            )
            
            adata_result = train_improved_vae(sample_adata_with_counts, config)
            
            assert 'X_autoencoder' in adata_result.obsm
            assert adata_result.obsm['X_autoencoder'].shape[0] == len(sample_adata_with_counts)
            assert adata_result.obsm['X_autoencoder'].shape[1] == config.autoencoder_embedding_dim

    def test_train_vae_poisson_likelihood(self, sample_adata_with_counts: ad.AnnData) -> None:
        """Test VAE training with Poisson likelihood."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                likelihood_type='poisson',
                autoencoder_embedding_dim=8,
                autoencoder_hidden_dims=[32, 16],
                autoencoder_epochs=2,
                autoencoder_batch_size=16
            )
            
            adata_result = train_improved_vae(sample_adata_with_counts, config)
            
            assert 'X_autoencoder' in adata_result.obsm
            assert adata_result.obsm['X_autoencoder'].shape[0] == len(sample_adata_with_counts)

    def test_train_vae_saves_history_with_likelihood(
        self, sample_adata_with_counts: ad.AnnData
    ) -> None:
        """Test that VAE training saves history with likelihood type in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for likelihood in ['mse', 'poisson']:
                config = Config(
                    output_dir=tmpdir,
                    likelihood_type=likelihood,
                    autoencoder_epochs=2,
                    autoencoder_batch_size=16
                )
                
                train_improved_vae(sample_adata_with_counts.copy(), config)
                
                history_file = Path(tmpdir) / f"vae_loss_history_{likelihood}.csv"
                assert history_file.exists()

    def test_train_vae_uses_counts_layer(self, sample_adata_with_counts: ad.AnnData) -> None:
        """Test that Poisson likelihood uses counts layer when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                likelihood_type='poisson',
                autoencoder_epochs=2,
                autoencoder_batch_size=16
            )
            
            # Ensure counts layer exists
            assert 'counts' in sample_adata_with_counts.layers
            
            adata_result = train_improved_vae(sample_adata_with_counts, config)
            
            assert 'X_autoencoder' in adata_result.obsm

    def test_train_vae_poisson_without_counts_layer(self) -> None:
        """Test that Poisson likelihood works without counts layer (with warning)."""
        np.random.seed(42)
        n_obs, n_vars = 50, 20
        X = np.random.poisson(lam=5, size=(n_obs, n_vars)).astype(np.float32)
        adata = ad.AnnData(X)
        # No counts layer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                likelihood_type='poisson',
                autoencoder_epochs=2,
                autoencoder_batch_size=16
            )
            
            adata_result = train_improved_vae(adata, config)
            
            assert 'X_autoencoder' in adata_result.obsm

    def test_train_vae_kl_warmup(self, sample_adata_with_counts: ad.AnnData) -> None:
        """Test VAE training with KL warmup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                warmup_epochs=5,
                kl_warmup_start=0.0,
                kl_warmup_end=1.0,
                autoencoder_epochs=10,
                autoencoder_batch_size=16
            )
            
            adata_result = train_improved_vae(sample_adata_with_counts, config)
            
            assert 'X_autoencoder' in adata_result.obsm
            
            # Check history file contains kl_weight
            import pandas as pd
            history_file = Path(tmpdir) / f"vae_loss_history_{config.likelihood_type}.csv"
            history = pd.read_csv(history_file)
            
            assert 'kl_weight' in history.columns
            # KL weight should increase in early epochs
            assert history['kl_weight'].iloc[0] < history['kl_weight'].iloc[-1]


class TestBackwardCompatibility:
    """Test suite for backward compatibility."""

    def test_old_loss_signature_still_works(self) -> None:
        """Test that calling loss with beta parameter still works (legacy support)."""
        batch_size, input_dim, latent_dim = 16, 30, 8
        recon_x = torch.randn(batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        
        config = Config()
        
        # Should accept beta parameter (though it's ignored with scientific config)
        loss, metrics = improved_vae_loss(
            recon_x, x, mu, logvar, config, epoch=0, beta=0.001
        )
        
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_default_config_unchanged(self) -> None:
        """Test that default Config still works as before."""
        config = Config()
        
        # Should have sensible defaults
        assert config.likelihood_type == 'mse'
        assert config.autoencoder_epochs == 50
        assert config.autoencoder_batch_size == 64


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_zero_warmup_epochs(self) -> None:
        """Test with zero warmup epochs."""
        batch_size, input_dim, latent_dim = 8, 20, 5
        recon_x = torch.randn(batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        
        config = Config(warmup_epochs=0)
        
        # Should use full KL weight immediately
        _, metrics = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=0)
        
        assert metrics['kl_weight'] == config.kl_warmup_end

    def test_custom_kl_warmup_range(self) -> None:
        """Test with custom KL warmup range."""
        batch_size, input_dim, latent_dim = 8, 20, 5
        recon_x = torch.randn(batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        
        config = Config(
            warmup_epochs=10,
            kl_warmup_start=0.1,
            kl_warmup_end=0.5
        )
        
        # Test at start
        _, metrics_start = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=0)
        assert abs(metrics_start['kl_weight'] - 0.1) < 0.01
        
        # Test at end
        _, metrics_end = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=10)
        assert abs(metrics_end['kl_weight'] - 0.5) < 0.01

    def test_single_cell_batch(self) -> None:
        """Test with batch size of 1."""
        recon_x = torch.randn(1, 50)
        x = torch.randn(1, 50)
        mu = torch.randn(1, 10)
        logvar = torch.randn(1, 10)
        
        config = Config()
        loss, metrics = improved_vae_loss(recon_x, x, mu, logvar, config, epoch=0)
        
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
