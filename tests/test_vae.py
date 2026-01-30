"""
Comprehensive tests for VAE module.
"""

import pytest
import torch
import numpy as np
import anndata as ad
import tempfile
from pathlib import Path

from scvae_annotator.config import Config
from scvae_annotator.vae import (
    EarlyStopping,
    ImprovedVAE,
    improved_vae_loss,
    train_improved_vae
)


class TestEarlyStopping:
    """Test suite for EarlyStopping class."""

    def test_initialization(self) -> None:
        """Test EarlyStopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.counter == 0
        assert es.best_loss is None

    def test_first_call(self) -> None:
        """Test first call to EarlyStopping."""
        es = EarlyStopping()
        result = es(1.0)
        
        assert result is False
        assert es.best_loss == 1.0
        assert es.counter == 0

    def test_improving_loss(self) -> None:
        """Test EarlyStopping with improving loss."""
        es = EarlyStopping(patience=3, min_delta=0.001)
        
        assert es(1.0) is False
        assert es(0.9) is False
        assert es(0.8) is False
        assert es.counter == 0

    def test_non_improving_loss(self) -> None:
        """Test EarlyStopping with non-improving loss."""
        es = EarlyStopping(patience=3, min_delta=0.001)
        
        assert es(1.0) is False
        assert es(1.0) is False
        assert es(1.0) is False
        assert es(1.0) is True  # Should trigger after patience
        assert es.counter == 3

    def test_slight_improvement_not_counted(self) -> None:
        """Test that slight improvements below min_delta don't reset counter."""
        es = EarlyStopping(patience=2, min_delta=0.1)
        
        assert es(1.0) is False
        assert es(0.95) is False  # Improvement < min_delta
        assert es(0.94) is True  # Should trigger
        assert es.counter == 2


class TestImprovedVAE:
    """Test suite for ImprovedVAE model."""

    def test_initialization_default(self) -> None:
        """Test ImprovedVAE with default parameters."""
        model = ImprovedVAE(input_dim=100)
        
        assert model.mu is not None
        assert model.logvar is not None
        assert model.encoder is not None
        assert model.decoder is not None

    def test_initialization_custom(self) -> None:
        """Test ImprovedVAE with custom parameters."""
        model = ImprovedVAE(
            input_dim=100,
            embedding_dim=16,
            hidden_dims=[128, 64],
            dropout=0.3
        )
        
        assert model.mu is not None

    def test_forward_pass(self) -> None:
        """Test forward pass through VAE."""
        model = ImprovedVAE(input_dim=50, embedding_dim=10)
        x = torch.randn(32, 50)  # batch_size=32, input_dim=50
        
        recon_x, mu, logvar = model(x)
        
        assert recon_x.shape == (32, 50)
        assert mu.shape == (32, 10)
        assert logvar.shape == (32, 10)

    def test_reparameterize(self) -> None:
        """Test reparameterization trick."""
        model = ImprovedVAE(input_dim=50, embedding_dim=10)
        mu = torch.randn(32, 10)
        logvar = torch.randn(32, 10)
        
        z = model.reparameterize(mu, logvar)
        
        assert z.shape == (32, 10)

    def test_model_training_mode(self) -> None:
        """Test model can switch between train and eval modes."""
        model = ImprovedVAE(input_dim=50)
        
        model.train()
        assert model.training is True
        
        model.eval()
        assert model.training is False


class TestImprovedVAELoss:
    """Test suite for improved_vae_loss function."""

    def test_loss_computation(self) -> None:
        """Test VAE loss computation."""
        batch_size, input_dim = 32, 50
        recon_x = torch.randn(batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, 10)
        logvar = torch.randn(batch_size, 10)
        
        loss = improved_vae_loss(recon_x, x, mu, logvar)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss.item() > 0

    def test_loss_with_beta(self) -> None:
        """Test VAE loss with different beta values."""
        batch_size, input_dim = 32, 50
        recon_x = torch.randn(batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, 10)
        logvar = torch.randn(batch_size, 10)
        
        loss1 = improved_vae_loss(recon_x, x, mu, logvar, beta=0.001)
        loss2 = improved_vae_loss(recon_x, x, mu, logvar, beta=0.01)
        
        # Different beta should give different losses
        assert loss1 != loss2


@pytest.fixture
def sample_adata() -> ad.AnnData:
    """Create sample AnnData for VAE training."""
    np.random.seed(42)
    n_obs, n_vars = 100, 30
    X = np.random.rand(n_obs, n_vars).astype(np.float32)
    
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    
    return adata


class TestTrainImprovedVAE:
    """Test suite for train_improved_vae function."""

    def test_train_vae_basic(self, sample_adata: ad.AnnData) -> None:
        """Test basic VAE training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                autoencoder_embedding_dim=8,
                autoencoder_hidden_dims=[32, 16],
                autoencoder_epochs=2,
                autoencoder_batch_size=16
            )
            
            adata_result = train_improved_vae(sample_adata, config)
            
            assert 'X_autoencoder' in adata_result.obsm
            assert adata_result.obsm['X_autoencoder'].shape[0] == len(sample_adata)
            assert adata_result.obsm['X_autoencoder'].shape[1] == config.autoencoder_embedding_dim

    def test_train_vae_saves_history(self, sample_adata: ad.AnnData) -> None:
        """Test that VAE training saves loss history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                autoencoder_epochs=3,
                autoencoder_batch_size=16
            )
            
            train_improved_vae(sample_adata, config)
            
            history_file = Path(tmpdir) / "vae_loss_history.csv"
            assert history_file.exists()

    def test_train_vae_early_stopping(self, sample_adata: ad.AnnData) -> None:
        """Test VAE training with early stopping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                autoencoder_epochs=100,  # High number
                autoencoder_patience=2,  # Low patience
                autoencoder_batch_size=16
            )
            
            adata_result = train_improved_vae(sample_adata, config)
            
            # Should have embeddings despite early stopping
            assert 'X_autoencoder' in adata_result.obsm

    def test_train_vae_embedding_dimensions(self, sample_adata: ad.AnnData) -> None:
        """Test VAE with different embedding dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for embed_dim in [4, 8, 16]:
                config = Config(
                    output_dir=tmpdir,
                    autoencoder_embedding_dim=embed_dim,
                    autoencoder_epochs=2
                )
                
                adata_result = train_improved_vae(sample_adata.copy(), config)
                
                assert adata_result.obsm['X_autoencoder'].shape[1] == embed_dim

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_vae_cuda(self, sample_adata: ad.AnnData) -> None:
        """Test VAE training on CUDA if available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_dir=tmpdir,
                autoencoder_epochs=2
            )
            
            adata_result = train_improved_vae(sample_adata, config)
            
            assert 'X_autoencoder' in adata_result.obsm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
