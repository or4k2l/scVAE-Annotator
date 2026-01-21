"""
Tests for VAE model.
"""

import pytest
import torch
import numpy as np
from scvae_annotator.model import VAEModel, Encoder, Decoder


def test_encoder_initialization():
    """Test Encoder initialization."""
    encoder = Encoder(n_genes=100, hidden_dims=[64, 32], n_latent=10)
    
    assert encoder is not None
    assert isinstance(encoder, torch.nn.Module)


def test_decoder_initialization():
    """Test Decoder initialization."""
    decoder = Decoder(n_latent=10, hidden_dims=[64, 32], n_genes=100)
    
    assert decoder is not None
    assert isinstance(decoder, torch.nn.Module)


def test_vae_model_initialization():
    """Test VAEModel initialization."""
    model = VAEModel(n_genes=100, n_latent=10, hidden_dims=[64, 32])
    
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.optimizer is not None


def test_vae_encode():
    """Test encoding functionality."""
    np.random.seed(42)
    model = VAEModel(n_genes=50, n_latent=10)
    
    # Create synthetic data
    X = np.random.randn(10, 50).astype(np.float32)
    
    # Encode
    latent = model.encode(X)
    
    assert latent.shape == (10, 10)


def test_vae_decode():
    """Test decoding functionality."""
    np.random.seed(42)
    model = VAEModel(n_genes=50, n_latent=10)
    
    # Create synthetic latent representation
    z = np.random.randn(10, 10).astype(np.float32)
    
    # Decode
    recon = model.decode(z)
    
    assert recon.shape == (10, 50)
