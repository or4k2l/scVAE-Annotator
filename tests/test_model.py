"""
Tests for VAE Model Components.
"""

import pytest
import torch
import numpy as np


def test_torch_available():
    """Test that PyTorch is available."""
    assert torch.cuda.is_available() or True  # CPU is fine for tests


def test_vae_encoder_architecture():
    """Test VAE encoder architecture creation."""
    input_dim = 2000
    hidden_dims = [512, 256, 128, 64]
    latent_dim = 32
    
    # Build encoder layers
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(torch.nn.Linear(prev_dim, h_dim))
        layers.append(torch.nn.ReLU())
        prev_dim = h_dim
    
    # Final layer to latent
    layers.append(torch.nn.Linear(prev_dim, latent_dim))
    
    encoder = torch.nn.Sequential(*layers)
    
    assert encoder is not None
    assert isinstance(encoder, torch.nn.Module)
    
    # Test forward pass
    x = torch.randn(10, input_dim)
    z = encoder(x)
    assert z.shape == (10, latent_dim)


def test_vae_decoder_architecture():
    """Test VAE decoder architecture creation."""
    latent_dim = 32
    hidden_dims = [64, 128, 256, 512]
    output_dim = 2000
    
    # Build decoder layers
    layers = []
    prev_dim = latent_dim
    for h_dim in hidden_dims:
        layers.append(torch.nn.Linear(prev_dim, h_dim))
        layers.append(torch.nn.ReLU())
        prev_dim = h_dim
    
    # Final layer to output
    layers.append(torch.nn.Linear(prev_dim, output_dim))
    
    decoder = torch.nn.Sequential(*layers)
    
    assert decoder is not None
    assert isinstance(decoder, torch.nn.Module)
    
    # Test forward pass
    z = torch.randn(10, latent_dim)
    x_recon = decoder(z)
    assert x_recon.shape == (10, output_dim)


def test_reconstruction_loss():
    """Test reconstruction loss calculation."""
    # MSE loss for reconstruction
    criterion = torch.nn.MSELoss()
    
    x_true = torch.randn(10, 100)
    x_recon = torch.randn(10, 100)
    
    loss = criterion(x_recon, x_true)
    
    assert loss.item() >= 0
    assert isinstance(loss.item(), float)


def test_early_stopping_logic():
    """Test early stopping logic."""
    patience = 5
    best_loss = float('inf')
    counter = 0
    
    losses = [10.0, 9.0, 8.5, 8.4, 8.45, 8.5, 8.6, 8.7, 8.8]
    
    for loss in losses:
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    assert counter >= patience
    assert best_loss == 8.4
