"""
VAE Model implementation for scVAE-Annotator.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .config import Config, logger


class EarlyStopping:
    """Early stopping handler for training."""
    def __init__(self, patience: int = 7, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class ImprovedVAE(nn.Module):
    """Improved Variational Autoencoder with batch normalization and dropout."""
    def __init__(self, input_dim: int, embedding_dim: int = 32,
                 hidden_dims: Optional[List[int]] = None, dropout: float = 0.2) -> None:
        super(ImprovedVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(prev_dim, embedding_dim)
        self.logvar = nn.Linear(prev_dim, embedding_dim)

        decoder_layers = []
        prev_dim = embedding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def improved_vae_loss(
    recon_x: torch.Tensor, 
    x: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    config: Config,
    epoch: int = 0,
    beta: float = 0.001
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Improved VAE loss with scientific modeling support.
    
    Supports both MSE (for normalized data) and Poisson (for count data)
    likelihoods, with KL warm-up for improved training stability.
    
    References
    ----------
    Grønbech et al. (2020). scVAE: Variational auto-encoders for 
    single-cell gene expression data. Bioinformatics.
        
    Lopez et al. (2018). Deep generative modeling for single-cell 
    transcriptomics. Nature Methods.
    
    Parameters
    ----------
    recon_x : torch.Tensor
        Reconstructed data (logits if Poisson, values if MSE)
    x : torch.Tensor
        Original data
    mu : torch.Tensor
        Latent mean
    logvar : torch.Tensor
        Latent log-variance
    config : Config
        Configuration with likelihood_type and stability parameters
    epoch : int
        Current epoch (for KL warm-up)
    beta : float
        Legacy beta parameter (ignored when using scientific config)
        
    Returns
    -------
    loss : torch.Tensor
        Total loss
    metrics : Dict[str, float]
        Loss components for logging
    """
    # Reconstruction loss
    if config.likelihood_type == 'mse':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    else:  # Poisson
        # Clamp logits for numerical stability
        # exp(15) ≈ 3.2M counts - sufficient for scRNA-seq
        recon_x_clamped = torch.clamp(
            recon_x, 
            min=config.recon_clip_min, 
            max=config.recon_clip_max
        )
        recon_loss = F.poisson_nll_loss(
            recon_x_clamped, 
            x, 
            log_input=True,  # Input is log(rate)
            full=False,      # Don't include Stirling approximation
            reduction='sum'
        )
    
    # KL Divergence with stability clamps
    logvar_clamped = torch.clamp(
        logvar, 
        min=config.logvar_clip_min, 
        max=config.logvar_clip_max
    )
    kl_loss = -0.5 * torch.sum(
        1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
    )
    
    # KL Warm-up (linear annealing)
    # Helps prevent posterior collapse in early training
    if epoch < config.warmup_epochs:
        kl_weight = (
            config.kl_warmup_start + 
            (config.kl_warmup_end - config.kl_warmup_start) * 
            (epoch / config.warmup_epochs)
        )
    else:
        kl_weight = config.kl_warmup_end
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    # Metrics for logging
    metrics = {
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'kl_weight': kl_weight,
        'total_loss': total_loss.item()
    }
    
    return total_loss, metrics


def train_improved_vae(adata: ad.AnnData, config: Config) -> ad.AnnData:
    """Train improved VAE with scientific loss functions."""
    logger.info(f"Training VAE with {config.likelihood_type} likelihood")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model initialization
    embedding_dim = config.latent_dim if config.latent_dim is not None else config.autoencoder_embedding_dim
    model = ImprovedVAE(
        input_dim=adata.X.shape[1],
        embedding_dim=embedding_dim,
        hidden_dims=config.autoencoder_hidden_dims,
        dropout=config.autoencoder_dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.autoencoder_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Data preparation based on likelihood type
    if config.likelihood_type == 'poisson':
        # Use raw counts for Poisson
        if 'counts' in adata.layers:
            data = adata.layers['counts']
            logger.info("Using raw counts from 'counts' layer for Poisson likelihood")
        else:
            logger.warning(
                "No 'counts' layer found. Using X for Poisson likelihood. "
                "For best results, provide raw counts in adata.layers['counts']"
            )
            data = adata.X
    else:
        # Use log-normalized for MSE
        data = adata.X
    
    # Convert to tensor
    data_tensor = torch.tensor(
        data.toarray() if issparse(data) else data,
        dtype=torch.float32
    )
    
    # Train/val split
    train_size = int(0.8 * len(data_tensor))
    train_data = data_tensor[:train_size]
    val_data = data_tensor[train_size:]
    
    train_loader = DataLoader(
        TensorDataset(train_data), 
        batch_size=config.autoencoder_batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data), 
        batch_size=config.autoencoder_batch_size, 
        shuffle=False
    )
    
    # Training loop
    early_stopping = EarlyStopping(patience=config.autoencoder_patience)
    history = []
    
    total_epochs = config.vae_epochs if config.vae_epochs is not None else config.autoencoder_epochs
    
    for epoch in range(total_epochs):
        # Training phase
        model.train()
        train_metrics = {'recon_loss': 0, 'kl_loss': 0, 'total_loss': 0, 'kl_weight': 0}
        
        for batch in train_loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            
            recon_batch, mu_batch, logvar_batch = model(batch_data)
            loss, metrics = improved_vae_loss(
                recon_batch, batch_data, mu_batch, logvar_batch, config, epoch
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            for key in train_metrics:
                train_metrics[key] += metrics[key]
        
        # Validation phase
        model.eval()
        val_metrics = {'recon_loss': 0, 'kl_loss': 0, 'total_loss': 0, 'kl_weight': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                batch_data = batch[0].to(device)
                recon_batch, mu_batch, logvar_batch = model(batch_data)
                loss, metrics = improved_vae_loss(
                    recon_batch, batch_data, mu_batch, logvar_batch, config, epoch
                )
                
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
        
        # Average metrics
        for key in train_metrics:
            if key != 'kl_weight':
                train_metrics[key] /= len(train_loader)
                val_metrics[key] /= len(val_loader)
        
        # Logging (use final KL weight from last batch)
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{total_epochs}, "
                f"Train: {train_metrics['total_loss']:.4f}, "
                f"Val: {val_metrics['total_loss']:.4f}, "
                f"KL Weight: {metrics['kl_weight']:.3f}"
            )
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_metrics['total_loss'],
            'train_recon': train_metrics['recon_loss'],
            'val_recon': val_metrics['recon_loss'],
            'train_kl': train_metrics['kl_loss'],
            'val_kl': val_metrics['kl_loss'],
            'kl_weight': metrics['kl_weight']
        })
        
        # Scheduler & Early stopping
        scheduler.step(val_metrics['total_loss'])
        
        if early_stopping(val_metrics['total_loss']):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Extract embeddings
    model.eval()
    with torch.no_grad():
        full_data = data_tensor.to(device)
        _, mu, _ = model(full_data)
        adata.obsm['X_autoencoder'] = mu.detach().cpu().numpy()
    
    # Save history with likelihood type in filename
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        f"{config.output_dir}/vae_loss_history_{config.likelihood_type}.csv", 
        index=False
    )
    
    logger.info(f"VAE training completed ({config.likelihood_type} likelihood)")
    return adata
