"""
VAE Model implementation for scVAE-Annotator.
"""

import logging
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .config import Config, logger


class EarlyStopping:
    """Early stopping handler for training."""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
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
                 hidden_dims: List[int] = None, dropout: float = 0.2):
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def improved_vae_loss(recon_x, x, mu, logvar, beta=0.001):
    """Improved VAE loss with beta-VAE regularization."""
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_improved_vae(adata: ad.AnnData, config: Config) -> ad.AnnData:
    """Train improved VAE with early stopping."""
    logger.info("Training VAE with early stopping...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedVAE(
        input_dim=adata.X.shape[1],
        embedding_dim=config.autoencoder_embedding_dim,
        hidden_dims=config.autoencoder_hidden_dims,
        dropout=config.autoencoder_dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.autoencoder_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    data_tensor = torch.tensor(
        adata.X.toarray() if issparse(adata.X) else adata.X,
        dtype=torch.float32
    )

    train_size = int(0.8 * len(data_tensor))
    train_data = data_tensor[:train_size]
    val_data = data_tensor[train_size:]

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config.autoencoder_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.autoencoder_batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=config.autoencoder_patience)
    model.train()
    loss_history = []

    for epoch in range(config.autoencoder_epochs):
        train_loss = 0
        model.train()
        for batch in train_dataloader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_data)
            loss = improved_vae_loss(recon_batch, batch_data, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = batch[0].to(device)
                recon_batch, mu, logvar = model(batch_data)
                loss = improved_vae_loss(recon_batch, batch_data, mu, logvar)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)

        loss_history.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{config.autoencoder_epochs}, Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}')

        if early_stopping(avg_val_loss):
            logger.info(f'Early stopping at epoch {epoch + 1}')
            break

    model.eval()
    with torch.no_grad():
        full_data = data_tensor.to(device)
        _, mu, _ = model(full_data)
        adata.obsm['X_autoencoder'] = mu.detach().cpu().numpy()

    pd.DataFrame(loss_history).to_csv(f"{config.output_dir}/vae_loss_history.csv", index=False)
    logger.info("VAE training completed")
    return adata
