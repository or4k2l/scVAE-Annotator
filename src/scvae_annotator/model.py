"""
VAE model implementation for scVAE-Annotator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List
import numpy as np
from tqdm import tqdm


class Encoder(nn.Module):
    """Encoder network for VAE."""
    
    def __init__(self, n_genes: int, hidden_dims: List[int], n_latent: int):
        super().__init__()
        
        layers = []
        prev_dim = n_genes
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, n_latent)
        self.fc_logvar = nn.Linear(prev_dim, n_latent)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE."""
    
    def __init__(self, n_latent: int, hidden_dims: List[int], n_genes: int):
        super().__init__()
        
        layers = []
        prev_dim = n_latent
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, n_genes))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)


class VAEModel:
    """
    Variational Autoencoder model for scRNA-seq data.
    
    Parameters
    ----------
    n_genes : int
        Number of genes (input features)
    n_latent : int
        Dimension of latent space
    hidden_dims : list
        Dimensions of hidden layers
    learning_rate : float
        Learning rate for optimization
    """
    
    def __init__(
        self,
        n_genes: int,
        n_latent: int = 10,
        hidden_dims: List[int] = None,
        learning_rate: float = 1e-3
    ):
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder = Encoder(n_genes, hidden_dims, n_latent).to(self.device)
        self.decoder = Decoder(n_latent, hidden_dims, n_genes).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss function (reconstruction + KL divergence)."""
        # Reconstruction loss (using MSE for count data)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
        
    def fit(self, X, epochs: int = 100, batch_size: int = 128):
        """
        Train the VAE model.
        
        Parameters
        ----------
        X : array-like
            Input data (cells x genes)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.FloatTensor(np.asarray(X.todense()) if hasattr(X, 'todense') else X)
        else:
            X_tensor = X
            
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x = batch[0].to(self.device)
                
                # Forward pass
                mu, logvar = self.encoder(x)
                z = self.reparameterize(mu, logvar)
                recon_x = self.decoder(z)
                
                # Compute loss
                loss = self.loss_function(recon_x, x, mu, logvar)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader.dataset)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                
    def encode(self, X):
        """
        Encode data to latent representation.
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        array
            Latent representation
        """
        self.encoder.eval()
        
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.FloatTensor(np.asarray(X.todense()) if hasattr(X, 'todense') else X)
        else:
            X_tensor = X
            
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            mu, _ = self.encoder(X_tensor)
            
        return mu.cpu().numpy()
        
    def decode(self, z):
        """
        Decode latent representation to gene expression.
        
        Parameters
        ----------
        z : array-like
            Latent representation
            
        Returns
        -------
        array
            Reconstructed gene expression
        """
        self.decoder.eval()
        
        if not isinstance(z, torch.Tensor):
            z_tensor = torch.FloatTensor(z)
        else:
            z_tensor = z
            
        with torch.no_grad():
            z_tensor = z_tensor.to(self.device)
            recon = self.decoder(z_tensor)
            
        return recon.cpu().numpy()
