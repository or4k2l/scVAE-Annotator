"""
Main Annotator class for scVAE-Annotator.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, Union
from pathlib import Path

from .model import VAEModel
from .preprocessing import preprocess_data
from .visualization import plot_umap, plot_marker_genes


class Annotator:
    """
    Main class for cell type annotation using VAE-based deep learning.
    
    Parameters
    ----------
    n_latent : int, default=10
        Number of latent dimensions in the VAE model
    hidden_dims : list, default=[128, 64]
        Hidden layer dimensions for encoder and decoder
    learning_rate : float, default=1e-3
        Learning rate for training
    """
    
    def __init__(
        self,
        n_latent: int = 10,
        hidden_dims: list = None,
        learning_rate: float = 1e-3
    ):
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        self.n_latent = n_latent
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        self.model = None
        self.adata = None
        self.is_trained = False
        
    def load_data(
        self,
        data: Union[str, Path, pd.DataFrame, "anndata.AnnData"],
        transpose: bool = False
    ):
        """
        Load single-cell RNA-seq data.
        
        Parameters
        ----------
        data : str, Path, DataFrame, or AnnData
            Input data containing gene expression counts
        transpose : bool, default=False
            Whether to transpose the data (genes as rows, cells as columns)
        """
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if data_path.suffix == '.h5ad':
                self.adata = sc.read_h5ad(data_path)
            elif data_path.suffix in ['.csv', '.tsv', '.txt']:
                df = pd.read_csv(data_path, sep='\t' if data_path.suffix == '.tsv' else ',', index_col=0)
                if transpose:
                    df = df.T
                self.adata = sc.AnnData(df)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        elif isinstance(data, pd.DataFrame):
            if transpose:
                data = data.T
            self.adata = sc.AnnData(data)
        else:
            self.adata = data
            
        # Preprocess the data
        self.adata = preprocess_data(self.adata)
        
    def train(self, epochs: int = 100, batch_size: int = 128):
        """
        Train the VAE model.
        
        Parameters
        ----------
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=128
            Batch size for training
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please call load_data() first.")
            
        n_genes = self.adata.n_vars
        
        # Initialize model
        self.model = VAEModel(
            n_genes=n_genes,
            n_latent=self.n_latent,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate
        )
        
        # Train model
        self.model.fit(
            self.adata.X,
            epochs=epochs,
            batch_size=batch_size
        )
        
        self.is_trained = True
        
    def annotate(self) -> pd.DataFrame:
        """
        Annotate cells with predicted cell types.
        
        Returns
        -------
        DataFrame
            Cell type annotations for each cell
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
            
        # Get latent representation
        latent = self.model.encode(self.adata.X)
        
        # Store in adata
        self.adata.obsm['X_vae'] = latent
        
        # Perform clustering
        sc.pp.neighbors(self.adata, use_rep='X_vae')
        sc.tl.leiden(self.adata)
        
        # TODO: Implement cell type prediction based on marker genes
        # For now, return clusters
        annotations = pd.DataFrame({
            'cell_id': self.adata.obs_names,
            'cluster': self.adata.obs['leiden'].values,
            'cell_type': 'Unknown'  # Placeholder
        })
        
        return annotations
        
    def plot_umap(self, annotations: Optional[pd.DataFrame] = None, save: Optional[str] = None):
        """
        Create UMAP visualization of cells.
        
        Parameters
        ----------
        annotations : DataFrame, optional
            Cell type annotations to color by
        save : str, optional
            Path to save the figure
        """
        if self.adata is None:
            raise ValueError("No data loaded.")
            
        # Compute UMAP if not already done
        if 'X_umap' not in self.adata.obsm:
            sc.tl.umap(self.adata)
            
        plot_umap(self.adata, annotations, save)
