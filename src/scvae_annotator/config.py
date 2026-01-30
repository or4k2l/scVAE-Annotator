"""
Configuration dataclass and utilities for scVAE-Annotator.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class Config:
    """Configuration class for the annotation pipeline."""
    # Clustering parameters
    leiden_resolution_range: Tuple[float, float] = (0.005, 0.1)
    leiden_resolution_steps: int = 10
    leiden_k_neighbors: int = 30

    # Autoencoder parameters
    autoencoder_embedding_dim: int = 32
    autoencoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 64
    autoencoder_lr: float = 0.001
    autoencoder_dropout: float = 0.2
    autoencoder_patience: int = 7

    # Classification parameters
    confidence_threshold: float = 0.7
    adaptive_quantile: float = 0.05
    n_jobs: int = -1
    use_smote: bool = True
    cross_validation_folds: int = 5
    use_hyperparameter_optimization: bool = True
    optuna_trials: int = 50
    subsample_optuna_train: Optional[int] = 5000

    # Data preprocessing
    n_top_genes: int = 3000
    min_genes_per_cell: int = 200
    max_mt_percent: float = 15
    min_ground_truth_ratio: float = 0.8

    # Marker genes
    marker_genes: List[str] = field(default_factory=lambda: [
        'CD3E', 'CD7', 'CD2', 'LEF1', 'TCF7',
        'CD4', 'IL7R', 'CCR7',
        'CD8A', 'CD8B',
        'NKG7', 'GNLY', 'KLRF1',
        'CD19', 'MS4A1', 'CD79A', 'CD79B',
        'CD14', 'FCGR3A', 'MS4A7',
        'FCER1A', 'CPA3', 'IRF7',
        'PPBP', 'PF4',
        'CD34', 'KIT'
    ])

    # Output paths
    output_dir: str = './results'
    random_state: int = 42

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def create_optimized_config() -> Config:
    """Create an optimized configuration."""
    return Config(
        leiden_resolution_range=(0.01, 0.2),
        leiden_resolution_steps=15,
        leiden_k_neighbors=30,
        autoencoder_embedding_dim=32,
        autoencoder_hidden_dims=[512, 256, 128, 64],
        autoencoder_epochs=100,
        autoencoder_batch_size=128,
        autoencoder_lr=0.001,
        autoencoder_dropout=0.1,
        autoencoder_patience=7,
        confidence_threshold=0.7,
        adaptive_quantile=0.05,
        use_smote=True,
        cross_validation_folds=3,
        use_hyperparameter_optimization=True,
        optuna_trials=50,
        subsample_optuna_train=5000,
        n_top_genes=3000,
        min_genes_per_cell=200,
        max_mt_percent=20,
        min_ground_truth_ratio=0.7,
        random_state=42,
        n_jobs=-1
    )
