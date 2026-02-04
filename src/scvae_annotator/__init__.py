"""
scVAE-Annotator: Advanced Single-Cell RNA-seq Annotation Pipeline

An optimized pipeline for automated cell type annotation using:
- Variational Autoencoder (VAE) with early stopping
- Leiden clustering with adaptive metrics
- Hyperparameter optimization with Optuna
- Calibrated confidence scoring

Usage:
    from scvae_annotator import create_optimized_config, run_annotation_pipeline
    
    config = create_optimized_config()
    results = run_annotation_pipeline(config)
"""

__version__ = "0.1.0"
__author__ = "scVAE-Annotator Team"

# Import core components from refactored modules
from .config import Config, create_optimized_config, logger
from .preprocessing import (
    discover_marker_genes,
    download_data,
    load_and_prepare_data,
    enhanced_preprocessing
)
from .clustering import optimized_leiden_clustering
from .vae import (
    EarlyStopping,
    ImprovedVAE,
    improved_vae_loss,
    train_improved_vae
)
from .annotator import EnhancedAutoencoderAnnotator, Annotator

import builtins as _builtins
if not hasattr(_builtins, "Annotator"):
    _builtins.Annotator = Annotator
from .visualization import create_visualizations
from .pipeline import (
    evaluate_predictions,
    run_annotation_pipeline,
    analyze_optimization_results
)
from .tenx_loader import (
    load_10x_data,
    detect_10x_chemistry,
    get_10x_metadata_summary
)

__all__ = [
    # Core config
    "Config",
    "create_optimized_config",
    "logger",
    
    # Preprocessing
    "discover_marker_genes",
    "download_data",
    "load_and_prepare_data",
    "enhanced_preprocessing",
    
    # Clustering
    "optimized_leiden_clustering",
    
    # VAE Model
    "EarlyStopping",
    "ImprovedVAE",
    "improved_vae_loss",
    "train_improved_vae",
    
    # Annotator
    "EnhancedAutoencoderAnnotator",
    "Annotator",
    
    # Visualization
    "create_visualizations",
    
    # Pipeline
    "evaluate_predictions",
    "run_annotation_pipeline",
    "analyze_optimization_results",
    
    # 10x Genomics support
    "load_10x_data",
    "detect_10x_chemistry",
    "get_10x_metadata_summary",
]

