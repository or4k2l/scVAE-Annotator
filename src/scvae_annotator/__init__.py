"""
scVAE-Annotator: Advanced Single-Cell RNA-seq Annotation Pipeline

An optimized pipeline for automated cell type annotation using:
- Variational Autoencoder (VAE) with early stopping
- Leiden clustering with adaptive metrics
- Hyperparameter optimization with Optuna
- Calibrated confidence scoring

Note: Main implementation is located in scvae_annotator.py in the root directory.
This package serves as a structure for future modularization.

Usage:
    from scvae_annotator import create_optimized_config, run_annotation_pipeline
    
    config = create_optimized_config()
    results = run_annotation_pipeline(config)
"""

__version__ = "0.1.0"
__author__ = "scVAE-Annotator Team"

# Note: Main implementation is in scvae_annotator.py at root level
# This package structure is for future modularization

try:
    import sys
    from pathlib import Path
    
    # Add root directory to path to import main implementation
    root_dir = Path(__file__).parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    
    # Import main components from root-level implementation
    from scvae_annotator import (
        Config,
        create_optimized_config,
        run_annotation_pipeline,
        analyze_optimization_results,
        create_visualizations,
        load_and_prepare_data,
        enhanced_preprocessing,
        optimized_leiden_clustering,
        train_improved_vae,
        EnhancedAutoencoderAnnotator
    )
    
    __all__ = [
        "Config",
        "create_optimized_config",
        "run_annotation_pipeline",
        "analyze_optimization_results",
        "create_visualizations",
        "load_and_prepare_data",
        "enhanced_preprocessing",
        "optimized_leiden_clustering",
        "train_improved_vae",
        "EnhancedAutoencoderAnnotator",
        "__version__"
    ]
    
except ImportError:
    # Fallback: Try to import legacy modules (for backwards compatibility)
    try:
        from .annotator import Annotator
        from .model import VAEModel
        __all__ = ["Annotator", "VAEModel", "__version__"]
    except ImportError:
        # No modules available
        __all__ = ["__version__"]

