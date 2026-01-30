"""
Command-line interface for scVAE-Annotator.
"""

import argparse
import sys
import torch

from .config import create_optimized_config, Config
from .pipeline import run_annotation_pipeline, analyze_optimization_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='scVAE-Annotator: Advanced Single-Cell RNA-seq Annotation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default optimized configuration
  scvae-annotate --data data/10x-Multiome-Pbmc10k-RNA.h5 --annotations data/pbmc10k_annotations.csv
  
  # Run with custom output directory
  scvae-annotate --data mydata.h5 --output-dir my_results
  
  # Disable hyperparameter optimization (faster)
  scvae-annotate --data mydata.h5 --no-optimization
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to data file (H5 format). If not provided, will download default PBMC dataset.'
    )
    
    parser.add_argument(
        '--annotations', '-a',
        type=str,
        default=None,
        help='Path to ground truth annotations (CSV format, optional)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--no-optimization',
        action='store_true',
        help='Disable Optuna hyperparameter optimization (faster, but may reduce accuracy)'
    )
    
    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=50,
        help='Number of Optuna trials for hyperparameter search (default: 50)'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=32,
        help='VAE embedding dimension (default: 32)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum VAE training epochs (default: 100, will stop early if validation loss stops improving)'
    )
    
    parser.add_argument(
        '--n-top-genes',
        type=int,
        default=3000,
        help='Number of highly variable genes to use (default: 3000)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Create configuration
    config = create_optimized_config()
    
    # Override with command-line arguments
    config.output_dir = args.output_dir
    config.autoencoder_embedding_dim = args.embedding_dim
    config.autoencoder_epochs = args.epochs
    config.n_top_genes = args.n_top_genes
    config.random_state = args.random_state
    config.use_hyperparameter_optimization = not args.no_optimization
    config.optuna_trials = args.optuna_trials

    print(f"🧬 scVAE-Annotator v0.1.0")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Data: {args.data or 'Will download PBMC10k dataset'}")
    print(f"  Annotations: {args.annotations or 'None'}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Hyperparameter optimization: {config.use_hyperparameter_optimization}")
    print(f"  Optuna trials: {config.optuna_trials}")
    print(f"  VAE embedding dimension: {config.autoencoder_embedding_dim}")
    print(f"  Max VAE epochs: {config.autoencoder_epochs}")
    print(f"  N top genes: {config.n_top_genes}")
    print(f"  Random state: {config.random_state}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"{'='*60}\n")

    try:
        # Run the pipeline
        adata = run_annotation_pipeline(
            config=config,
            data_path=args.data,
            annotations_path=args.annotations
        )
        
        # Analyze results
        summary = analyze_optimization_results(config)
        
        print(f"\n{'='*60}")
        print(f"✅ Pipeline completed successfully!")
        print(f"{'='*60}")
        print(f"Results saved to: {config.output_dir}/")
        print(f"  - annotated_data.h5ad: Annotated AnnData object")
        print(f"  - umap_comparison.png: UMAP visualizations")
        print(f"  - confusion_matrix.png: Prediction accuracy matrix")
        print(f"  - classification_report.csv: Detailed metrics per cell type")
        print(f"  - optimization_summary.json: Complete summary of results")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
