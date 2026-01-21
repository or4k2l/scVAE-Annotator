"""
Command-line interface for scVAE-Annotator.
"""

import argparse
from pathlib import Path
from scvae_annotator import Annotator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='scVAE-Annotator: Automated cell type annotation for scRNA-seq data'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file path (CSV, TSV, or H5AD format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file path for annotations (CSV format)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to pre-trained model (optional)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--latent-dim', '-l',
        type=int,
        default=10,
        help='Latent dimension size (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=128,
        help='Batch size for training (default: 128)'
    )
    
    parser.add_argument(
        '--transpose', '-t',
        action='store_true',
        help='Transpose input data (genes as rows, cells as columns)'
    )
    
    args = parser.parse_args()
    
    # Initialize annotator
    print("Initializing scVAE-Annotator...")
    annotator = Annotator(n_latent=args.latent_dim)
    
    # Load data
    print(f"Loading data from {args.input}...")
    annotator.load_data(args.input, transpose=args.transpose)
    print(f"Loaded {annotator.adata.n_obs} cells and {annotator.adata.n_vars} genes")
    
    # Train or load model
    if args.model:
        print(f"Loading pre-trained model from {args.model}...")
        # TODO: Implement model loading
        raise NotImplementedError("Model loading not yet implemented")
    else:
        print(f"Training model for {args.epochs} epochs...")
        annotator.train(epochs=args.epochs, batch_size=args.batch_size)
    
    # Annotate cells
    print("Annotating cells...")
    annotations = annotator.annotate()
    
    # Save results
    print(f"Saving annotations to {args.output}...")
    annotations.to_csv(args.output, index=False)
    
    print("Done!")


if __name__ == '__main__':
    main()
