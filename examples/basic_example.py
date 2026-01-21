"""
Basic example of using scVAE-Annotator

This example demonstrates how to:
1. Load single-cell RNA-seq data
2. Train a VAE model
3. Annotate cell types
4. Visualize results
"""

import numpy as np
import pandas as pd
from scvae_annotator import Annotator

def main():
    print("scVAE-Annotator Basic Example")
    print("=" * 50)
    
    # Generate synthetic data for demonstration
    # In practice, you would load your own data
    print("\n1. Generating synthetic data...")
    np.random.seed(42)
    n_cells = 500
    n_genes = 2000
    
    # Create expression matrix
    data = pd.DataFrame(
        np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"cell_{i}" for i in range(n_cells)]
    )
    print(f"   Generated {n_cells} cells × {n_genes} genes")
    
    # Initialize annotator
    print("\n2. Initializing annotator...")
    annotator = Annotator(
        n_latent=10,
        hidden_dims=[128, 64],
        learning_rate=1e-3
    )
    
    # Load data
    print("\n3. Loading and preprocessing data...")
    annotator.load_data(data)
    print(f"   After preprocessing: {annotator.adata.n_obs} cells × {annotator.adata.n_vars} genes")
    
    # Train model
    print("\n4. Training VAE model...")
    print("   This may take a few minutes...")
    annotator.train(epochs=10, batch_size=128)  # Using fewer epochs for demo
    print("   Training complete!")
    
    # Annotate cells
    print("\n5. Annotating cells...")
    annotations = annotator.annotate()
    print(f"   Identified {annotations['cluster'].nunique()} clusters")
    print("\nCluster distribution:")
    print(annotations['cluster'].value_counts().sort_index())
    
    # Save results
    print("\n6. Saving results...")
    annotations.to_csv('annotations.csv', index=False)
    print("   Saved to: annotations.csv")
    
    # Create visualization (if possible)
    try:
        print("\n7. Creating visualization...")
        annotator.plot_umap(save='umap_visualization.png')
        print("   Saved to: umap_visualization.png")
    except Exception as e:
        print(f"   Could not create visualization: {e}")
    
    print("\n" + "=" * 50)
    print("Example complete!")
    print("\nNext steps:")
    print("- Load your own scRNA-seq data")
    print("- Adjust model parameters for your dataset")
    print("- Train for more epochs for better results")
    print("- Explore marker genes for each cluster")

if __name__ == '__main__':
    main()
