"""
Basic example of using scVAE-Annotator optimized pipeline

This example demonstrates how to:
1. Configure the optimized pipeline
2. Run the annotation pipeline
3. Analyze results
4. Create visualizations

Prerequisites:
- Install the package: pip install -e .
- Set SCIPY_ARRAY_API=0 or SCIPY_ARRAY_API=1 if encountering scipy compatibility issues
  (depending on your environment, try both if one doesn't work)
"""

import os
import sys

# Try to import the package - if it fails, provide helpful error message
try:
    from scvae_annotator import (
        create_optimized_config,
        run_annotation_pipeline,
        analyze_optimization_results
    )
except ImportError as e:
    print("❌ Error: Could not import scvae_annotator")
    print("\nPlease install the package first:")
    print("  cd /workspaces/scVAE-Annotator")
    print("  pip install -e .")
    print("\nIf you see scipy array API errors, try:")
    print("  export SCIPY_ARRAY_API=0")
    print("  or:")
    print("  export SCIPY_ARRAY_API=1")
    print("  (depends on your scipy/numpy versions)")
    sys.exit(1)


def main():
    """Run the basic scVAE-Annotator example."""
    print("scVAE-Annotator Basic Example - Optimized Pipeline")
    print("=" * 60)
    
    # Create optimized configuration
    print("\n1. Creating optimized configuration...")
    config = create_optimized_config()
    print(f"   Hyperparameter optimization: {config.use_hyperparameter_optimization}")
    print(f"   Optuna trials: {config.optuna_trials}")
    print(f"   Early stopping patience: {config.autoencoder_patience}")
    
    # Run pipeline with default data (PBMC 10k)
    print("\n2. Running annotation pipeline...")
    print("   This will download PBMC 10k data if not present")
    print("   Pipeline includes: preprocessing, clustering, VAE, optimization")
    
    try:
        adata = run_annotation_pipeline(config)
        print(f"\n3. Pipeline completed successfully!")
        print(f"   Annotated {adata.shape[0]} cells")
        print(f"   Results saved to: {config.output_dir}")
        
        # Analyze results
        print("\n4. Analyzing optimization results...")
        summary = analyze_optimization_results(config)
        
        print("\n5. Summary of results:")
        if 'accuracy' in summary:
            print(f"   Accuracy: {summary['accuracy']:.4f}")
            print(f"   Kappa: {summary['kappa']:.4f}")
        if 'best_model_type' in summary:
            print(f"   Best model: {summary['best_model_type']}")
        if 'vae_early_stopped' in summary:
            print(f"   VAE early stopped: {summary['vae_early_stopped']}")
            
        print("\n✅ Example completed successfully!")
        print(f"\nCheck {config.output_dir}/ for:")
        print("   - annotated_data.h5ad")
        print("   - umap_comparison.png")
        print("   - confusion_matrix.png")
        print("   - confidence_analysis.png")
        
        print("\n" + "=" * 50)
        print("Next steps:")
        print("- Load your own scRNA-seq data")
        print("- Adjust model parameters for your dataset")
        print("- Train for more epochs for better results")
        print("- Explore marker genes for each cluster")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print("\nMake sure you have:")
        print("   1. Installed the package: pip install -e .")
        print("   2. Installed all dependencies: pip install -r requirements.txt")
        print("   3. Internet connection (for data download)")
        print("   4. Sufficient disk space (~500MB)")
        print("\nIf you see scipy errors, try both:")
        print("   export SCIPY_ARRAY_API=0")
        print("   export SCIPY_ARRAY_API=1")
        print("   (depends on your scipy/numpy versions)")
        sys.exit(1)


if __name__ == "__main__":
    main()
