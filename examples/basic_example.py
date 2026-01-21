"""
Basic example of using scVAE-Annotator optimized pipeline

This example demonstrates how to:
1. Configure the optimized pipeline
2. Run the annotation pipeline
3. Analyze results
4. Create visualizations
"""

import sys
sys.path.insert(0, '..')

from scvae_annotator import create_optimized_config, run_annotation_pipeline, analyze_optimization_results

def main():
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
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print("\nMake sure you have:")
        print("   1. Installed all dependencies: pip install -r requirements.txt")
        print("   2. Internet connection (for data download)")
        print("   3. Sufficient disk space (~500MB)")

if __name__ == "__main__":
    main()

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
