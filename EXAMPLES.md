# scVAE-Annotator Examples

## Complete Pipeline (PBMC 10k)

Run the end-to-end Poisson vs MSE comparison and save the figure:

```bash
python examples/pbmc_10k_complete_pipeline.py
```

Outputs:
- Script: [examples/pbmc_10k_complete_pipeline.py](examples/pbmc_10k_complete_pipeline.py)
- Figure: [figures/poisson_vs_mse_comparison.png](figures/poisson_vs_mse_comparison.png)

## Quick Start

```python
from scvae_annotator import create_optimized_config, run_annotation_pipeline

# Use default configuration
config = create_optimized_config()

# Run pipeline
adata = run_annotation_pipeline(config)
```

## Custom Configuration

```python
from scvae_annotator import Config, run_annotation_pipeline

# Custom configuration
config = Config(
    # Output directory
    output_dir='./my_custom_results',
    
    # VAE parameters
    autoencoder_embedding_dim=64,  # Larger embedding
    autoencoder_hidden_dims=[1024, 512, 256, 128],  # Deeper network
    autoencoder_epochs=150,
    autoencoder_batch_size=256,
    autoencoder_patience=10,
    
    # Hyperparameter optimization
    use_hyperparameter_optimization=True,
    optuna_trials=100,  # More trials for better results
    subsample_optuna_train=10000,  # Larger subsample
    
    # Clustering
    leiden_resolution_range=(0.01, 0.3),
    leiden_resolution_steps=20,
    
    # Preprocessing
    n_top_genes=5000,  # More genes
    min_genes_per_cell=300,
    max_mt_percent=15,
    
    # Additional options
    random_state=123,
    n_jobs=-1  # Use all available CPUs
)

# Run pipeline with your own data
adata = run_annotation_pipeline(
    config,
    data_path='/path/to/your/data.h5',
    annotations_path='/path/to/annotations.csv'
)
```

## Run Specific Steps Only

```python
from scvae_annotator import (
    load_and_prepare_data,
    enhanced_preprocessing,
    optimized_leiden_clustering,
    train_improved_vae,
    EnhancedAutoencoderAnnotator,
    Config
)

config = Config()

# Load data
adata = load_and_prepare_data('path/to/data.h5')

# Preprocessing
adata = enhanced_preprocessing(adata, config)

# Clustering
adata, n_clusters = optimized_leiden_clustering(adata, config)

# Train VAE
adata = train_improved_vae(adata, config)

# Train annotator and predict
annotator = EnhancedAutoencoderAnnotator(config)
annotator.train(adata)
annotator.predict(adata)
```

## Inference on New Data Only

```python
import scanpy as sc
from scvae_annotator import EnhancedAutoencoderAnnotator, Config

# Load configuration and model
config = Config()
annotator = EnhancedAutoencoderAnnotator(config)

# Assumption: Model has been trained and saved
# annotator.load_model('path/to/saved/model')

# Load new data
new_adata = sc.read_h5ad('new_data.h5ad')

# Make predictions
annotator.predict(new_adata)

# Display results
print(new_adata.obs['autoencoder_predictions'].value_counts())
```

## Analyze Results

```python
from scvae_annotator import analyze_optimization_results, Config

config = Config(output_dir='./results')

# Analyze results
summary = analyze_optimization_results(config)

# Print specific metrics
print(f"Accuracy: {summary['accuracy']:.4f}")
print(f"Best Model: {summary['best_model_type']}")
print(f"VAE Early Stopped: {summary['vae_early_stopped']}")
```

## Create Visualizations

```python
from scvae_annotator import create_visualizations, Config
import scanpy as sc

# Load data
adata = sc.read_h5ad('results/annotated_data.h5ad')

# Configuration
config = Config()

# Create visualizations
create_visualizations(adata, config)
```
