# Documentation

This directory contains the documentation for scVAE-Annotator.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/or4k2l/scVAE-Annotator.git
cd scVAE-Annotator

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
from scvae_annotator import create_optimized_config, run_annotation_pipeline

# Create optimized configuration
config = create_optimized_config()

# Run pipeline
results = run_annotation_pipeline(config)
```

## Documentation Structure

### Main Documentation

- [README.md](../README.md) - Main project overview and quick start
- [EXAMPLES.md](../EXAMPLES.md) - Detailed usage examples
- [ANALYSIS_REPORT.md](../ANALYSIS_REPORT.md) - Complete analysis report for PBMC 10k
- [VALIDATION_REPORT.md](../VALIDATION_REPORT.md) - Cross-dataset validation
- [TECHNICAL_APPENDIX.md](../TECHNICAL_APPENDIX.md) - Detailed technical metrics

### Additional Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contributing guidelines
- [LICENSE](../LICENSE) - License information

## Data

- [data/README.md](../data/README.md) - Data preparation guide
- [data/prepare_pbmc3k.py](../data/prepare_pbmc3k.py) - PBMC 3k preparation script

## Visualizations

- [figures/README.md](../figures/README.md) - Overview of all visualizations

## API Reference

### Main Components

#### Config

Configuration class for the pipeline:

```python
from scvae_annotator import Config

config = Config(
    data_path='data/pbmc10k.h5ad',
    output_dir='results',
    target_genes=2000,
    n_neighbors=30,
    leiden_resolution=0.4,
    latent_dim=32
)
```

#### create_optimized_config()

Creates optimized configuration based on hyperparameter optimization results:

```python
config = create_optimized_config()
```

#### run_annotation_pipeline()

Runs the complete annotation pipeline:

```python
results = run_annotation_pipeline(config)
```

### Metrics

The pipeline provides the following metrics:

- **Accuracy**: Overall annotation accuracy
- **Precision/Recall/F1**: Per-class metrics
- **Confidence Scores**: Calibrated confidence values
- **Clustering Metrics**: NMI, ARI, Silhouette Score

## Advanced Topics

### Custom Data

See [EXAMPLES.md](../EXAMPLES.md) for guides on using your own data.

### Hyperparameter Tuning

The pipeline uses Optuna for automatic hyperparameter tuning. 
See [TECHNICAL_APPENDIX.md](../TECHNICAL_APPENDIX.md) for details.

### Batch Correction

Harmony integration for batch effect correction:

```python
config = Config(
    use_harmony=True,
    harmony_theta=2.0
)
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'scvae_annotator'`

**Solution**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**Issue**: GPU errors

**Solution**: The pipeline works with CPU. PyTorch automatically detects available hardware.

**Issue**: Memory errors with large datasets

**Solution**: Reduce `target_genes` or use batch processing:
```python
config = Config(target_genes=1000)
```

## Community

- **Issues**: Report bugs or feature requests on GitHub
- **Discussions**: Questions and discussions in GitHub Discussions
- **Pull Requests**: Contributions are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Citations

If you use scVAE-Annotator in your research, please cite:

```bibtex
@software{scvae_annotator,
  title = {scVAE-Annotator: Automated Cell Type Annotation for scRNA-seq},
  author = {scVAE-Annotator Team},
  year = {2024},
  url = {https://github.com/or4k2l/scVAE-Annotator}
}
```
