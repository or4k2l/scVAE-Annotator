# scVAE-Annotator

A deep learning-based tool for automated cell type annotation in single-cell RNA-sequencing data using Variational Autoencoders.

## Overview

scVAE-Annotator leverages the power of Variational Autoencoders (VAEs) to provide robust and accurate cell type annotation for single-cell RNA-seq (scRNA-seq) datasets. The tool combines dimensionality reduction, clustering, and marker gene analysis to automatically identify and annotate cell types in complex biological samples.

## Features

- **Automated Cell Type Annotation**: Leverages VAE-based deep learning for accurate cell type identification
- **Scalable Processing**: Efficiently handles large-scale scRNA-seq datasets
- **Flexible Input**: Supports multiple input formats (CSV, TSV, H5AD, MTX)
- **Comprehensive Visualization**: Built-in plotting functions for quality control and results interpretation
- **Marker Gene Analysis**: Integrated tools for identifying and validating cell type markers
- **Pre-trained Models**: Includes pre-trained models for common tissue types

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/or4k2l/scVAE-Annotator.git
cd scVAE-Annotator
pip install -e .
```

### Install with pip (once published)

```bash
pip install scvae-annotator
```

## Quick Start

### Basic Usage

```python
from scvae_annotator import Annotator

# Load your single-cell data
annotator = Annotator()
annotator.load_data("path/to/your/counts.csv")

# Train the model
annotator.train(epochs=100)

# Annotate cells
annotations = annotator.annotate()

# Visualize results
annotator.plot_umap(annotations)
```

### Command Line Interface

```bash
# Train and annotate
scvae-annotate --input counts.csv --output annotations.csv

# Use pre-trained model
scvae-annotate --input counts.csv --model pretrained/pbmc.h5 --output annotations.csv
```

## Documentation

For detailed documentation, please visit [our documentation site](https://scvae-annotator.readthedocs.io/) (coming soon).

## Examples

Example notebooks and scripts can be found in the `examples/` directory:

- `basic_annotation.ipynb`: Introduction to basic cell type annotation
- `custom_training.ipynb`: How to train custom models on your data
- `marker_analysis.ipynb`: Identifying and validating marker genes

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use scVAE-Annotator in your research, please cite:

```
@software{scvae_annotator,
  title = {scVAE-Annotator: Automated Cell Type Annotation for Single-Cell RNA-seq},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/or4k2l/scVAE-Annotator}
}
```

## Support

For questions and support:
- Open an issue on [GitHub Issues](https://github.com/or4k2l/scVAE-Annotator/issues)
- Check our [FAQ](docs/FAQ.md) (coming soon)

## Acknowledgments

This project builds upon the foundational work of the scVAE project and the broader single-cell analysis community.