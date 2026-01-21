# Examples

This directory contains examples for using scVAE-Annotator.

## Available Examples

### 1. Basic Example (`basic_example.py`)

Complete end-to-end example with the PBMC 10k dataset:

```bash
python examples/basic_example.py
```

This example demonstrates:
- Loading the PBMC 10k dataset
- Configuring the optimized pipeline
- Running the complete annotation
- Evaluating results

### 2. CLI Examples (`cli_examples.sh`)

Command-line examples for various use cases:

```bash
bash examples/cli_examples.sh  # Shows available commands
```

## Using Your Own Data

### From H5AD File:

```python
from scvae_annotator import Config, run_annotation_pipeline

config = Config(
    data_path='path/to/your_data.h5ad',
    output_dir='my_results',
    target_genes=2000,
    n_neighbors=15
)

results = run_annotation_pipeline(config)
```

### From Scanpy AnnData:

```python
import scanpy as sc
from scvae_annotator import Config, run_annotation_pipeline

# Load your data
adata = sc.read_h5ad('your_data.h5ad')
# or: adata = sc.read_10x_mtx('filtered_feature_bc_matrix/')

config = Config(
    data_path=None,  # Not needed when passing adata directly
    output_dir='results'
)

results = run_annotation_pipeline(config, adata=adata)
```

## Advanced Examples

See [EXAMPLES.md](../EXAMPLES.md) in the main directory for:
- Hyperparameter optimization
- Batch correction with Harmony
- Custom classifiers
- Visualizations and analysis

## Datasets

Example datasets can be downloaded from:
- [10x Genomics public datasets](https://www.10xgenomics.com/resources/datasets) (e.g., PBMC 10k)
- [Single Cell Portal](https://singlecell.broadinstitute.org/)

### PBMC 3k Preparation:

```bash
python data/prepare_pbmc3k.py
```

This prepares the PBMC 3k dataset for validation.
