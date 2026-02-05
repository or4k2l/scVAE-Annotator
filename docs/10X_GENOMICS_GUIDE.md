# 10x Genomics Integration Guide

scVAE-Annotator provides native support for 10x Genomics single-cell RNA-seq data with automatic format detection and metadata preservation.

## Supported Formats

✅ **Cell Ranger MTX Output**
```python
adata = load_10x_data('filtered_feature_bc_matrix/')
```

✅ **Cell Ranger H5 Output**
```python
adata = load_10x_data('filtered_feature_bc_matrix.h5')
```

✅ **Pre-processed H5AD**
```python
adata = load_10x_data('processed_data.h5ad')
```

## Quick Start

### Option 1: Using the 10x Loader (Recommended)

```python
from scvae_annotator import run_annotation_pipeline, create_optimized_config
from scvae_annotator.tenx_loader import load_10x_data

# Load 10x data
adata = load_10x_data('filtered_feature_bc_matrix/')

# Run annotation
config = create_optimized_config()
adata = run_annotation_pipeline(config, data_path=None, adata=adata)
```

### Option 2: Direct Pipeline

```python
from scvae_annotator import run_annotation_pipeline, create_optimized_config

config = create_optimized_config()

# Pipeline automatically detects 10x MTX format
adata = run_annotation_pipeline(
    config,
    data_path='filtered_feature_bc_matrix/'
)
```

## Google Colab Demo

Try it instantly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/or4k2l/scVAE-Annotator/blob/main/examples/colab_10x_demo1.ipynb)

## Feature Preservation

scVAE-Annotator preserves all 10x-specific metadata:

- ✅ **Ensembl Gene IDs** (`gene_ids` column)
- ✅ **Feature Types** (Gene Expression, Antibody Capture, etc.)
- ✅ **Genome Information**
- ✅ **Cell Barcodes**

### Example: Accessing 10x Metadata

```python
from scvae_annotator.tenx_loader import get_10x_metadata_summary

summary = get_10x_metadata_summary(adata)
print(summary)
# Output:
# {
#   'n_cells': 2698,
#   'n_genes': 32738,
#   'chemistry': 'v3 (inferred)',
#   'has_gene_ids': True,
#   'feature_types': {'Gene Expression': 32738}
# }
```

## Validation Results

Tested with PBMC 3k dataset from 10x Genomics:

| Metric | Value |
|--------|-------|
| **Cells processed** | 2,698 |
| **Genes** | 32,738 → 2,013 (after QC/HVG) |
| **Accuracy** | 96.13% |
| **Cohen's Kappa** | 0.9420 |
| **High-confidence predictions** | 93.0% |
| **Runtime (Colab)** | ~8 minutes |

## Advanced Usage

### Chemistry Detection

```python
from scvae_annotator.tenx_loader import detect_10x_chemistry

chemistry = detect_10x_chemistry(adata)
print(f"Detected chemistry: {chemistry}")
```

### Working with Multimodal Data

For datasets with multiple feature types (e.g., CITE-seq):

```python
# Filter for Gene Expression only
adata_rna = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()

# Run annotation on RNA
adata_rna = run_annotation_pipeline(config, adata=adata_rna)
```

## Troubleshooting

### Issue: "contains more than one genome"

**Solution**: Specify genome explicitly
```python
adata = load_10x_data('data.h5', genome='GRCh38')
```

### Issue: Gene names not unique

**Solution**: Already handled automatically
```python
# load_10x_data() calls this internally:
adata.var_names_make_unique()
```

## Best Practices

1. **Use load_10x_data()** for explicit 10x handling
2. **Validate metadata** after loading
3. **Check chemistry version** for version-specific QC
4. **Preserve original data** before filtering

## Examples

See `examples/` directory for more:
- `examples/colab_10x_demo1.ipynb` - Interactive Colab demo
- `examples/basic_example.py` - Basic annotation example

## Support

For issues specific to 10x data:
1. Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
2. Open an issue with dataset details
3. Join discussions on GitHub
