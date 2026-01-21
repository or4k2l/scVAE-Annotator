# scVAE-Annotator

**Advanced Single-Cell RNA-seq Annotation Pipeline with VAE and Automated Hyperparameter Optimization**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/tests.yml/badge.svg)](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/tests.yml)
[![Code Quality](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/lint.yml/badge.svg)](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/or4k2l/scVAE-Annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/or4k2l/scVAE-Annotator)

## 🎯 Overview

scVAE-Annotator is an optimized pipeline for automated cell type annotation in single-cell RNA-seq data. It combines:

- **Variational Autoencoder (VAE)** with early stopping
- **Leiden clustering** with adaptive metrics
- **Automated hyperparameter optimization** with Optuna
- **Calibrated confidence scores** for predictions
- **Adaptive marker gene discovery**

## ✨ Key Features

- ✅ **Adaptive marker gene discovery** based on ground truth data
- ✅ **Smart ARI weighting** based on ground truth coverage
- ✅ **VAE early stopping** with validation loss monitoring
- ✅ **Automatic model selection** (XGBoost, Logistic Regression, SVC)
- ✅ **Calibrated confidence scores** on hold-out set
- ✅ **Reproducible UMAP visualizations** with fixed random state
- ✅ **Comprehensive evaluation and visualization**

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/or4k2l/scVAE-Annotator.git
cd scVAE-Annotator
```

2. **Install the package in editable mode:**
```bash
pip install -e .
```

This will automatically install all dependencies from `requirements.txt`.

### Alternative: Install Dependencies Only

If you prefer to install dependencies without installing the package:

```bash
pip install -r requirements.txt
```

**Note:** If you skip the `pip install -e .` step, you'll need to manually adjust Python paths when running examples.

### Troubleshooting

**SciPy Array API Issues:**
If you encounter scipy array API compatibility warnings or errors, set this environment variable:
```bash
export SCIPY_ARRAY_API=0
```

Or add it to your shell profile (`~/.bashrc` or `~/.zshrc`):
```bash
echo 'export SCIPY_ARRAY_API=0' >> ~/.bashrc
source ~/.bashrc
```

📝 **For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

## 📖 Usage

### Quick Start

After installation, run the basic example:

```bash
# Set environment variable if needed
export SCIPY_ARRAY_API=0

# Run the example
python examples/basic_example.py
```

This will download the PBMC 10k dataset and run the complete annotation pipeline.

### Simple Example

```python
from scvae_annotator import create_optimized_config, run_annotation_pipeline

# Create configuration
config = create_optimized_config()

# Run pipeline
adata = run_annotation_pipeline(config)
```

### With Custom Data

```python
from scvae_annotator import Config, run_annotation_pipeline

# Custom configuration
config = Config(
    output_dir='./my_results',
    autoencoder_epochs=100,
    optuna_trials=50,
    use_hyperparameter_optimization=True
)

# Run pipeline with your own data
adata = run_annotation_pipeline(
    config,
    data_path='path/to/your/data.h5',
    annotations_path='path/to/annotations.csv'
)
```

### Run as Script

```bash
python scvae_annotator.py
```

## 🔧 Configuration

Key parameters can be customized through the `Config` class:

```python
config = Config(
    # Clustering
    leiden_resolution_range=(0.01, 0.2),
    leiden_resolution_steps=15,
    
    # VAE
    autoencoder_embedding_dim=32,
    autoencoder_hidden_dims=[512, 256, 128, 64],
    autoencoder_epochs=100,
    autoencoder_patience=7,  # Early stopping
    
    # Klassifizierung
    use_hyperparameter_optimization=True,
    optuna_trials=50,
    subsample_optuna_train=5000,
    confidence_threshold=0.7,
    
    # Preprocessing
    n_top_genes=3000,
    min_genes_per_cell=200,
    max_mt_percent=20
)
```

## 📊 Output

The pipeline generates the following outputs in `output_dir`:

- `annotated_data.h5ad` - Annotated AnnData file
- `umap_comparison.png` - UMAP visualization
- `confusion_matrix.png` - Confusion matrix
- `confidence_analysis.png` - Confidence score analysis
- `calibration_plot.png` - Calibration plot
- `classification_report.csv` - Detailed classification report
- `evaluation_metrics.json` - Evaluation metrics
- `optimization_summary.json` - Optimization summary
- `vae_loss_history.csv` - VAE training history
- `clustering_metrics.csv` - Clustering metrics

## 🔬 Methodology

### 1. Preprocessing
- Quality control (mitochondrial genes, ribosomal genes)
- Normalization and log transformation
- Highly variable genes selection
- Adaptive marker gene integration
- Batch correction with Harmony

### 2. Clustering
- Leiden algorithm with automatic resolution optimization
- Adaptive metric weighting (Silhouette + ARI)
- Ground truth coverage consideration

### 3. Feature Extraction
- Variational Autoencoder (VAE)
- Early stopping to prevent overfitting
- Validation loss monitoring

### 4. Classification
- Hyperparameter optimization with Optuna
- SMOTE for class balancing
- Model calibration on hold-out set
- Adaptive confidence thresholds

### 5. Evaluation
- Accuracy, Cohen's Kappa
- Confusion matrix
- Confidence calibration plot
- Per-class performance metrics

## 📈 Example Results

### PBMC 10k Dataset (Primary Analysis)

- **Accuracy**: **99.38%**
- **Cohen's Kappa**: **0.9925**
- **High-Confidence Predictions**: **98.8%** (10,292/10,412 cells)
- **VAE Training**: **13% faster** with early stopping
- **Cell Types Identified**: 16 distinct populations
- **Perfect Classifications**: HSPC, Plasma, pDC (F1=1.000)

📊 **[View Full Analysis Report](ANALYSIS_REPORT.md)** for detailed results and visualizations.

🎨 **[View Figures Gallery](figures/README.md)** for all visualization outputs and interpretations.

### PBMC 3k Dataset (Cross-Dataset Validation)

- **Accuracy**: **93.01%** (93.6% retention from PBMC 10k)
- **Cohen's Kappa**: **0.9120**
- **High-Confidence Predictions**: **98.1%** (2,646/2,700 cells)
- **VAE Training**: **40% faster** with early stopping
- **Cell Types Identified**: 10 distinct populations
- **Generalization**: Robust cross-dataset performance validated

🔬 **[View Validation Report](VALIDATION_REPORT.md)** for cross-dataset generalization analysis.

## 🤝 Contributing

Contributions are welcome! Please create a pull request or open an issue.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- [Scanpy](https://scanpy.readthedocs.io/) for single-cell analysis
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [PyTorch](https://pytorch.org/) for deep learning
- 10x Genomics for example data

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{scvae_annotator,
  title = {scVAE-Annotator: Advanced Single-Cell RNA-seq Annotation Pipeline},
  author = Yahya Akbay,
  year = {2025},
  url = {https://github.com/or4k2l/scVAE-Annotator}
}
```

This project builds upon the foundational work of the scVAE project and the broader single-cell analysis community.
