# Troubleshooting Guide

This guide helps you resolve common issues when using scVAE-Annotator.

## Installation Issues

### ModuleNotFoundError: No module named 'scvae_annotator'

**Problem:** The package is not installed or not in the Python path.

**Solution:**
```bash
cd /path/to/scVAE-Annotator
pip install -e .
```

The `-e` flag installs the package in "editable" mode, allowing you to modify the source code without reinstalling.

### Import errors when running examples

**Problem:** Examples can't import the module even after installation.

**Solution 1 (Recommended):**
```bash
pip install -e .
```

**Solution 2 (Alternative):**
```bash
export PYTHONPATH=/path/to/scVAE-Annotator:$PYTHONPATH
python examples/basic_example.py
```

## Runtime Issues

### SciPy Array API Warnings/Errors

**Problem:** You see warnings like:
```
scipy.ndimage.label: object of type 'numpy.ndarray' cannot be interpreted as an integer
```
Or errors related to `scipy.sparse` and array API compatibility.

**Solution:**
Set the environment variable to disable the array API:
```bash
export SCIPY_ARRAY_API=0
```

To make this permanent, add it to your shell configuration:
```bash
# For bash
echo 'export SCIPY_ARRAY_API=0' >> ~/.bashrc
source ~/.bashrc

# For zsh
echo 'export SCIPY_ARRAY_API=0' >> ~/.zshrc
source ~/.zshrc
```

**Why this happens:**
SciPy 1.13+ introduced an experimental array API that may conflict with older NumPy/Scanpy versions. Disabling it ensures compatibility.

### CUDA/GPU Issues

**Problem:** CUDA errors or GPU not detected.

**Solution:**
The pipeline works with CPU-only mode. If you're having GPU issues:

1. Check CUDA installation:
```bash
nvidia-smi
```

2. Verify PyTorch/TensorFlow GPU support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

3. Force CPU mode by setting:
```bash
export CUDA_VISIBLE_DEVICES=""
```

### Memory Issues

**Problem:** Out of memory errors during processing.

**Solution:**
Reduce batch sizes and subsample parameters in the config:

```python
from scvae_annotator import Config

config = Config(
    autoencoder_batch_size=32,  # Default: 64
    subsample_optuna_train=2000,  # Default: 5000
    n_top_genes=2000  # Default: 3000
)
```

### Data Download Issues

**Problem:** Can't download PBMC datasets.

**Solution:**
1. Check internet connection
2. Manually download from [10x Genomics](https://www.10xgenomics.com/resources/datasets)
3. Place in `data/` directory
4. Update config to point to local file:

```python
config = Config(
    data_path='data/pbmc10k_filtered_feature_bc_matrix.h5'
)
```

## Package Dependency Issues

### Conflicting versions

**Problem:** Version conflicts between packages.

**Solution:**
Create a fresh virtual environment:

```bash
python -m venv scvae_env
source scvae_env/bin/activate  # On Windows: scvae_env\Scripts\activate
pip install -e .
```

### Missing system dependencies

**Problem:** Errors installing `python-igraph` or `leidenalg`.

**Solution:**
Install system dependencies first:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev libigraph0-dev
pip install -e .
```

**macOS:**
```bash
brew install igraph
pip install -e .
```

**Windows:**
Use Anaconda/Miniconda:
```bash
conda install -c conda-forge python-igraph leidenalg
pip install -e .
```

## Example-Specific Issues

### basic_example.py fails immediately

**Checklist:**
1. ✅ Package installed: `pip install -e .`
2. ✅ Dependencies installed: `pip install -r requirements.txt`
3. ✅ SCIPY_ARRAY_API set: `export SCIPY_ARRAY_API=0`
4. ✅ Internet connection available
5. ✅ Sufficient disk space (~500MB)

### Pipeline runs but produces poor results

**Tips:**
- Increase training epochs: `config.autoencoder_epochs = 200`
- Increase Optuna trials: `config.optuna_trials = 100`
- Adjust resolution range: `config.leiden_resolution_range = (0.01, 0.5)`
- Ensure proper preprocessing of input data

## Getting Help

If your issue persists:

1. **Check existing issues:** [GitHub Issues](https://github.com/or4k2l/scVAE-Annotator/issues)
2. **Create a new issue** with:
   - Error message (full traceback)
   - Python version: `python --version`
   - Package versions: `pip list | grep -E "scanpy|anndata|scipy|numpy"`
   - Operating system
   - Steps to reproduce

## Quick Reference

Common commands for a fresh start:

```bash
# Clean install
pip uninstall scvae-annotator -y
pip install -e .

# Set environment
export SCIPY_ARRAY_API=0

# Run example
python examples/basic_example.py

# Run tests
pytest tests/ -v
```

## Performance Optimization

### Speed up training

```python
config = Config(
    autoencoder_patience=5,  # Earlier stopping
    optuna_trials=20,  # Fewer trials
    subsample_optuna_train=3000  # Smaller training set
)
```

### Improve accuracy

```python
config = Config(
    autoencoder_epochs=200,  # More training
    autoencoder_patience=15,  # Later stopping
    optuna_trials=100,  # More trials
    n_top_genes=5000  # More genes
)
```
