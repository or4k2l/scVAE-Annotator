# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-01-21

### Changed

#### Improved "Out-of-the-Box" Experience
- **examples/basic_example.py**: Improved error messages and import logic
  - Clearer guidance to use `pip install -e .` as the primary installation method
  - More flexible SCIPY_ARRAY_API recommendations (both 0 and 1)
  - Removal of PYTHONPATH workarounds in favor of package installation

- **TROUBLESHOOTING.md**: Expanded SciPy Array API documentation
  - Documentation of both SCIPY_ARRAY_API values (0 and 1)
  - Explanation that the correct value depends on the specific environment
  - Step-by-step instructions to test both options

- **README.md** and **examples/README.md**: Consistent updates
  - Unified notes on SCIPY_ARRAY_API flexibility
  - Improved installation instructions

### Motivation
These changes are based on user feedback to further improve usability:
- Eliminate manual workarounds in example scripts
- Account for different scipy/numpy versions across environments
- Focus on `pip install -e .` as the primary and simplest installation method

## [0.1.0] - 2024

### Added

#### Main Implementation
- **scvae_annotator.py**: Complete pipeline implementation (~800 lines)
  - Config class for configuration management
  - VAE model with early stopping
  - Leiden clustering with optimal parameters
  - Hyperparameter optimization with Optuna (50 trials)
  - XGBoost classifier with confidence calibration
  - Harmony batch correction support
  - Visualizations (UMAP, confusion matrix, confidence analysis)

#### Documentation
- **README.md**: Main project overview with complete description
- **EXAMPLES.md**: Detailed usage examples and code snippets
- **ANALYSIS_REPORT.md**: Comprehensive analysis report for PBMC 10k dataset
  - 99.38% accuracy on PBMC 10k
  - Detailed metrics per cell type
  - Clustering quality analyses (NMI: 0.9832, ARI: 0.9701)
- **VALIDATION_REPORT.md**: Cross-dataset validation
  - 93.01% accuracy on PBMC 3k
  - Demonstrated generalization capability
- **TECHNICAL_APPENDIX.md**: Detailed technical documentation
  - Hyperparameter optimization results
  - Model architecture details
  - Performance benchmarks
- **CONTRIBUTING.md**: Contribution guidelines
- **docs/README.md**: Central documentation overview with API reference
- **examples/README.md**: Overview of available examples
- **data/README.md**: Data preparation guide
- **figures/README.md**: Visualization gallery

#### Data and Scripts
- **data/prepare_pbmc3k.py**: Script for PBMC 3k data preparation
- **examples/basic_example.py**: Complete end-to-end example
- **examples/cli_examples.sh**: Command-line usage examples

#### Tests
- **tests/test_annotator.py**: Tests for Config and pipeline components
- **tests/test_model.py**: Tests for VAE model architecture

#### Configuration
- **requirements.txt**: Complete dependency list (17 packages)
  - scanpy>=1.9.0
  - torch>=1.12.0
  - optuna>=3.0.0
  - xgboost>=1.6.0
  - scikit-learn>=1.2.0
  - harmony-pytorch>=0.1.0
  - and more
- **pyproject.toml**: Build configuration with all dependencies

### Optimized

#### Pipeline Architecture
- **Optimal Hyperparameters** (via Optuna with 50 trials):
  - Target Genes: 2000
  - n_neighbors: 30
  - Leiden Resolution: 0.4
  - Latent Dimension: 32 (512→256→128→64→32)
  - VAE Epochs: 100 (with early stopping, patience: 10)

#### Model Performance
- **PBMC 10k Dataset**:
  - Accuracy: 99.38%
  - Balanced Accuracy: 99.22%
  - Macro F1-Score: 0.9928
  - Weighted F1-Score: 0.9938
  - NMI: 0.9832
  - ARI: 0.9701

- **PBMC 3k Dataset** (Validation):
  - Accuracy: 93.01%
  - Demonstrated generalization capability

#### Classifier Selection
- XGBoost selected as best classifier
- Calibrated confidence scores via Platt scaling
- Robust performance across all cell types

### Updated

#### Package Structure
- **src/scvae_annotator/__init__.py**: 
  - Import of main components from root-level scvae_annotator.py
  - Fallback to legacy modules for backwards compatibility
  - Clear documentation of package structure

- **examples/**: Updated to new API
  - basic_example.py uses create_optimized_config()
  - cli_examples.sh shows Python API instead of CLI tool

- **tests/**: Updated to new pipeline API
  - test_annotator.py tests Config class
  - test_model.py tests VAE architecture

### Fixed
- Inconsistencies between pyproject.toml and requirements.txt
- Import errors in src/scvae_annotator/__init__.py
- Outdated API usage in examples
- Missing error handling in package import

### Technical Details

#### Architecture
- VAE: 5-layer deep architecture (512→256→128→64→32D)
- Leiden Clustering: Adaptive resolution (0.4)
- Classifier: XGBoost with hyperparameter tuning
- Confidence: Platt scaling calibration

#### Workflow
1. Data preprocessing (normalization, HVG selection)
2. Optional: Harmony batch correction
3. VAE training with early stopping
4. Leiden clustering on VAE embeddings
5. Feature extraction (PCA, cluster stats, VAE features)
6. XGBoost training with confidence calibration
7. Evaluation and visualization

#### Performance Characteristics
- Training Time: ~5-10 min (PBMC 10k, CPU)
- Memory: ~2-4 GB RAM
- GPU: Optional, automatically detected
- Scalable for large datasets (>100k cells)

### Known Limitations
- src/scvae_annotator/* modules are legacy placeholders
- Main implementation is in root-level scvae_annotator.py
- Future modularization planned

---

## Planned Features

### Version 0.2.0 (Planned)
- [ ] Pipeline modularization into separate modules
- [ ] CLI tool for command-line usage
- [ ] Web interface for interactive analysis
- [ ] Support for additional datasets
- [ ] Pre-trained models for common cell types
- [ ] Integration with Cell Ontology

### Version 0.3.0 (Planned)
- [ ] Multi-batch support
- [ ] Transfer learning
- [ ] Explainable AI features
- [ ] Docker container
- [ ] Jupyter notebook tutorials

---

## Contributors

Thank you to all who have contributed to this project!

---

## License

See [LICENSE](LICENSE) for details.
