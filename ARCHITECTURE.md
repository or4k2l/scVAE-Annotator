# scVAE-Annotator Architecture

## Module Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     scvae_annotator                         │
│                     ================                         │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐    ┌─────────────┐ │
│  │   config.py  │────▶│  pipeline.py │◀───│   cli.py    │ │
│  │  (102 Zeilen)│     │ (255 Zeilen) │    │ (152 Zeilen)│ │
│  └──────────────┘     └──────┬───────┘    └─────────────┘ │
│         │                    │                              │
│         │                    ▼                              │
│         │         ┌──────────────────┐                      │
│         └────────▶│ preprocessing.py │                      │
│                   │  (132 Zeilen)    │                      │
│                   └────────┬─────────┘                      │
│                            │                                │
│                            ▼                                │
│                   ┌──────────────────┐                      │
│                   │  clustering.py   │                      │
│                   │   (75 Zeilen)    │                      │
│                   └────────┬─────────┘                      │
│                            │                                │
│                            ▼                                │
│                   ┌──────────────────┐                      │
│                   │     vae.py       │                      │
│                   │  (173 Zeilen)    │                      │
│                   └────────┬─────────┘                      │
│                            │                                │
│                            ▼                                │
│                   ┌──────────────────┐                      │
│                   │  annotator.py    │                      │
│                   │  (230 Zeilen)    │                      │
│                   └────────┬─────────┘                      │
│                            │                                │
│                            ▼                                │
│                   ┌──────────────────┐                      │
│                   │visualization.py  │                      │
│                   │  (54 Zeilen)     │                      │
│                   └──────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. CLI / Python API
        │
        ▼
2. Create config
        │
        ▼
3. Data Loading & Preprocessing
   └─ download_data()
   └─ load_and_prepare_data()
   └─ enhanced_preprocessing()
        │
        ▼
4. Leiden Clustering
   └─ optimized_leiden_clustering()
        │
        ▼
5. VAE Training
   └─ train_improved_vae()
        │
        ▼
6. Cell Type Annotation
   └─ EnhancedAutoencoderAnnotator.train()
   └─ EnhancedAutoencoderAnnotator.predict()
        │
        ▼
7. Evaluation & Visualization
   └─ evaluate_predictions()
   └─ create_visualizations()
        │
        ▼
8. Results & Reports
   └─ annotated_data.h5ad
        └─ Plots (UMAP, confusion matrix)
   └─ Metrics (JSON, CSV)
```

## Module Responsibilities

| Module | Responsibility | LoC |
|-------|--------------|-----|
| `config.py` | Configuration, parameters, logging | 102 |
| `preprocessing.py` | Data processing, QC, normalization | 132 |
| `clustering.py` | Leiden clustering, metric optimization | 75 |
| `vae.py` | VAE architecture, training, early stopping | 173 |
| `annotator.py` | Classification, Optuna, calibration | 230 |
| `visualization.py` | UMAP plots, confidence plots | 54 |
| `pipeline.py` | Orchestration, evaluation | 255 |
| `cli.py` | Command-line interface | 152 |
| `__init__.py` | Package exports | 75 |
| `__main__.py` | Python module entry point | 8 |
| **TOTAL** | | **1256** |

## Comparison: Before vs. After

### Before (Monolith)
```
scvae_annotator.py (997 lines) ❌
└─ Everything in one file
```

### After (Modular)
```
src/scvae_annotator/ ✅
├── config.py        (102 lines)
├── preprocessing.py (132 lines)
├── clustering.py    ( 75 lines)
├── vae.py          (173 lines)
├── annotator.py    (230 lines)
├── visualization.py ( 54 lines)
├── pipeline.py     (255 lines)
├── cli.py          (152 lines)
├── __init__.py     ( 75 lines)
└── __main__.py     (  8 lines)
```

## Benefits of the New Architecture

✅ **Separation of concerns**: Each module has a clear role  
✅ **Testability**: Modules can be tested in isolation  
✅ **Reusability**: Modules can be reused in other projects  
✅ **Maintainability**: Smaller files are easier to understand  
✅ **Extensibility**: New features are easy to add  
✅ **Type safety**: Better IDE support  
✅ **Performance**: Optimized imports  

## Installation

```bash
pip install -e .
```

## Usage

### Python API
```python
from scvae_annotator import create_optimized_config, run_annotation_pipeline

config = create_optimized_config()
adata = run_annotation_pipeline(config)
```

### Command Line
```bash
scvae-annotate --data data.h5 --annotations annotations.csv
```

### Python Module
```bash
python -m scvae_annotator --data data.h5
```
