# scVAE-Annotator Architektur

## Modulübersicht

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

## Datenfluss

```
1. CLI / Python API
        │
        ▼
2. Config erstellen
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
   └─ Plots (UMAP, Confusion Matrix)
   └─ Metrics (JSON, CSV)
```

## Modul-Verantwortlichkeiten

| Modul | Verantwortung | LoC |
|-------|--------------|-----|
| `config.py` | Konfiguration, Parameter, Logging | 102 |
| `preprocessing.py` | Datenverarbeitung, QC, Normalisierung | 132 |
| `clustering.py` | Leiden-Clustering, Metrik-Optimierung | 75 |
| `vae.py` | VAE-Architektur, Training, Early-Stopping | 173 |
| `annotator.py` | Klassifikation, Optuna, Calibration | 230 |
| `visualization.py` | UMAP-Plots, Confidence-Plots | 54 |
| `pipeline.py` | Orchestrierung, Evaluation | 255 |
| `cli.py` | Command-Line Interface | 152 |
| `__init__.py` | Paket-Exports | 75 |
| `__main__.py` | Python-Modul-Einstieg | 8 |
| **TOTAL** | | **1256** |

## Vergleich: Vorher vs. Nachher

### Vorher (Monolith)
```
scvae_annotator.py (997 Zeilen) ❌
└─ Alles in einer Datei
```

### Nachher (Modular)
```
src/scvae_annotator/ ✅
├── config.py        (102 Zeilen)
├── preprocessing.py (132 Zeilen)
├── clustering.py    ( 75 Zeilen)
├── vae.py          (173 Zeilen)
├── annotator.py    (230 Zeilen)
├── visualization.py ( 54 Zeilen)
├── pipeline.py     (255 Zeilen)
├── cli.py          (152 Zeilen)
├── __init__.py     ( 75 Zeilen)
└── __main__.py     (  8 Zeilen)
```

## Vorteile der neuen Architektur

✅ **Separation of Concerns**: Jedes Modul hat eine klare Aufgabe  
✅ **Testbarkeit**: Module können einzeln getestet werden  
✅ **Wiederverwendbarkeit**: Module können in anderen Projekten genutzt werden  
✅ **Wartbarkeit**: Kleinere Dateien sind einfacher zu verstehen  
✅ **Erweiterbarkeit**: Neue Features einfach hinzuzufügen  
✅ **Type-Safety**: Bessere IDE-Unterstützung  
✅ **Performance**: Optimierte Imports  

## Installation

```bash
pip install -e .
```

## Verwendung

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

### Python Modul
```bash
python -m scvae_annotator --data data.h5
```
