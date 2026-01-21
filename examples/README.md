# Examples

Dieses Verzeichnis enthält Beispiele für die Verwendung von scVAE-Annotator.

## Verfügbare Beispiele

### 1. Basic Example (`basic_example.py`)

Vollständiges End-to-End Beispiel mit dem PBMC 10k Datensatz:

```bash
python examples/basic_example.py
```

Dieses Beispiel zeigt:
- Laden des PBMC 10k Datensatzes
- Konfiguration der optimierten Pipeline
- Ausführung der vollständigen Annotation
- Auswertung der Ergebnisse

### 2. CLI Examples (`cli_examples.sh`)

Kommandozeilen-Beispiele für verschiedene Anwendungsfälle:

```bash
bash examples/cli_examples.sh  # Zeigt verfügbare Befehle
```

## Eigene Daten verwenden

### Aus H5AD-Datei:

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

### Aus Scanpy AnnData:

```python
import scanpy as sc
from scvae_annotator import Config, run_annotation_pipeline

# Laden Sie Ihre Daten
adata = sc.read_h5ad('your_data.h5ad')
# oder: adata = sc.read_10x_mtx('filtered_feature_bc_matrix/')

config = Config(
    data_path=None,  # Nicht benötigt wenn adata direkt übergeben wird
    output_dir='results'
)

results = run_annotation_pipeline(config, adata=adata)
```

## Erweiterte Beispiele

Siehe [EXAMPLES.md](../EXAMPLES.md) im Hauptverzeichnis für:
- Hyperparameter-Optimierung
- Batch-Korrektur mit Harmony
- Benutzerdefinierte Klassifikatoren
- Visualisierungen und Analysen

## Datensets

Beispiel-Datensätze können heruntergeladen werden von:
- [10x Genomics public datasets](https://www.10xgenomics.com/resources/datasets) (z.B. PBMC 10k)
- [Single Cell Portal](https://singlecell.broadinstitute.org/)

### PBMC 3k Vorbereitung:

```bash
python data/prepare_pbmc3k.py
```

Dies bereitet den PBMC 3k Datensatz für die Validierung vor.
