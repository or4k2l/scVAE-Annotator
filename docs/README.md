# Dokumentation

Diese Verzeichnis enthält die Dokumentation für scVAE-Annotator.

## Schnellstart

### Installation

```bash
# Repository klonen
git clone https://github.com/yourusername/scVAE-Annotator.git
cd scVAE-Annotator

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Verwendung

```python
from scvae_annotator import create_optimized_config, run_annotation_pipeline

# Optimierte Konfiguration erstellen
config = create_optimized_config()

# Pipeline ausführen
results = run_annotation_pipeline(config)
```

## Dokumentationsstruktur

### Haupt-Dokumentation

- [README.md](../README.md) - Hauptprojektübersicht und Schnellstart
- [EXAMPLES.md](../EXAMPLES.md) - Detaillierte Verwendungsbeispiele
- [ANALYSIS_REPORT.md](../ANALYSIS_REPORT.md) - Vollständiger Analysebericht für PBMC 10k
- [VALIDATION_REPORT.md](../VALIDATION_REPORT.md) - Cross-Dataset Validierung
- [TECHNICAL_APPENDIX.md](../TECHNICAL_APPENDIX.md) - Detaillierte technische Metriken

### Weitere Dokumentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Anleitung für Beitragende
- [LICENSE](../LICENSE) - Lizenzinformationen

## Daten

- [data/README.md](../data/README.md) - Anleitung zur Datenvorbereitung
- [data/prepare_pbmc3k.py](../data/prepare_pbmc3k.py) - PBMC 3k Vorbereitungsskript

## Visualisierungen

- [figures/README.md](../figures/README.md) - Übersicht über alle Visualisierungen

## API-Referenz

### Hauptkomponenten

#### Config

Konfigurationsklasse für die Pipeline:

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

Erstellt optimierte Konfiguration basierend auf den Ergebnissen der Hyperparameter-Optimierung:

```python
config = create_optimized_config()
```

#### run_annotation_pipeline()

Führt die vollständige Annotations-Pipeline aus:

```python
results = run_annotation_pipeline(config)
```

### Metriken

Die Pipeline liefert folgende Metriken:

- **Accuracy**: Gesamtgenauigkeit der Annotation
- **Precision/Recall/F1**: Pro-Klassen-Metriken
- **Confidence Scores**: Kalibrierte Konfidenzwerte
- **Clustering Metrics**: NMI, ARI, Silhouette Score

## Erweiterte Themen

### Eigene Daten

Siehe [EXAMPLES.md](../EXAMPLES.md) für Anleitungen zur Verwendung eigener Daten.

### Hyperparameter-Tuning

Die Pipeline verwendet Optuna für automatisches Hyperparameter-Tuning. 
Siehe [TECHNICAL_APPENDIX.md](../TECHNICAL_APPENDIX.md) für Details.

### Batch-Korrektur

Harmony-Integration für Batch-Effekt-Korrektur:

```python
config = Config(
    use_harmony=True,
    harmony_theta=2.0
)
```

## Troubleshooting

### Häufige Probleme

**Problem**: `ModuleNotFoundError: No module named 'scvae_annotator'`

**Lösung**: Stellen Sie sicher, dass alle Abhängigkeiten installiert sind:
```bash
pip install -r requirements.txt
```

**Problem**: GPU-Fehler

**Lösung**: Die Pipeline funktioniert auch mit CPU. PyTorch erkennt automatisch verfügbare Hardware.

**Problem**: Speicherfehler bei großen Datensätzen

**Lösung**: Reduzieren Sie `target_genes` oder verwenden Sie Batch-Processing:
```python
config = Config(target_genes=1000)
```

## Community

- **Issues**: Melden Sie Bugs oder Feature-Requests auf GitHub
- **Discussions**: Diskussionen und Fragen im GitHub Discussions Bereich
- **Pull Requests**: Beiträge sind willkommen! Siehe [CONTRIBUTING.md](../CONTRIBUTING.md)

## Zitate

Wenn Sie scVAE-Annotator in Ihrer Forschung verwenden, zitieren Sie bitte:

```bibtex
@software{scvae_annotator,
  title = {scVAE-Annotator: Automated Cell Type Annotation for scRNA-seq},
  author = {scVAE-Annotator Team},
  year = {2024},
  url = {https://github.com/yourusername/scVAE-Annotator}
}
```
