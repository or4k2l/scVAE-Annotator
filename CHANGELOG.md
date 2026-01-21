# Changelog

Alle wichtigen Änderungen an diesem Projekt werden in dieser Datei dokumentiert.

## [0.1.0] - 2024

### Hinzugefügt

#### Hauptimplementierung
- **scvae_annotator.py**: Vollständige Pipeline-Implementierung (~800 Zeilen)
  - Config-Klasse für Konfigurationsmanagement
  - VAE-Modell mit Early Stopping
  - Leiden-Clustering mit optimalen Parametern
  - Hyperparameter-Optimierung mit Optuna (50 Trials)
  - XGBoost-Klassifikator mit Confidence-Calibration
  - Harmony-Batch-Korrektur Support
  - Visualisierungen (UMAP, Confusion Matrix, Confidence Analysis)

#### Dokumentation
- **README.md**: Hauptprojektübersicht mit vollständiger Beschreibung
- **EXAMPLES.md**: Detaillierte Verwendungsbeispiele und Code-Snippets
- **ANALYSIS_REPORT.md**: Umfassender Analysebericht für PBMC 10k Dataset
  - 99.38% Accuracy auf PBMC 10k
  - Detaillierte Metriken pro Zelltyp
  - Clustering-Qualitätsanalysen (NMI: 0.9832, ARI: 0.9701)
- **VALIDATION_REPORT.md**: Cross-Dataset Validierung
  - 93.01% Accuracy auf PBMC 3k
  - Generalisierungsfähigkeit nachgewiesen
- **TECHNICAL_APPENDIX.md**: Detaillierte technische Dokumentation
  - Hyperparameter-Optimierungsergebnisse
  - Modellarchitektur-Details
  - Performance-Benchmarks
- **CONTRIBUTING.md**: Richtlinien für Beitragende
- **docs/README.md**: Zentrale Dokumentationsübersicht mit API-Referenz
- **examples/README.md**: Übersicht über verfügbare Beispiele
- **data/README.md**: Anleitung zur Datenvorbereitung
- **figures/README.md**: Visualisierungsgalerie

#### Daten und Skripte
- **data/prepare_pbmc3k.py**: Skript zur PBMC 3k Datenvorbereitung
- **examples/basic_example.py**: Vollständiges End-to-End Beispiel
- **examples/cli_examples.sh**: Kommandozeilen-Verwendungsbeispiele

#### Tests
- **tests/test_annotator.py**: Tests für Config und Pipeline-Komponenten
- **tests/test_model.py**: Tests für VAE-Modell-Architektur

#### Konfiguration
- **requirements.txt**: Vollständige Abhängigkeitenliste (17 Pakete)
  - scanpy>=1.9.0
  - torch>=1.12.0
  - optuna>=3.0.0
  - xgboost>=1.6.0
  - scikit-learn>=1.2.0
  - harmony-pytorch>=0.1.0
  - und weitere
- **pyproject.toml**: Build-Konfiguration mit allen Abhängigkeiten

### Optimiert

#### Pipeline-Architektur
- **Optimale Hyperparameter** (via Optuna mit 50 Trials):
  - Target Genes: 2000
  - n_neighbors: 30
  - Leiden Resolution: 0.4
  - Latent Dimension: 32 (512→256→128→64→32)
  - VAE Epochs: 100 (mit Early Stopping, Patience: 10)

#### Modell-Performance
- **PBMC 10k Dataset**:
  - Accuracy: 99.38%
  - Balanced Accuracy: 99.22%
  - Macro F1-Score: 0.9928
  - Weighted F1-Score: 0.9938
  - NMI: 0.9832
  - ARI: 0.9701

- **PBMC 3k Dataset** (Validierung):
  - Accuracy: 93.01%
  - Demonstrierte Generalisierungsfähigkeit

#### Klassifikator-Auswahl
- XGBoost ausgewählt als bester Klassifikator
- Kalibrierte Confidence Scores via Platt Scaling
- Robuste Performance über alle Zelltypen

### Aktualisiert

#### Package-Struktur
- **src/scvae_annotator/__init__.py**: 
  - Import der Hauptkomponenten aus Root-Level scvae_annotator.py
  - Fallback auf Legacy-Module für Rückwärtskompatibilität
  - Klare Dokumentation der Paketstruktur

- **examples/**: Aktualisiert auf neue API
  - basic_example.py verwendet create_optimized_config()
  - cli_examples.sh zeigt Python-API statt CLI-Tool

- **tests/**: Aktualisiert auf neue Pipeline-API
  - test_annotator.py testet Config-Klasse
  - test_model.py testet VAE-Architektur

### Behoben
- Inkonsistenzen zwischen pyproject.toml und requirements.txt
- Import-Fehler in src/scvae_annotator/__init__.py
- Veraltete API-Verwendung in Beispielen
- Fehlende Fehlerbehandlung beim Package-Import

### Technische Details

#### Architektur
- VAE: 5-Layer Deep Architecture (512→256→128→64→32D)
- Leiden Clustering: Adaptive Resolution (0.4)
- Classifier: XGBoost mit Hyperparameter-Tuning
- Confidence: Platt Scaling Calibration

#### Workflow
1. Daten-Preprocessing (Normalisierung, HVG-Selektion)
2. Optional: Harmony Batch-Korrektur
3. VAE-Training mit Early Stopping
4. Leiden Clustering auf VAE-Embeddings
5. Feature-Extraktion (PCA, Cluster-Stats, VAE-Features)
6. XGBoost-Training mit Confidence-Calibration
7. Evaluation und Visualisierung

#### Performance-Charakteristika
- Training Time: ~5-10 min (PBMC 10k, CPU)
- Memory: ~2-4 GB RAM
- GPU: Optional, automatisch erkannt
- Skalierbar für große Datasets (>100k Zellen)

### Bekannte Einschränkungen
- src/scvae_annotator/* Module sind Legacy-Platzhalter
- Hauptimplementierung liegt in Root-Level scvae_annotator.py
- Zukünftige Modularisierung geplant

---

## Geplante Features

### Version 0.2.0 (Geplant)
- [ ] Modularisierung der Pipeline in separate Module
- [ ] CLI-Tool für Kommandozeilen-Verwendung
- [ ] Web-Interface für interaktive Analyse
- [ ] Unterstützung für weitere Datasets
- [ ] Pre-trained Models für häufige Zelltypen
- [ ] Integration mit Cell Ontology

### Version 0.3.0 (Geplant)
- [ ] Multi-Batch Support
- [ ] Transfer Learning
- [ ] Explainable AI Features
- [ ] Docker Container
- [ ] Jupyter Notebook Tutorials

---

## Mitwirkende

Vielen Dank an alle, die zu diesem Projekt beigetragen haben!

---

## Lizenz

Siehe [LICENSE](LICENSE) für Details.
