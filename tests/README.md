# Tests

Dieses Verzeichnis enthält alle Tests für scVAE-Annotator.

## Test-Struktur

```
tests/
├── __init__.py
├── test_annotator.py       # Config und Pipeline Tests
├── test_model.py            # VAE Architektur Tests
├── test_preprocessing.py    # Preprocessing-Funktionen Tests
└── test_integration.py      # End-to-End Integration Tests
```

## Tests ausführen

### Alle Tests

```bash
pytest tests/
```

### Mit Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Schnelle Tests (ohne slow marker)

```bash
pytest tests/ -m "not slow"
```

### Spezifische Test-Datei

```bash
pytest tests/test_annotator.py -v
```

### Mit Make

```bash
make test              # Standard Tests
make test-quick        # Nur schnelle Tests
make test-full         # Alle Tests mit Coverage
```

### Mit run_tests.sh Skript

```bash
./run_tests.sh              # Standard
./run_tests.sh --quick      # Schnelle Tests
./run_tests.sh --full       # Vollständige Suite
./run_tests.sh --lint       # Nur Linting
./run_tests.sh --fix        # Auto-fix Linting
```

## Test-Kategorien

### Unit Tests (`test_annotator.py`, `test_model.py`)

Testen einzelne Funktionen und Klassen isoliert:

- Config-Initialisierung
- VAE-Architektur
- Modell-Komponenten

```bash
pytest tests/test_annotator.py tests/test_model.py -v
```

### Preprocessing Tests (`test_preprocessing.py`)

Testen Datenverarbeitungs-Pipeline:

- Filtern von Zellen und Genen
- Normalisierung
- Log-Transformation
- HVG-Selektion
- PCA, UMAP, Clustering

```bash
pytest tests/test_preprocessing.py -v
```

### Integration Tests (`test_integration.py`)

Testen die komplette Pipeline End-to-End:

- Vollständige Pipeline mit synthetischen Daten
- Output-Validierung
- Fehlerbehandlung

⚠️ **Hinweis**: Integration-Tests sind als "slow" markiert und werden standardmäßig übersprungen.

```bash
pytest tests/test_integration.py -v -m "slow"
```

## Test Markers

Tests können mit Markern versehen werden:

- `@pytest.mark.slow` - Langsame Tests (>10 Sekunden)
- `@pytest.mark.integration` - Integration Tests
- `@pytest.mark.unit` - Unit Tests

### Marker verwenden

```bash
# Nur Unit Tests
pytest -m unit

# Ohne langsame Tests
pytest -m "not slow"

# Nur Integration Tests
pytest -m integration
```

## Code Coverage

### Coverage Report generieren

```bash
pytest --cov=. --cov-report=html
```

Der Report wird in `htmlcov/index.html` gespeichert.

### Coverage im Terminal

```bash
pytest --cov=. --cov-report=term-missing
```

### Coverage-Ziel

Angestrebte Coverage: **≥ 80%**

Aktuelle Coverage anzeigen:

```bash
coverage report
```

## Fixtures

Verfügbare Pytest Fixtures (in `test_integration.py`):

### `temp_output_dir`

Temporäres Verzeichnis für Test-Outputs:

```python
def test_example(temp_output_dir):
    # temp_output_dir ist ein temporäres Verzeichnis
    # wird nach dem Test automatisch gelöscht
    pass
```

### `synthetic_adata`

Synthetische AnnData für Tests:

```python
def test_example(synthetic_adata):
    # synthetic_adata ist ein AnnData Objekt mit 200 Zellen, 100 Genen
    assert synthetic_adata.n_obs == 200
    assert synthetic_adata.n_vars == 100
```

## Kontinuierliche Integration (CI)

### GitHub Actions

Tests werden automatisch bei jedem Push/PR ausgeführt:

- **tests.yml**: Läuft auf Ubuntu, macOS, Windows mit Python 3.8-3.12
- **lint.yml**: Code-Qualitätschecks (Black, Flake8, Pylint, etc.)
- **docs.yml**: Dokumentations-Validierung

### Lokale CI-Simulation

```bash
# Simuliere GitHub Actions lokal
./run_tests.sh --full
make lint
```

## Troubleshooting

### Import-Fehler

Stelle sicher, dass das Paket installiert ist:

```bash
pip install -r requirements.txt
```

### Timeout-Fehler

Erhöhe das Timeout für langsame Tests:

```bash
pytest --timeout=60
```

### Memory-Fehler bei großen Tests

Reduziere parallele Workers:

```bash
pytest -n 2  # Nur 2 parallele Prozesse
```

### Coverage nicht gefunden

Installiere Coverage-Tools:

```bash
pip install pytest-cov coverage[toml]
```

## Best Practices

### Neue Tests schreiben

1. **Dateinamen**: `test_*.py`
2. **Funktionsnamen**: `test_*`
3. **Docstrings**: Beschreibe was getestet wird
4. **Assertions**: Verwende klare Assertions
5. **Fixtures**: Nutze Fixtures für Setup/Teardown

Beispiel:

```python
def test_config_initialization():
    """Test dass Config mit Defaults initialisiert wird."""
    config = Config()
    assert config.target_genes == 2000
    assert config.n_neighbors == 30
```

### Test-Isolation

Jeder Test sollte unabhängig sein:

- Keine Abhängigkeiten zwischen Tests
- Temporäre Dateien aufräumen
- Fixture für Setup verwenden

### Performance

- Markiere langsame Tests mit `@pytest.mark.slow`
- Verwende Mock-Objekte wo möglich
- Reduziere Datengrößen in Tests

## Zusätzliche Tools

### pytest-xdist (Parallel Tests)

```bash
pip install pytest-xdist
pytest -n auto  # Automatische CPU-Anzahl
```

### pytest-timeout (Timeout Protection)

```bash
pip install pytest-timeout
pytest --timeout=30
```

### pytest-watch (Auto-Rerun)

```bash
pip install pytest-watch
ptw  # Läuft Tests automatisch bei Dateiänderungen
```

## Weitere Informationen

- [pytest Dokumentation](https://docs.pytest.org/)
- [Coverage.py Dokumentation](https://coverage.readthedocs.io/)
- [GitHub Actions für Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
