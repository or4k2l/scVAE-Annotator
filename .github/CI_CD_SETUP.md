# CI/CD Setup

Dieses Dokument beschreibt das CI/CD-Setup für scVAE-Annotator.

## GitHub Actions Workflows

### 1. Tests (`tests.yml`)

Läuft bei jedem Push/PR auf `main` und `develop` Branches.

**Matrix-Tests:**
- **Betriebssysteme**: Ubuntu, macOS, Windows
- **Python-Versionen**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Wöchentlicher Cron-Job**: Sonntags um 00:00 UTC

**Schritte:**
1. Checkout Code
2. Python Setup mit Cache
3. System-Abhängigkeiten installieren (HDF5)
4. Python-Abhängigkeiten installieren
5. Tests mit Coverage ausführen
6. Coverage zu Codecov hochladen
7. HTML Coverage-Report als Artifact

**Kommandos:**
```bash
pytest tests/ -v --cov=. --cov-report=xml --cov-report=html
```

### 2. Code Quality (`lint.yml`)

Läuft bei jedem Push/PR auf `main` und `develop` Branches.

**Checks:**
- **Black**: Code-Formatierung
- **isort**: Import-Sortierung
- **Flake8**: Style-Guide Enforcement (PEP 8)
- **Pylint**: Code-Analyse und Qualität
- **Bandit**: Security-Checks
- **Safety**: Dependency-Vulnerability-Checks

**Kommandos:**
```bash
black --check --diff .
isort --check-only --diff .
flake8 .
pylint scvae_annotator.py src/
bandit -r .
safety check
```

### 3. Documentation (`docs.yml`)

Läuft bei Push/PR auf `main` Branch.

**Checks:**
- Markdown-Links validieren
- README-Struktur prüfen
- Sphinx-Build (geplant)

### 4. Release (`release.yml`)

Läuft bei Version-Tags (`v*.*.*`).

**Schritte:**
1. Build Python Package
2. Package-Qualität prüfen
3. GitHub Release erstellen
4. (Optional) PyPI Upload

**Trigger:**
```bash
git tag v0.1.0
git push origin v0.1.0
```

## Badges

Füge diese Badges zum README hinzu:

```markdown
[![Python Tests](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/tests.yml/badge.svg)](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/tests.yml)
[![Code Quality](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/lint.yml/badge.svg)](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/or4k2l/scVAE-Annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/or4k2l/scVAE-Annotator)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

## Lokale CI-Simulation

### Mit run_tests.sh

```bash
# Vollständige CI-Simulation
./run_tests.sh --full

# Nur Linting
./run_tests.sh --lint

# Auto-Fix
./run_tests.sh --fix
```

### Mit Make

```bash
# Tests
make test          # Standard
make test-quick    # Schnell
make test-full     # Vollständig

# Linting
make lint          # Prüfen
make lint-fix      # Auto-Fix
```

### Manuell

```bash
# Tests
pytest tests/ -v --cov=. --cov-report=html

# Black
black --check .

# Flake8
flake8 .

# Pylint
pylint scvae_annotator.py src/

# Bandit
bandit -r .

# Safety
safety check
```

## Coverage-Ziele

- **Minimum**: 70% Gesamt-Coverage
- **Ziel**: 80%+ Gesamt-Coverage
- **Kritische Funktionen**: 100% Coverage

### Coverage anzeigen

```bash
# Terminal
pytest --cov=. --cov-report=term-missing

# HTML-Report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# XML für CI
pytest --cov=. --cov-report=xml
```

## Pre-Commit Hooks (Optional)

Für automatische Checks vor jedem Commit:

```bash
# pre-commit installieren
pip install pre-commit

# Hooks aktivieren
pre-commit install

# Manuell ausführen
pre-commit run --all-files
```

`.pre-commit-config.yaml` erstellen:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Continuous Deployment

### PyPI Release

Nach erfolgreichem Release-Workflow:

1. Package wird automatisch gebaut
2. GitHub Release wird erstellt
3. (Optional) Upload zu PyPI

PyPI Token hinzufügen:
```bash
# GitHub Repository Settings > Secrets
PYPI_API_TOKEN=<your-token>
```

### Docker Build (Geplant)

```yaml
# .github/workflows/docker.yml
name: Docker Build
on:
  push:
    tags: ['v*']
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: scvae-annotator:latest
```

## Monitoring

### Status-Checks

Alle PRs müssen folgende Checks bestehen:

- ✅ Tests (Ubuntu, Python 3.10)
- ✅ Linting (Black, Flake8, Pylint)
- ✅ Coverage ≥ 70%

### Branch Protection

Aktiviere Branch Protection für `main`:

1. GitHub Repository > Settings > Branches
2. Add Rule für `main`
3. Aktiviere:
   - Require status checks before merging
   - Require branches to be up to date
   - Tests und Lint als erforderlich markieren

## Troubleshooting

### Workflow-Fehler

**Problem**: Tests schlagen auf Windows fehl

**Lösung**: Prüfe Pfad-Separatoren (`/` vs `\`)

**Problem**: Coverage-Upload schlägt fehl

**Lösung**: Codecov Token in Repository Secrets hinzufügen

**Problem**: Dependency-Installation dauert zu lange

**Lösung**: Nutze `cache: 'pip'` in setup-python Action

### Lokale Fehler

**Problem**: `pytest` nicht gefunden

**Lösung**: 
```bash
pip install pytest pytest-cov
```

**Problem**: Import-Fehler in Tests

**Lösung**:
```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Weitere Ressourcen

- [GitHub Actions Dokumentation](https://docs.github.com/en/actions)
- [pytest Dokumentation](https://docs.pytest.org/)
- [Coverage.py Dokumentation](https://coverage.readthedocs.io/)
- [Codecov Dokumentation](https://docs.codecov.com/)
