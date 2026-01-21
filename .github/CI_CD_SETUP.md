# CI/CD Setup

This document describes the CI/CD setup for scVAE-Annotator.

## GitHub Actions Workflows

### 1. Tests (`tests.yml`)

Runs on every push/PR to `main` and `develop` branches.

**Matrix Tests:**
- **Operating Systems**: Ubuntu, macOS, Windows
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Weekly Cron Job**: Sundays at 00:00 UTC

**Steps:**
1. Checkout code
2. Python setup with cache
3. Install system dependencies (HDF5)
4. Install Python dependencies
5. Run tests with coverage
6. Upload coverage to Codecov
7. HTML coverage report as artifact

**Commands:**
```bash
pytest tests/ -v --cov=. --cov-report=xml --cov-report=html
```

### 2. Code Quality (`lint.yml`)

Runs on every push/PR to `main` and `develop` branches.

**Checks:**
- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Style guide enforcement (PEP 8)
- **Pylint**: Code analysis and quality
- **Bandit**: Security checks
- **Safety**: Dependency vulnerability checks

**Commands:**
```bash
black --check --diff .
isort --check-only --diff .
flake8 .
pylint scvae_annotator.py src/
bandit -r .
safety check
```

### 3. Documentation (`docs.yml`)

Runs on push/PR to `main` branch.

**Checks:**
- Validate markdown links
- Check README structure
- Sphinx build (planned)

### 4. Release (`release.yml`)

Runs on version tags (`v*.*.*`).

**Steps:**
1. Build Python package
2. Check package quality
3. Create GitHub release
4. (Optional) PyPI upload

**Trigger:**
```bash
git tag v0.1.0
git push origin v0.1.0
```

## Badges

Add these badges to the README:

```markdown
[![Python Tests](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/tests.yml/badge.svg)](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/tests.yml)
[![Code Quality](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/lint.yml/badge.svg)](https://github.com/or4k2l/scVAE-Annotator/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/or4k2l/scVAE-Annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/or4k2l/scVAE-Annotator)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

## Local CI Simulation

### With run_tests.sh

```bash
# Full CI simulation
./run_tests.sh --full

# Linting only
./run_tests.sh --lint

# Auto-fix
./run_tests.sh --fix
```

### With Make

```bash
# Tests
make test          # Standard
make test-quick    # Quick
make test-full     # Full

# Linting
make lint          # Check
make lint-fix      # Auto-fix
```

### Manual

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

## Coverage Goals

- **Minimum**: 70% overall coverage
- **Target**: 80%+ overall coverage
- **Critical Functions**: 100% coverage

### View Coverage

```bash
# Terminal
pytest --cov=. --cov-report=term-missing

# HTML report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# XML for CI
pytest --cov=. --cov-report=xml
```

## Pre-Commit Hooks (Optional)

For automatic checks before each commit:

```bash
# Install pre-commit
pip install pre-commit

# Activate hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:

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

After successful release workflow:

1. Package is automatically built
2. GitHub release is created
3. (Optional) Upload to PyPI

Add PyPI token:
```bash
# GitHub Repository Settings > Secrets
PYPI_API_TOKEN=<your-token>
```

### Docker Build (Planned)

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

### Status Checks

All PRs must pass the following checks:

- ✅ Tests (Ubuntu, Python 3.10)
- ✅ Linting (Black, Flake8, Pylint)
- ✅ Coverage ≥ 70%

### Branch Protection

Enable branch protection for `main`:

1. GitHub Repository > Settings > Branches
2. Add rule for `main`
3. Enable:
   - Require status checks before merging
   - Require branches to be up to date
   - Mark tests and lint as required

## Troubleshooting

### Workflow Errors

**Problem**: Tests fail on Windows

**Solution**: Check path separators (`/` vs `\`)

**Problem**: Coverage upload fails

**Solution**: Add Codecov token to repository secrets

**Problem**: Dependency installation takes too long

**Solution**: Use `cache: 'pip'` in setup-python action

### Local Errors

**Problem**: `pytest` not found

**Solution**: 
```bash
pip install pytest pytest-cov
```

**Problem**: Import errors in tests

**Solution**:
```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Further Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Codecov Documentation](https://docs.codecov.com/)
