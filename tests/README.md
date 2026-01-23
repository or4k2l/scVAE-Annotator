# Tests

This directory contains all tests for scVAE-Annotator.

## Test Structure

```
tests/
├── __init__.py
├── test_annotator.py       # Config and Pipeline Tests
├── test_model.py            # VAE Architecture Tests
├── test_preprocessing.py    # Preprocessing Function Tests
└── test_integration.py      # End-to-End Integration Tests
```

## Running Tests

### All Tests

```bash
pytest tests/
```

### With Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Quick Tests (without slow marker)

```bash
pytest tests/ -m "not slow"
```

### Specific Test File

```bash
pytest tests/test_annotator.py -v
```

### With Make

```bash
make test              # Standard Tests
make test-quick        # Quick Tests Only
make test-full         # All Tests with Coverage
```

### With run_tests.sh Script

```bash
./run_tests.sh              # Standard
./run_tests.sh --quick      # Quick Tests
./run_tests.sh --full       # Full Suite
./run_tests.sh --lint       # Linting Only
./run_tests.sh --fix        # Auto-fix Linting
```

## Test Categories

### Unit Tests (`test_annotator.py`, `test_model.py`)

Test individual functions and classes in isolation:

- Config initialization
- VAE architecture
- Model components

```bash
pytest tests/test_annotator.py tests/test_model.py -v
```

### Preprocessing Tests (`test_preprocessing.py`)

Test data processing pipeline:

- Cell and gene filtering
- Normalization
- Log transformation
- HVG selection
- PCA, UMAP, Clustering

```bash
pytest tests/test_preprocessing.py -v
```

### Integration Tests (`test_integration.py`)

Test the complete pipeline end-to-end:

- Full pipeline with synthetic data
- Output validation
- Error handling

⚠️ **Note**: Integration tests are marked as "slow" and are skipped by default.

```bash
pytest tests/test_integration.py -v -m "slow"
```

## Test Markers

Tests can be marked with:

- `@pytest.mark.slow` - Slow tests (>10 seconds)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests

### Using Markers

```bash
# Unit tests only
pytest -m unit

# Without slow tests
pytest -m "not slow"

# Integration tests only
pytest -m integration
```

## Code Coverage

### Generate Coverage Report

```bash
pytest --cov=. --cov-report=html
```

The report will be saved in `htmlcov/index.html`.

### Coverage in Terminal

```bash
pytest --cov=. --cov-report=term-missing
```

### Coverage Goal

Target Coverage: **≥ 80%**

View current coverage:

```bash
coverage report
```

## Fixtures

Available Pytest Fixtures (in `test_integration.py`):

### `temp_output_dir`

Temporary directory for test outputs:

```python
def test_example(temp_output_dir):
    # temp_output_dir is a temporary directory
    # automatically deleted after the test
    pass
```

### `synthetic_adata`

Synthetic AnnData for tests:

```python
def test_example(synthetic_adata):
    # synthetic_adata is an AnnData object with 200 cells, 100 genes
    assert synthetic_adata.n_obs == 200
    assert synthetic_adata.n_vars == 100
```

## Continuous Integration (CI)

### GitHub Actions

Tests run automatically on every push/PR:

- **tests.yml**: Runs on Ubuntu, macOS, Windows with Python 3.8-3.12
- **lint.yml**: Code quality checks (Black, Flake8, Pylint, etc.)
- **docs.yml**: Documentation validation

### Local CI Simulation

```bash
# Simulate GitHub Actions locally
./run_tests.sh --full
make lint
```

## Troubleshooting

### Import Errors

Ensure the package is installed:

```bash
pip install -r requirements.txt
```

### Timeout Errors

Increase timeout for slow tests:

```bash
pytest --timeout=60
```

### Memory Errors with Large Tests

Reduce parallel workers:

```bash
pytest -n 2  # Only 2 parallel processes
```

### Coverage Not Found

Install coverage tools:

```bash
pip install pytest-cov coverage[toml]
```

## Best Practices

### Writing New Tests

1. **File names**: `test_*.py`
2. **Function names**: `test_*`
3. **Docstrings**: Describe what is tested
4. **Assertions**: Use clear assertions
5. **Fixtures**: Use fixtures for setup/teardown

Example:

```python
def test_config_initialization():
    """Test that Config initializes with defaults."""
    config = Config()
    assert config.n_top_genes == 3000
    assert config.leiden_k_neighbors == 30
```

### Test Isolation

Each test should be independent:

- No dependencies between tests
- Clean up temporary files
- Use fixtures for setup

### Performance

- Mark slow tests with `@pytest.mark.slow`
- Use mock objects where possible
- Reduce data sizes in tests

## Additional Tools

### pytest-xdist (Parallel Tests)

```bash
pip install pytest-xdist
pytest -n auto  # Automatic CPU count
```

### pytest-timeout (Timeout Protection)

```bash
pip install pytest-timeout
pytest --timeout=30
```

### pytest-watch (Auto-Rerun)

```bash
pip install pytest-watch
ptw  # Runs tests automatically on file changes
```

## More Information

- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
