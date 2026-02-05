# Testing Guide - scVAE-Annotator

## 🎯 Test Strategy

This project follows a strict testing philosophy:
- **90%+ coverage** as the minimum goal for production code
- **mypy strict mode** with 100% type safety
- **pytest** as the primary test framework
- **Comprehensive testing**: Unit → Integration → End-to-End

## 📊 Current Status

| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| config.py | 97.78% | 14 ✅ | EXCELLENT |
| vae.py | 100% | 17 ✅ | PERFECT |
| preprocessing.py | 81% | 19 (mixed) | GOOD |
| clustering.py | 41.82% | 8 (mixed) | NEEDS_WORK |
| annotator.py | 11.88% | 0 | CRITICAL |
| pipeline.py | 9.18% | 0 | CRITICAL |
| visualization.py | 14.63% | 0 | CRITICAL |
| cli.py | 0% | 0 | CRITICAL |
| **TOTAL** | **31.10%** | 30 ✅ / 19 ❌ | IN_PROGRESS |

## 🚀 Quick Start

### Install test dependencies

```bash
pip install pytest pytest-cov pytest-mock mypy
```

### Run tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/scvae_annotator --cov-report=html --cov-report=term

# Only passing tests
pytest tests/test_config.py tests/test_vae.py -v

# Test a specific module
pytest tests/test_config.py -v --tb=short
```

### Type checking

```bash
# Entire project
mypy src/scvae_annotator

# Single module
mypy src/scvae_annotator/config.py
```

## 📝 Test Structure

### Fully implemented

#### 1. **test_config.py** (14 tests, 97.78% coverage)
```python
# Covered:
✅ Config dataclass validation
✅ Parameter constraints (batch_size, epochs, etc.)
✅ Random seed handling
✅ create_optimized_config() factory
✅ Edge cases (zero values, negative numbers)
✅ Type safety (int/float conversions)
```

**Example:**
```python
def test_config_creation():
    config = Config(
        batch_size=128,
        max_epochs=100,
        learning_rate=0.001
    )
    assert config.batch_size == 128
    assert config.max_epochs == 100
```

#### 2. **test_vae.py** (17 tests, 100% coverage)
```python
# Covered:
✅ EarlyStopping logic (patience, delta)
✅ ImprovedVAE forward/loss
✅ train_improved_vae() with different configs
✅ CUDA handling (automatic fallback to CPU)
✅ Loss computation and convergence
✅ Edge cases (empty training, single batch)
```

**Example:**
```python
def test_vae_training():
    adata = create_test_adata(n_obs=100, n_vars=50)
    trained_vae, losses = train_improved_vae(adata, config)
    assert len(losses) > 0
    assert all(loss >= 0 for loss in losses)
```

### Partially implemented

#### 3. **test_preprocessing.py** (19 tests, 81% coverage)
```python
# Covered:
✅ enhanced_preprocessing() basics
✅ discover_marker_genes()
⚠️ QC filtering often yields empty data
⚠️ Test fixtures need realistic metrics
```

**Issues:**
- Synthetic data does not survive QC filters
- `n_genes_by_counts` and `pct_counts_mt` are often missing
- Recommendation: more robust fixture generation

#### 4. **test_clustering.py** (8 tests, 41.82% coverage)
```python
# Covered:
✅ optimized_leiden_clustering() basics
⚠️ Missing: PCA/neighbors setup in fixtures
⚠️ Missing: ARI/Silhouette metrics
```

**Issues:**
- Test data lacks `.obsm['X_pca']`
- Clustering fails without a neighbors graph
- Recommendation: run `sc.pp.neighbors()` in fixtures

### Not yet implemented

#### 5. **test_annotator.py** (0% coverage - CRITICAL)
```python
# Needed:
❌ EnhancedAutoencoderAnnotator.__init__()
❌ train() with Optuna optimization
❌ predict() with confidence scores
❌ SMOTE handling for imbalanced data
❌ Calibration (Platt scaling)
❌ Edge cases (unknown labels, single class)
```

**Priority:** HIGH - Core functionality

#### 6. **test_pipeline.py** (0% coverage - CRITICAL)
```python
# Needed:
❌ run_annotation_pipeline() end-to-end
❌ evaluate_predictions() metrics
❌ analyze_optimization_results()
❌ save_results() file handling
❌ Integration: preprocessing → clustering → VAE → annotator
```

**Priority:** HIGH - Orchestration

#### 7. **test_visualization.py** (0% coverage)
```python
# Needed:
❌ create_visualizations() plot generation
❌ UMAP consistency
❌ Confidence plots
❌ File saving (PNG/PDF)
```

**Priority:** MEDIUM - Output

#### 8. **test_cli.py** (0% coverage)
```python
# Needed:
❌ main() argument parsing
❌ Command execution (--help, --version)
❌ File path validation
❌ Error handling
```

**Priority:** MEDIUM - User Interface

## 🔧 Test Fixtures Best Practices

### Robust AnnData generation

```python
@pytest.fixture
def realistic_adata():
    """Create AnnData with realistic QC metrics."""
    n_obs, n_vars = 200, 100
    X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars))
    
    adata = ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame({
            'n_genes_by_counts': np.random.randint(50, 500, n_obs),
            'total_counts': X.sum(axis=1),
            'pct_counts_mt': np.random.uniform(0, 15, n_obs),
            'cell_type': np.random.choice(['A', 'B', 'C'], n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)]),
        var=pd.DataFrame({
            'gene_ids': [f'GENE_{i}' for i in range(n_vars)],
            'n_cells_by_counts': np.random.randint(10, n_obs, n_vars),
            'highly_variable': np.random.choice([True, False], n_vars)
        }, index=[f'gene_{i}' for i in range(n_vars)])
    )
    
    # Preprocessing for tests
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=min(30, n_obs - 1, n_vars - 1))
    sc.pp.neighbors(adata)
    
    return adata
```

## 📈 Roadmap to 90%+ Coverage

### Phase 1: Critical modules (Week 1)
```
✅ config.py (97.78%)
✅ vae.py (100%)
🔄 annotator.py (11.88% → 90%+)
🔄 pipeline.py (9.18% → 90%+)
```

### Phase 2: Integration (Week 2)
```
🔄 preprocessing.py (81% → 90%+)
🔄 clustering.py (41.82% → 90%+)
🔄 visualization.py (14.63% → 90%+)
```

### Phase 3: User Interface (Week 3)
```
🔄 cli.py (0% → 90%+)
📝 Integration tests
📝 End-to-end tests
```

### Phase 4: Polish (Week 4)
```
📝 Performance tests
📝 Edge case hardening
📝 Documentation updates
🎉 90%+ coverage reached!
```

## 🛠️ Debugging failed tests

### Problem: "Empty data after filtering"

```python
# Root cause
def test_preprocessing():
    adata = ad.AnnData(X=np.random.rand(100, 50))  # ❌ Too simple
    result = enhanced_preprocessing(adata)  # Filters everything out!

# Fix
def test_preprocessing():
    adata = create_realistic_adata()  # ✅ With QC metrics
    result = enhanced_preprocessing(adata)
```

### Problem: "KeyError: 'X_pca'"

```python
# Root cause
def test_clustering():
    adata = ad.AnnData(X=np.random.rand(100, 50))
    optimized_leiden_clustering(adata)  # ❌ No PCA

# Fix
def test_clustering():
    adata = create_realistic_adata()  # ✅ With PCA/neighbors
    optimized_leiden_clustering(adata)
```

## 📊 Generate a coverage report

```bash
# Terminal report
pytest --cov=src/scvae_annotator --cov-report=term-missing

# HTML report (recommended!)
pytest --cov=src/scvae_annotator --cov-report=html
# Open: htmlcov/index.html

# XML for CI/CD
pytest --cov=src/scvae_annotator --cov-report=xml
```

## 🎯 CI/CD Integration

### GitHub Actions workflow (planned)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      
      - name: Type checking
        run: mypy src/scvae_annotator
      
      - name: Run tests
        run: pytest --cov --cov-fail-under=90
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 📚 Additional resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Plugin](https://pytest-cov.readthedocs.io/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [scanpy Testing Guide](https://scanpy.readthedocs.io/en/stable/dev/testing.html)

---

**Current:** 31.10% coverage | **Target:** 90%+ | **Status:** 🚧 In Progress
