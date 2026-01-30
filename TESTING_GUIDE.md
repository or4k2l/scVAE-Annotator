# Testing Guide - scVAE-Annotator

## 🎯 Test-Strategie

Dieses Projekt folgt einer strikten Test-Philosophie:
- **90%+ Coverage** als Minimalziel für Production-Code
- **mypy strict mode** mit 100% Type-Safety
- **pytest** als primäres Test-Framework
- **Comprehensive Testing**: Unit → Integration → End-to-End

## 📊 Aktueller Status

| Modul | Coverage | Tests | Status |
|-------|----------|-------|--------|
| config.py | 97.78% | 14 ✅ | EXCELLENT |
| vae.py | 100% | 17 ✅ | PERFECT |
| preprocessing.py | 81% | 19 (mixed) | GOOD |
| clustering.py | 41.82% | 8 (mixed) | NEEDS_WORK |
| annotator.py | 11.88% | 0 | CRITICAL |
| pipeline.py | 9.18% | 0 | CRITICAL |
| visualization.py | 14.63% | 0 | CRITICAL |
| cli.py | 0% | 0 | CRITICAL |
| **GESAMT** | **31.10%** | 30 ✅ / 19 ❌ | IN_PROGRESS |

## 🚀 Schnellstart

### Installation Test-Dependencies

```bash
pip install pytest pytest-cov pytest-mock mypy
```

### Tests ausführen

```bash
# Alle Tests
pytest tests/ -v

# Mit Coverage-Report
pytest tests/ --cov=src/scvae_annotator --cov-report=html --cov-report=term

# Nur erfolgreiche Tests
pytest tests/test_config.py tests/test_vae.py -v

# Bestimmtes Modul testen
pytest tests/test_config.py -v --tb=short
```

### Type-Checking

```bash
# Komplettes Projekt
mypy src/scvae_annotator

# Einzelnes Modul
mypy src/scvae_annotator/config.py
```

## 📝 Test-Struktur

### Erfolgreich implementiert

#### 1. **test_config.py** (14 Tests, 97.78% Coverage)
```python
# Abgedeckt:
✅ Config dataclass validation
✅ Parameter constraints (batch_size, epochs, etc.)
✅ Random seed handling
✅ create_optimized_config() factory
✅ Edge cases (zero values, negative numbers)
✅ Type safety (int/float conversions)
```

**Beispiel:**
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

#### 2. **test_vae.py** (17 Tests, 100% Coverage)
```python
# Abgedeckt:
✅ EarlyStopping logic (patience, delta)
✅ ImprovedVAE forward/loss
✅ train_improved_vae() mit verschiedenen Configs
✅ CUDA handling (automatisches Fallback zu CPU)
✅ Loss computation und convergence
✅ Edge cases (leeres Training, single batch)
```

**Beispiel:**
```python
def test_vae_training():
    adata = create_test_adata(n_obs=100, n_vars=50)
    trained_vae, losses = train_improved_vae(adata, config)
    assert len(losses) > 0
    assert all(loss >= 0 for loss in losses)
```

### Teilweise implementiert

#### 3. **test_preprocessing.py** (19 Tests, 81% Coverage)
```python
# Abgedeckt:
✅ enhanced_preprocessing() basis
✅ discover_marker_genes()
⚠️ QC-Filtering führt oft zu leeren Daten
⚠️ Test-Fixtures benötigen realistische Metriken
```

**Probleme:**
- Synthetic data überlebt QC-Filter nicht
- `n_genes_by_counts` und `pct_counts_mt` fehlen oft
- Empfehlung: Robustere Fixture-Generierung

#### 4. **test_clustering.py** (8 Tests, 41.82% Coverage)
```python
# Abgedeckt:
✅ optimized_leiden_clustering() Basis
⚠️ Fehlt: PCA/neighbors-Setup in Fixtures
⚠️ Fehlt: ARI/Silhouette-Metriken
```

**Probleme:**
- Test-Daten haben keine `.obsm['X_pca']`
- Clustering schlägt fehl ohne neighbors graph
- Empfehlung: `sc.pp.neighbors()` in Fixtures

### Noch nicht implementiert

#### 5. **test_annotator.py** (0% Coverage - KRITISCH)
```python
# Benötigt:
❌ EnhancedAutoencoderAnnotator.__init__()
❌ train() mit Optuna-Optimization
❌ predict() mit Confidence-Scores
❌ SMOTE-Handling für imbalanced data
❌ Calibration (Platt scaling)
❌ Edge cases (unknown labels, single class)
```

**Priorität:** HIGH - Core-Funktionalität

#### 6. **test_pipeline.py** (0% Coverage - KRITISCH)
```python
# Benötigt:
❌ run_annotation_pipeline() End-to-End
❌ evaluate_predictions() Metriken
❌ analyze_optimization_results()
❌ save_results() File-Handling
❌ Integration: preprocessing → clustering → VAE → annotator
```

**Priorität:** HIGH - Orchestrierung

#### 7. **test_visualization.py** (0% Coverage)
```python
# Benötigt:
❌ create_visualizations() Plot-Generierung
❌ UMAP consistency
❌ Confidence plots
❌ File-Saving (PNG/PDF)
```

**Priorität:** MEDIUM - Output

#### 8. **test_cli.py** (0% Coverage)
```python
# Benötigt:
❌ main() Argument-Parsing
❌ Command execution (--help, --version)
❌ File path validation
❌ Error handling
```

**Priorität:** MEDIUM - User-Interface

## 🔧 Test-Fixtures Best Practices

### Robuste AnnData-Generierung

```python
@pytest.fixture
def realistic_adata():
    """Erstellt AnnData mit realistischen QC-Metriken"""
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
    
    # Preprocessing für Tests
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=min(30, n_obs-1, n_vars-1))
    sc.pp.neighbors(adata)
    
    return adata
```

## 📈 Roadmap zu 90%+ Coverage

### Phase 1: Kritische Module (Woche 1)
```
✅ config.py (97.78%)
✅ vae.py (100%)
🔄 annotator.py (11.88% → 90%+)
🔄 pipeline.py (9.18% → 90%+)
```

### Phase 2: Integration (Woche 2)
```
🔄 preprocessing.py (81% → 90%+)
🔄 clustering.py (41.82% → 90%+)
🔄 visualization.py (14.63% → 90%+)
```

### Phase 3: User-Interface (Woche 3)
```
🔄 cli.py (0% → 90%+)
📝 Integration tests
📝 End-to-End tests
```

### Phase 4: Polish (Woche 4)
```
📝 Performance tests
📝 Edge case hardening
📝 Documentation updates
🎉 90%+ Coverage erreicht!
```

## 🛠️ Debugging fehlgeschlagener Tests

### Problem: "Empty data after filtering"

```python
# Fehlerursache
def test_preprocessing():
    adata = ad.AnnData(X=np.random.rand(100, 50))  # ❌ Zu simpel
    result = enhanced_preprocessing(adata)  # Filtert alles raus!

# Lösung
def test_preprocessing():
    adata = create_realistic_adata()  # ✅ Mit QC-Metriken
    result = enhanced_preprocessing(adata)
```

### Problem: "KeyError: 'X_pca'"

```python
# Fehlerursache
def test_clustering():
    adata = ad.AnnData(X=np.random.rand(100, 50))
    optimized_leiden_clustering(adata)  # ❌ Kein PCA

# Lösung
def test_clustering():
    adata = create_realistic_adata()  # ✅ Mit PCA/neighbors
    optimized_leiden_clustering(adata)
```

## 📊 Coverage-Report generieren

```bash
# Terminal-Report
pytest --cov=src/scvae_annotator --cov-report=term-missing

# HTML-Report (empfohlen!)
pytest --cov=src/scvae_annotator --cov-report=html
# Öffne: htmlcov/index.html

# XML für CI/CD
pytest --cov=src/scvae_annotator --cov-report=xml
```

## 🎯 CI/CD Integration

### GitHub Actions Workflow (geplant)

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

## 📚 Weitere Ressourcen

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Plugin](https://pytest-cov.readthedocs.io/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [scanpy Testing Guide](https://scanpy.readthedocs.io/en/stable/dev/testing.html)

---

**Stand:** 31.10% Coverage | **Ziel:** 90%+ | **Status:** 🚧 In Progress
