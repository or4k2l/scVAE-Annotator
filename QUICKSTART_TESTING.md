# Quick Start - Testing scVAE-Annotator

## 🚀 5-Minute Setup

### 1. Install Dependencies
```bash
pip install pytest pytest-cov pytest-mock mypy
```

### 2. Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov

# Only passing tests
pytest tests/test_config.py tests/test_vae.py
```

### 3. Check Types
```bash
mypy src/scvae_annotator
```

---

## ✅ Current Test Status

### Passing Tests (30/58)
```bash
# Config tests (14/14) ✅
pytest tests/test_config.py -v
# Result: 97.78% coverage

# VAE tests (16/17) ✅
pytest tests/test_vae.py -v
# Result: 100% coverage (1 skipped for CUDA)
```

### Mixed Results (28/58)
```bash
# Preprocessing tests (19 tests, mixed)
pytest tests/test_preprocessing.py -v

# Clustering tests (8 tests, mixed)
pytest tests/test_clustering.py -v
```

---

## 📊 Quick Coverage Check

```bash
# Terminal report
pytest --cov=src/scvae_annotator --cov-report=term

# HTML report (recommended!)
pytest --cov=src/scvae_annotator --cov-report=html
open htmlcov/index.html
```

**Current Coverage:** 31.10%  
**Target:** 90%+

---

## 🔧 Common Commands

### Testing
```bash
# Run specific test
pytest tests/test_config.py::test_config_creation -v

# Run with verbose output
pytest -vv

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run only fast tests
pytest -m "not slow"
```

### Coverage
```bash
# Coverage with missing lines
pytest --cov --cov-report=term-missing

# Multiple report formats
pytest --cov --cov-report=html --cov-report=xml

# Coverage for specific module
pytest --cov=src/scvae_annotator/config
```

### Type Checking
```bash
# All modules
mypy src/scvae_annotator

# Specific module
mypy src/scvae_annotator/config.py

# Show error codes
mypy --show-error-codes src/scvae_annotator
```

---

## 🎯 What to Test Next

### Priority 1: Fix Failing Tests
```bash
# Identify failures
pytest tests/test_preprocessing.py -v
pytest tests/test_clustering.py -v

# Fix: Update fixtures with realistic data
# See: TESTING_GUIDE.md for examples
```

### Priority 2: Critical Modules
```bash
# Create missing test files:
touch tests/test_annotator.py    # 11.88% → 90%+
touch tests/test_pipeline.py     # 9.18% → 90%+
touch tests/test_cli.py          # 0% → 90%+
touch tests/test_visualization.py # 14.63% → 90%+
```

---

## 📚 Documentation

- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Complete testing reference
- [COVERAGE_REPORT.md](COVERAGE_REPORT.md) - Detailed coverage analysis
- [REFACTORING_FINAL_REPORT.md](REFACTORING_FINAL_REPORT.md) - Transformation summary

---

## ❓ Troubleshooting

### "ImportError: No module named scvae_annotator"
```bash
# Solution: Install package in editable mode
pip install -e .
```

### "pytest: command not found"
```bash
# Solution: Install pytest
pip install pytest pytest-cov pytest-mock
```

### "mypy: No module named 'scanpy'"
```bash
# Expected: Third-party libraries ignored in pyproject.toml
# Check: pyproject.toml [tool.mypy.overrides]
```

### Tests fail with "Empty data after filtering"
```bash
# Solution: Use realistic test fixtures
# See: TESTING_GUIDE.md section "Robuste AnnData-Generierung"
```

---

## 🎓 Best Practices

1. **Run tests before committing**
   ```bash
   pytest && mypy src/scvae_annotator
   ```

2. **Check coverage regularly**
   ```bash
   pytest --cov --cov-report=term-missing
   ```

3. **Write tests for new code**
   - Aim for 90%+ coverage on new modules
   - Test happy path and edge cases
   - Use fixtures for reusable test data

4. **Keep tests fast**
   - Mark slow tests with `@pytest.mark.slow`
   - Run fast tests during development
   - Run full suite before pushing

---

## 📊 Progress Tracking

### Completed ✅
- [x] test_config.py (14 tests, 97.78% coverage)
- [x] test_vae.py (17 tests, 100% coverage)
- [x] mypy strict mode (100% type coverage)
- [x] pytest configuration

### In Progress 🔄
- [ ] Fix failing tests (19 failures)
- [ ] test_annotator.py (0 → 90%+ coverage)
- [ ] test_pipeline.py (0 → 90%+ coverage)

### Planned 📅
- [ ] test_cli.py (0 → 90%+ coverage)
- [ ] test_visualization.py (0 → 90%+ coverage)
- [ ] CI/CD setup (GitHub Actions)

---

## 🚀 Quick Win Commands

```bash
# Show only passing tests
pytest tests/test_config.py tests/test_vae.py -v

# Generate coverage badge
pytest --cov --cov-report=term | grep "TOTAL"

# Type check all modules
mypy src/scvae_annotator --strict

# Install and test
pip install -e . && pytest --cov
```

---

**Last Updated:** January 2025  
**Status:** 31.10% → 90%+ (in progress)  
**Next Steps:** Fix failing tests, add critical module tests
