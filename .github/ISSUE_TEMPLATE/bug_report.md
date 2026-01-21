---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

---

## 🐛 Bug Description
A clear and concise description of what the bug is.

## 📋 Steps to Reproduce
Steps to reproduce the behavior:
1. Load data '...'
2. Run pipeline with '...'
3. See error

## ✅ Expected Behavior
A clear and concise description of what you expected to happen.

## ❌ Actual Behavior
A clear and concise description of what actually happened.

## 📊 Data Information
- **Dataset size**: [e.g., 10,000 cells, 2,000 genes]
- **Data format**: [e.g., .h5ad, .csv]
- **Cell types**: [e.g., PBMC, tissue-specific]

## 🖥️ Environment
- **OS**: [e.g., Ubuntu 22.04, macOS 13, Windows 11]
- **Python version**: [e.g., 3.10.5]
- **scVAE-Annotator version**: [e.g., 0.1.0]
- **Installation method**: [e.g., pip, git clone]

## 📝 Error Log
```
Paste the full error message here
```

## 🔧 Configuration
```python
# Paste your Config or relevant code here
config = Config(
    target_genes=2000,
    ...
)
```

## 📸 Screenshots
If applicable, add screenshots to help explain your problem.

## 🔄 Reproducible Example
If possible, provide a minimal reproducible example:
```python
import scanpy as sc
from scvae_annotator import Config, run_annotation_pipeline

# Your code here
```

## ℹ️ Additional Context
Add any other context about the problem here.

## ✔️ Checklist
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided all required information
- [ ] I have included a reproducible example
- [ ] I have checked the documentation
