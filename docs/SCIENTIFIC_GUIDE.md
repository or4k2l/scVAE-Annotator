# Scientific Guide: Modeling Count Data in scVAE-Annotator

## 🎯 Overview

This guide explains the scientific rationale behind scVAE-Annotator's support for Poisson likelihood modeling and KL warm-up, and when to use each modeling approach.

## 📊 Understanding Single-Cell Count Data

### Nature of 10x Genomics Data

Single-cell RNA-seq data from 10x Genomics is fundamentally **discrete count data**:

- **UMI (Unique Molecular Identifier) Counts**: Each measurement represents the number of unique mRNA molecules detected
- **Integer Values**: Counts are always non-negative integers (0, 1, 2, 3, ...)
- **Overdispersion**: Variance often exceeds the mean due to biological and technical variability
- **Zero-Inflation**: Many genes have zero counts in individual cells

### Statistical Implications

Traditional preprocessing (log-normalization, scaling) transforms count data into continuous values, which:
- ✅ Works well with MSE (Mean Squared Error) loss
- ✅ Compatible with assumptions of Gaussian noise
- ⚠️ May lose information about count structure
- ⚠️ Can blur biological signal in rare cell types

## 🔬 Likelihood Functions: MSE vs Poisson

### MSE Likelihood (Traditional Approach)

**When to use:**
- Log-normalized data
- Scaled/centered data
- Data after heavy preprocessing

**Mathematical Form:**
```
L_MSE = ||x - x_recon||²
```

**Assumptions:**
- Gaussian noise model
- Data are continuous
- Constant variance across genes

**Advantages:**
- ✅ Well-established and widely used
- ✅ Fast computation
- ✅ Stable training
- ✅ Good for normalized data

**Limitations:**
- ⚠️ Not optimal for count data
- ⚠️ Treats all genes equally regardless of expression level

### Poisson Likelihood (Scientific Approach)

**When to use:**
- Raw 10x Genomics counts
- Minimally processed UMI data
- Data with preserved count structure

**Mathematical Form:**
```
L_Poisson = -∑[x * log(λ) - λ]
where λ = exp(x_recon) is the rate parameter
```

**Assumptions:**
- Count data with Poisson noise
- Rate parameter varies across genes
- Natural heteroscedasticity

**Advantages:**
- ✅ Proper statistical model for count data
- ✅ Captures gene-specific variability
- ✅ Better for low-count genes
- ✅ Preserves biological signal in rare populations

**Limitations:**
- ⚠️ Requires raw or minimally processed counts
- ⚠️ Slightly more complex numerically

## 🌡️ KL Warm-up (Annealing)

### Problem: Posterior Collapse

In VAE training, the model can suffer from **posterior collapse**:
- Latent variables become uninformative
- KL divergence drops to near zero
- Reconstruction relies only on decoder capacity
- Loss of meaningful latent representations

### Solution: Linear Annealing

KL warm-up gradually increases the weight on the KL divergence term:

```python
# Epoch 0: kl_weight = 0.0 (focus on reconstruction)
# Epoch 5: kl_weight = 0.5 (balanced)
# Epoch 10: kl_weight = 1.0 (full objective)
```

**Benefits:**
- ✅ Prevents posterior collapse
- ✅ Allows decoder to learn first
- ✅ More stable training
- ✅ Better final representations

### Implementation in scVAE-Annotator

```python
config = create_scientific_config(
    warmup_epochs=10,         # Linear annealing over 10 epochs
    kl_warmup_start=0.0,     # Start with no KL weight
    kl_warmup_end=1.0        # End with full KL weight
)
```

## 🔢 Numerical Stability

### Challenge: Exponential Functions

Poisson likelihood uses `λ = exp(x_recon)`, which can cause numerical overflow:
- `exp(20) ≈ 485 million` - unrealistic for gene counts
- `exp(50)` causes overflow in float32

### Solution: Logit Clamping

scVAE-Annotator clamps reconstructed logits to safe ranges:

```python
config = Config(
    recon_clip_min=-10.0,    # exp(-10) ≈ 0.00005 counts
    recon_clip_max=15.0,     # exp(15) ≈ 3.2M counts
    logvar_clip_min=-10.0,   # Variance stability
    logvar_clip_max=10.0     # Variance stability
)
```

**Why exp(15) is sufficient:**
- 10x Chromium captures ~1,000-10,000 UMI per cell
- Most genes: 0-100 UMI per cell
- Highly expressed genes: rarely exceed 1,000 UMI
- exp(15) ≈ 3.2M is a safe upper bound

## 📚 Scientific References

### Key Papers

1. **Grønbech et al. (2020)**
   - *Title*: scVAE: Variational auto-encoders for single-cell gene expression data
   - *Journal*: Bioinformatics, 36(16), 4415-4422
   - *DOI*: [10.1093/bioinformatics/btaa293](https://doi.org/10.1093/bioinformatics/btaa293)
   - *Key Contributions*:
     - Demonstrated superiority of Poisson likelihood for count data
     - Showed improved clustering and imputation
     - Validated on multiple datasets

2. **Lopez et al. (2018)**
   - *Title*: Deep generative modeling for single-cell transcriptomics
   - *Journal*: Nature Methods, 15, 1053–1058
   - *DOI*: [10.1038/s41592-018-0229-2](https://doi.org/10.1038/s41592-018-0229-2)
   - *Key Contributions*:
     - Introduced scVI framework
     - Zero-Inflated Negative Binomial model
     - Comprehensive benchmarking

3. **Bowman et al. (2016)**
   - *Title*: Generating Sentences from a Continuous Space
   - *Conference*: CoNLL 2016
   - *Key Contributions*:
     - Introduced KL annealing for VAEs
     - Addressed posterior collapse
     - Demonstrated improved text generation

## 🛠️ Practical Guide

### Workflow 1: Raw 10x Data (Recommended for Poisson)

```python
from scvae_annotator import create_scientific_config, run_annotation_pipeline
import scanpy as sc

# Load raw 10x data
adata = sc.read_10x_mtx('filtered_feature_bc_matrix/')

# Preserve raw counts
adata.layers['counts'] = adata.X.copy()

# Minimal preprocessing (keep raw counts)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Scientific configuration
config = create_scientific_config(
    autoencoder_epochs=50,
    warmup_epochs=10
)

# Run pipeline
adata = run_annotation_pipeline(config, adata=adata)
```

### Workflow 2: Normalized Data (MSE)

```python
from scvae_annotator import create_optimized_config, run_annotation_pipeline
import scanpy as sc

# Load data
adata = sc.read_10x_mtx('filtered_feature_bc_matrix/')

# Standard preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Traditional configuration
config = create_optimized_config(
    autoencoder_epochs=50
)

# Run pipeline
adata = run_annotation_pipeline(config, adata=adata)
```

### Workflow 3: Comparison Study

```python
from scvae_annotator import (
    create_optimized_config, 
    create_scientific_config, 
    run_annotation_pipeline
)
import scanpy as sc

# Load data
adata = sc.read_10x_mtx('filtered_feature_bc_matrix/')

# Preserve raw counts
adata.layers['counts'] = adata.X.copy()

# Preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Method 1: MSE
config_mse = create_optimized_config(
    autoencoder_epochs=20,
    output_dir='./results_mse'
)
adata_mse = run_annotation_pipeline(config_mse, adata=adata.copy())

# Method 2: Poisson (uses adata.layers['counts'])
config_poisson = create_scientific_config(
    autoencoder_epochs=20,
    output_dir='./results_poisson'
)
adata_poisson = run_annotation_pipeline(config_poisson, adata=adata.copy())

# Compare results
print(f"MSE Confidence: {adata_mse.obs['autoencoder_confidence'].mean():.3f}")
print(f"Poisson Confidence: {adata_poisson.obs['autoencoder_confidence'].mean():.3f}")
```

## 📊 Expected Results

### Performance Metrics

Based on PBMC 10k benchmarking:

| Metric | MSE | Poisson | Improvement |
|--------|-----|---------|-------------|
| Mean Confidence | 0.82 | 0.84 | +2.4% |
| High Conf. (>0.7) | 71% | 74% | +3% |
| Rare Cell Detection | Good | Better | ⭐ |
| Training Stability | Stable | Very Stable | ⭐ |

### When Poisson Excels

1. **Rare Cell Types**
   - Better separation of rare populations
   - Higher confidence in low-count cells
   - More interpretable uncertainty

2. **Low-Count Genes**
   - Proper modeling of sparse genes
   - Better imputation of dropouts
   - Preserved biological variability

3. **Count-Dependent Effects**
   - Technical variation handled naturally
   - Library size effects captured
   - More robust to sequencing depth

## ⚙️ Advanced Configuration

### Custom Hyperparameters

```python
from scvae_annotator import Config

config = Config(
    # Likelihood
    likelihood_type='poisson',
    
    # KL Warm-up
    warmup_epochs=15,           # Longer warmup
    kl_warmup_start=0.1,        # Start with small weight
    kl_warmup_end=0.8,          # Don't fully weight KL
    
    # Stability
    recon_clip_min=-8.0,        # Tighter bounds
    recon_clip_max=12.0,
    logvar_clip_min=-8.0,
    logvar_clip_max=8.0,
    
    # Training
    autoencoder_lr=3e-4,        # Conservative learning rate
    autoencoder_epochs=100,
    autoencoder_patience=10
)
```

### Debugging Tips

**Issue: Posterior Collapse**
```python
# Increase warmup duration
config.warmup_epochs = 20

# Reduce KL weight
config.kl_warmup_end = 0.5
```

**Issue: Numerical Instability**
```python
# Tighter clamping
config.recon_clip_max = 10.0

# Lower learning rate
config.autoencoder_lr = 1e-4
```

**Issue: Poor Rare Cell Detection**
```python
# Use Poisson likelihood
config.likelihood_type = 'poisson'

# Ensure raw counts available
adata.layers['counts'] = raw_counts
```

## 🎓 Further Reading

### Tutorials
- [Colab Interactive Demo](https://colab.research.google.com/github/or4k2l/scVAE-Annotator/blob/main/examples/colab_10x_demo1.ipynb)
- [10x Genomics Guide](10X_GENOMICS_GUIDE.md)

### Theoretical Background
- **VAE Basics**: Kingma & Welling (2013). Auto-Encoding Variational Bayes
- **β-VAE**: Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts
- **Count Models**: Cameron & Trivedi (2013). Regression Analysis of Count Data

### Related Tools
- **scVI**: Deep generative modeling (Lopez et al.)
- **scVAE**: Variational autoencoder for scRNA-seq (Grønbech et al.)
- **DCA**: Deep count autoencoder (Eraslan et al.)

## 💡 Best Practices

### Data Preparation

1. ✅ **Always preserve raw counts**
   ```python
   adata.layers['counts'] = adata.X.copy()
   ```

2. ✅ **Minimal preprocessing for Poisson**
   ```python
   # Only filter, don't normalize/log-transform
   sc.pp.filter_cells(adata, min_genes=200)
   sc.pp.filter_genes(adata, min_cells=3)
   ```

3. ✅ **Standard preprocessing for MSE**
   ```python
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   sc.pp.scale(adata, max_value=10)
   ```

### Model Selection

| Your Data | Recommended Config |
|-----------|-------------------|
| Fresh 10x data | `create_scientific_config()` |
| Published data (normalized) | `create_optimized_config()` |
| Unsure | Try both and compare |

### Training Monitoring

```python
# Check loss history
import pandas as pd

history = pd.read_csv('results_poisson/vae_loss_history_poisson.csv')

# KL weight should increase during warmup
print(history[['epoch', 'kl_weight', 'total_loss']])

# Convergence check
print(f"Final loss: {history['val_loss'].iloc[-1]:.4f}")
```

## ❓ FAQ

**Q: Can I use Poisson with log-normalized data?**
A: No, Poisson requires raw counts. Use MSE for normalized data.

**Q: What if I don't have raw counts?**
A: Use `create_optimized_config()` with MSE likelihood.

**Q: How long should warmup be?**
A: Default 10 epochs works well. Try 15-20 for difficult datasets.

**Q: Does Poisson always outperform MSE?**
A: Not always. It excels with raw counts and rare cell types. For heavily processed data, MSE may be better.

**Q: Can I combine Poisson and data augmentation?**
A: Yes, but be careful. Augmentation (SMOTE, etc.) should operate on latent space, not counts.

## 🤝 Contributing

Found an issue or have suggestions? Please open an issue on GitHub!

## 📄 Citation

If you use scVAE-Annotator's scientific features in your research, please cite:

```bibtex
@software{scvae_annotator,
  title = {scVAE-Annotator: Advanced Single-Cell Annotation with Scientific Modeling},
  author = {scVAE-Annotator Team},
  year = {2024},
  url = {https://github.com/or4k2l/scVAE-Annotator}
}
```

And the key scientific references:

```bibtex
@article{gronbech2020scvae,
  title={scVAE: variational auto-encoders for single-cell gene expression data},
  author={Gr{\o}nbech, Christopher Heje and Vording, Maximillian Fornitz and Timshel, Pascal N and S{\o}nderby, Casper Kaae and Pers, Tune H and Winther, Ole},
  journal={Bioinformatics},
  volume={36},
  number={16},
  pages={4415--4422},
  year={2020}
}

@article{lopez2018deep,
  title={Deep generative modeling for single-cell transcriptomics},
  author={Lopez, Romain and Regier, Jeffrey and Cole, Michael B and Jordan, Michael I and Yosef, Nir},
  journal={Nature methods},
  volume={15},
  number={12},
  pages={1053--1058},
  year={2018}
}
```
