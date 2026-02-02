# @title 🏆 scVAE-Annotator: Complete Pipeline (PBMC 10k) 
# ==========================================
# 0. IMPORTS & SETUP
# ==========================================
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scanpy as sc
import anndata
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0 

print("🚀 Starting complete pipeline...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Hardware: {device}")

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
print("\n[1/5] Loading PBMC 10k dataset...")

class Config:
    def __init__(self, input_dim=2000, mode='poisson'):
        self.input_dim = input_dim
        self.latent_dim = 10
        self.hidden_dim = 128
        self.learning_rate = 5e-4  # Conservative LR for stability
        self.epochs = 35
        self.batch_size = 128
        self.dropout_rate = 0.1
        self.likelihood_type = mode
        self.warmup_epochs = 10


def load_data(n_top_genes=2000):
    filename = "pbmc_10k_v3_filtered_feature_bc_matrix.h5"
    if not os.path.exists(filename):
        print("   Downloading dataset (~35 MB)...")
        os.system("wget -q https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5")
    
    adata = sc.read_10x_h5(filename)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Backup raw counts (crucial for Poisson loss)
    adata.layers["counts"] = adata.X.copy()
    
    # Log-normalization (for MSE baseline & HVG selection)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    
    return adata

adata = load_data()
print(f"   Data loaded: {adata.n_obs} cells x {adata.n_vars} genes")

# Prepare data arrays
X_mse = adata.X.toarray()
X_pois = adata.layers["counts"].toarray()
indices = np.arange(adata.n_obs)
train_idx, _ = train_test_split(indices, test_size=0.1, random_state=42)

def get_loader(data, batch_size):
    # Ensure Float32 data type
    return DataLoader(TensorDataset(torch.from_numpy(data.astype(np.float32))), 
                      batch_size=batch_size, shuffle=True, num_workers=0)

# ==========================================
# 2. MODEL DEFINITION (Bulletproof VAE)
# ==========================================
class ScVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.input_dim)
        )

    def reparameterize(self, mu, logvar):
        # STABILITY CLAMP 1: Limit variance to prevent explosion
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar # Returns raw logits

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def loss_function(recon_x, x, mu, logvar, config, epoch):
    if config.likelihood_type == 'mse':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    else: # Poisson
        # STABILITY CLAMP 2: Limit output logits for safe exp() calculation
        # exp(15) is approx 3.2 million counts, sufficient for RNA-seq
        recon_x = torch.clamp(recon_x, min=-10, max=15)
        recon_loss = F.poisson_nll_loss(recon_x, x, log_input=True, full=False, reduction='sum')

    # KL Divergence with safety clamps
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # KL Warm-up
    beta = min(1.0, epoch / config.warmup_epochs)
    return recon_loss + beta * kl_loss

def train(mode_name, data_loader, raw_data):
    print(f"\n[Training] Starting {mode_name.upper()} model...")
    cfg = Config(input_dim=adata.n_vars, mode=mode_name)
    model = ScVAE(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0
        for x_batch, in data_loader:
            x_batch = x_batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x_batch)
            loss = loss_function(recon, x_batch, mu, logvar, cfg, epoch)
            
            if torch.isnan(loss):
                print("   ⚠️ NaN Warning (batch ignored)")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{cfg.epochs} Loss: {total_loss/len(data_loader.dataset):.1f}")
            
    # Extract Latents
    model.eval()
    latents = []
    full_loader = DataLoader(TensorDataset(torch.from_numpy(raw_data.astype(np.float32))), batch_size=512)
    with torch.no_grad():
        for x_batch, in full_loader:
            h = model.enc(x_batch.to(device))
            latents.append(model.mu(h).cpu().numpy())
    return np.concatenate(latents)

# Train both models
z_mse = train('mse', get_loader(X_mse[train_idx], 128), X_mse)
z_pois = train('poisson', get_loader(X_pois[train_idx], 128), X_pois)

# Save to AnnData
adata.obsm['X_vae_mse'] = z_mse
adata.obsm['X_vae_pois'] = z_pois

# ==========================================
# 4. CLUSTERING & AUTO-ANNOTATION
# ==========================================
print("\n[4/5] Computing neighbors & annotation...")

# Compute UMAPs
for key in ['mse', 'pois']:
    sc.pp.neighbors(adata, use_rep=f'X_vae_{key}', key_added=key)
    sc.tl.umap(adata, neighbors_key=key)
    adata.obsm[f'X_umap_{key}'] = adata.obsm['X_umap'].copy()

# Clustering based on Poisson (scientifically more robust)
sc.tl.leiden(adata, neighbors_key='pois', key_added='leiden', resolution=0.4)

# Marker Gene Definition
marker_genes_raw = {
    'CD4+ T-Cells': ['IL7R', 'CD3D', 'CD3E', 'LTB'],
    'CD8+ T-Cells': ['CD8A', 'CD8B', 'GZMK'],
    'B-Cells':      ['MS4A1', 'CD79A', 'CD79B'],
    'NK Cells':     ['GNLY', 'NKG7', 'KLRB1'],
    'CD14+ Mono':   ['CD14', 'LYZ', 'FCN1'],
    'FCGR3A+ Mono': ['FCGR3A', 'MS4A7'],
    'Dendritic':    ['FCER1A', 'CST3', 'CD1C'],
    'Platelets':    ['PPBP', 'PF4'],
}

# 1. Filter genes (Fix for missing genes in reduced datasets)
marker_genes = {k: [m for m in v if m in adata.var_names] for k,v in marker_genes_raw.items()}
marker_genes = {k: v for k,v in marker_genes.items() if v} # Remove empty keys

# 2. Scoring & Voting
for ct, markers in marker_genes.items():
    sc.tl.score_genes(adata, markers, score_name=ct)

score_keys = list(marker_genes.keys())
adata.obs['predicted'] = adata.obs[score_keys].idxmax(axis=1)

# Majority Vote per Cluster
cluster_map = {}
for cl in adata.obs['leiden'].unique():
    top_type = adata.obs[adata.obs['leiden'] == cl]['predicted'].mode()[0]
    cluster_map[cl] = top_type
adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_map)

print("   Annotation complete. Clusters named.")

# ==========================================
# 5. FINAL PLOTTING
# ==========================================
print("\n[5/5] Generating plots...")
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 2)

# Plot 1: MSE (Top Left)
ax1 = fig.add_subplot(gs[0, 0])
sc.pl.embedding(adata, basis='umap_mse', color='cell_type', ax=ax1, show=False, 
               title='Standard VAE (MSE)', frameon=False, legend_loc='none')

# Plot 2: Poisson (Top Right)
ax2 = fig.add_subplot(gs[0, 1])
sc.pl.embedding(adata, basis='umap_pois', color='cell_type', ax=ax2, show=False, 
               title='Scientific Upgrade (Poisson)', frameon=False)

# Plot 3: Dotplot Validation (Bottom, spanning both columns)
ax3 = fig.add_subplot(gs[1, :])
sc.pl.dotplot(adata, marker_genes, groupby='cell_type', standard_scale='var', 
              ax=ax3, show=False, title='Biological Validation (Marker Expression)')

plt.tight_layout()
output_path = os.path.join('..', 'figures', 'poisson_vs_mse_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"   Plot saved to {output_path}")
plt.show()
print("✅ Done! Analysis complete.")
