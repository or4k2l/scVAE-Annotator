#!/usr/bin/env python3
"""
scVAE-Annotator: Advanced Single-Cell RNA-seq Annotation Pipeline
===================================================================

An optimized pipeline for automated cell type annotation using:
- Variational Autoencoder (VAE) with early stopping
- Leiden clustering with adaptive metrics
- Hyperparameter optimization with Optuna
- Calibrated confidence scoring

Author: Your Name
License: See LICENSE file
"""

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           cohen_kappa_score, classification_report, confusion_matrix,
                           silhouette_score, adjusted_rand_score)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, kl_divergence
import scanpy.external as sce
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import logging
from pathlib import Path
import json
import optuna
from optuna.samplers import TPESampler

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class Config:
    """Configuration class for the annotation pipeline."""
    # Clustering parameters
    leiden_resolution_range: Tuple[float, float] = (0.005, 0.1)
    leiden_resolution_steps: int = 10
    leiden_k_neighbors: int = 30

    # Autoencoder parameters
    autoencoder_embedding_dim: int = 32
    autoencoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 64
    autoencoder_lr: float = 0.001
    autoencoder_dropout: float = 0.2
    autoencoder_patience: int = 7

    # Classification parameters
    confidence_threshold: float = 0.7
    adaptive_quantile: float = 0.05
    n_jobs: int = -1
    use_smote: bool = True
    cross_validation_folds: int = 5
    use_hyperparameter_optimization: bool = True
    optuna_trials: int = 50
    subsample_optuna_train: Optional[int] = 5000

    # Data preprocessing
    n_top_genes: int = 3000
    min_genes_per_cell: int = 200
    max_mt_percent: float = 15
    min_ground_truth_ratio: float = 0.8

    # Marker genes
    marker_genes: List[str] = field(default_factory=lambda: [
        'CD3E', 'CD7', 'CD2', 'LEF1', 'TCF7',
        'CD4', 'IL7R', 'CCR7',
        'CD8A', 'CD8B',
        'NKG7', 'GNLY', 'KLRF1',
        'CD19', 'MS4A1', 'CD79A', 'CD79B',
        'CD14', 'FCGR3A', 'MS4A7',
        'FCER1A', 'CPA3', 'IRF7',
        'PPBP', 'PF4',
        'CD34', 'KIT'
    ])

    # Output paths
    output_dir: str = './results'
    random_state: int = 42

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def discover_marker_genes(adata: ad.AnnData, config: Config) -> List[str]:
    """Automatically discover marker genes if ground truth is available."""
    if 'cell_type_ground_truth' not in adata.obs:
        logger.info("No ground truth available, using predefined marker genes")
        return config.marker_genes

    valid_cells = adata.obs['cell_type_ground_truth'].dropna()
    if len(valid_cells) < 100:
        logger.info("Too few labeled cells for marker discovery, using predefined markers")
        return config.marker_genes

    logger.info("Discovering marker genes from ground truth...")

    subset_adata = adata[valid_cells.index].copy()
    subset_adata.obs['celltype'] = valid_cells

    sc.tl.rank_genes_groups(subset_adata, 'celltype', method='wilcoxon', n_genes=10)

    discovered_markers = []
    for celltype in subset_adata.obs['celltype'].unique():
        markers = sc.get.rank_genes_groups_df(subset_adata, group=celltype)['names'].head(5).tolist()
        discovered_markers.extend(markers)

    all_markers = list(set(config.marker_genes + discovered_markers))
    logger.info(f"Using {len(all_markers)} marker genes ({len(discovered_markers)} discovered)")

    return all_markers


def download_data():
    """Download required data files with error handling."""
    urls_and_paths = [
        ('https://cf.10xgenomics.com/samples/cell-arc/1.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5',
         './data/10x-Multiome-Pbmc10k-RNA.h5'),
        ('https://www.dropbox.com/s/3g63m832mbeec4s/PBMC10k_multiome_vPBMCatlas.zip?dl=1',
         './data/PBMC10k_multiome_vPBMCatlas.zip')
    ]

    Path('./data').mkdir(parents=True, exist_ok=True)

    for url, path in urls_and_paths:
        if not os.path.exists(path):
            logger.info(f"Downloading {url}")
            os.system(f'wget -O {path} "{url}"')

    if not os.path.exists('./data/pbmc10k_annotations.csv'):
        os.system('unzip -o ./data/PBMC10k_multiome_vPBMCatlas.zip -d ./data/')
        os.system('mv ./data/PBMC10k_multiome_vPBMCatlas/Seurat_RNA_annotation.csv ./data/pbmc10k_annotations.csv')


def load_and_prepare_data(data_path: str = './data/10x-Multiome-Pbmc10k-RNA.h5') -> ad.AnnData:
    """Load data with comprehensive error handling."""
    try:
        adata = sc.read_10x_h5(data_path)
        adata.var_names_make_unique()
        logger.info(f"Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
        return adata
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def enhanced_preprocessing(adata: ad.AnnData, config: Config) -> ad.AnnData:
    """Enhanced preprocessing with adaptive marker genes."""
    logger.info("Starting preprocessing...")

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)

    n_cells_before = adata.shape[0]
    adata = adata[adata.obs.n_genes_by_counts > config.min_genes_per_cell]
    adata = adata[adata.obs.pct_counts_mt < config.max_mt_percent]
    logger.info(f"Filtered {n_cells_before - adata.shape[0]} low-quality cells")

    sc.pp.filter_genes(adata, min_cells=3)
    adata.raw = adata

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5,
                                n_top_genes=config.n_top_genes)

    marker_genes = discover_marker_genes(adata, config)
    marker_mask = adata.var_names.isin(marker_genes)
    adata.var['highly_variable'] = adata.var['highly_variable'] | marker_mask
    logger.info(f"Added {marker_mask.sum()} marker genes to highly variable genes")

    adata = adata[:, adata.var.highly_variable].copy()

    if adata.X.toarray().sum() / (adata.shape[0] * adata.shape[1]) < 0.05:
        logger.info("Data is very sparse, applying imputation...")
        imputer = KNNImputer(n_neighbors=5)
        adata.X = imputer.fit_transform(adata.X.toarray() if issparse(adata.X) else adata.X)

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, random_state=config.random_state)

    if 'batch' not in adata.obs.columns:
        adata.obs['batch'] = 'batch1'

    try:
        sce.pp.harmony_integrate(adata, 'batch')
        logger.info("Applied Harmony batch correction")
    except Exception as e:
        logger.warning(f"Harmony failed: {e}. Continuing without batch correction.")

    sc.pp.neighbors(adata, n_neighbors=config.leiden_k_neighbors, n_pcs=50, random_state=config.random_state)

    logger.info(f"Preprocessing complete. Final shape: {adata.shape}")
    return adata


def optimized_leiden_clustering(adata: ad.AnnData, config: Config) -> Tuple[ad.AnnData, int]:
    """Optimized Leiden clustering with adaptive metrics."""
    logger.info("Starting Leiden clustering optimization...")

    resolutions = np.linspace(*config.leiden_resolution_range, config.leiden_resolution_steps)
    best_res, best_score = config.leiden_resolution_range[0], -1
    best_n_clusters = 0

    use_ari = False
    if 'cell_type_ground_truth' in adata.obs:
        labeled_ratio = adata.obs['cell_type_ground_truth'].dropna().shape[0] / adata.shape[0]
        use_ari = labeled_ratio >= config.min_ground_truth_ratio
        logger.info(f"Ground truth coverage: {labeled_ratio:.1%}, using ARI: {use_ari}")

    metrics_history = []

    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, random_state=config.random_state)
        n_clusters = len(adata.obs['leiden'].unique())

        if n_clusters > 1:
            try:
                sil_score = silhouette_score(adata.obsm['X_pca'], adata.obs['leiden'])

                ari_score = 0
                if use_ari:
                    valid_idx = adata.obs['cell_type_ground_truth'].dropna().index
                    ari_score = adjusted_rand_score(
                        adata.obs.loc[valid_idx, 'cell_type_ground_truth'],
                        adata.obs.loc[valid_idx, 'leiden']
                    )

                if use_ari:
                    composite_score = 0.6 * sil_score + 0.4 * ari_score
                else:
                    composite_score = sil_score

                metrics_history.append({
                    'resolution': res,
                    'n_clusters': n_clusters,
                    'silhouette': sil_score,
                    'ari': ari_score,
                    'composite': composite_score
                })

                if composite_score > best_score:
                    best_res, best_score = res, composite_score
                    best_n_clusters = n_clusters

            except Exception as e:
                logger.warning(f"Error computing metrics for resolution {res}: {e}")

    sc.tl.leiden(adata, resolution=best_res, random_state=config.random_state)
    adata.obs['leiden_labels'] = adata.obs['leiden']

    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(f"{config.output_dir}/clustering_metrics.csv", index=False)

    logger.info(f"Best resolution: {best_res:.4f}, Clusters: {best_n_clusters}, Score: {best_score:.4f}")
    return adata, best_n_clusters


class EarlyStopping:
    """Early stopping handler for training."""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class ImprovedVAE(nn.Module):
    """Improved Variational Autoencoder with batch normalization and dropout."""
    def __init__(self, input_dim: int, embedding_dim: int = 32,
                 hidden_dims: List[int] = None, dropout: float = 0.2):
        super(ImprovedVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(prev_dim, embedding_dim)
        self.logvar = nn.Linear(prev_dim, embedding_dim)

        decoder_layers = []
        prev_dim = embedding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def improved_vae_loss(recon_x, x, mu, logvar, beta=0.001):
    """Improved VAE loss with beta-VAE regularization."""
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_improved_vae(adata: ad.AnnData, config: Config) -> ad.AnnData:
    """Train improved VAE with early stopping."""
    logger.info("Training VAE with early stopping...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedVAE(
        input_dim=adata.X.shape[1],
        embedding_dim=config.autoencoder_embedding_dim,
        hidden_dims=config.autoencoder_hidden_dims,
        dropout=config.autoencoder_dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.autoencoder_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    data_tensor = torch.tensor(
        adata.X.toarray() if issparse(adata.X) else adata.X,
        dtype=torch.float32
    )

    train_size = int(0.8 * len(data_tensor))
    train_data = data_tensor[:train_size]
    val_data = data_tensor[train_size:]

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config.autoencoder_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.autoencoder_batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=config.autoencoder_patience)
    model.train()
    loss_history = []

    for epoch in range(config.autoencoder_epochs):
        train_loss = 0
        model.train()
        for batch in train_dataloader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_data)
            loss = improved_vae_loss(recon_batch, batch_data, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = batch[0].to(device)
                recon_batch, mu, logvar = model(batch_data)
                loss = improved_vae_loss(recon_batch, batch_data, mu, logvar)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)

        loss_history.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{config.autoencoder_epochs}, Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}')

        if early_stopping(avg_val_loss):
            logger.info(f'Early stopping at epoch {epoch + 1}')
            break

    model.eval()
    with torch.no_grad():
        full_data = data_tensor.to(device)
        _, mu, _ = model(full_data)
        adata.obsm['X_autoencoder'] = mu.detach().cpu().numpy()

    pd.DataFrame(loss_history).to_csv(f"{config.output_dir}/vae_loss_history.csv", index=False)
    logger.info("VAE training completed")
    return adata


class EnhancedAutoencoderAnnotator:
    """Enhanced annotator with Optuna optimization and calibration."""
    def __init__(self, config: Config):
        self.config = config
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.smote = SMOTE(random_state=config.random_state) if config.use_smote else None
        self.confidence_threshold = config.confidence_threshold

    def _objective(self, trial, X_train_resampled, y_train_resampled, cv):
        """Optuna objective function for hyperparameter optimization."""
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        model_name = trial.suggest_categorical('model', ['xgb', 'lr', 'svc'])

        subsample_size = self.config.subsample_optuna_train
        if subsample_size is not None and subsample_size < len(X_train_resampled):
            logger.info(f"Subsampling {len(X_train_resampled)} training samples to {subsample_size} for trial {trial.number}")
            idx = np.random.RandomState(self.config.random_state + trial.number).choice(len(X_train_resampled), subsample_size, replace=False)
            X_sub, y_sub = X_train_resampled[idx], y_train_resampled[idx]
        else:
            X_sub, y_sub = X_train_resampled, y_train_resampled

        if model_name == 'xgb':
            import xgboost as xgb
            model = xgb.XGBClassifier(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                max_depth=trial.suggest_int('xgb_max_depth', 3, 10),
                learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.2, log=True),
                n_estimators=trial.suggest_int('xgb_n_estimators', 50, 200)
            )
        elif model_name == 'lr':
            C = trial.suggest_float('lr_C', 0.01, 100, log=True)
            penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2'])
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            model = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                random_state=self.config.random_state,
                max_iter=1000
            )
        else:
            C = trial.suggest_float('svc_C', 0.01, 100, log=True)
            model = SVC(
                C=C,
                gamma='scale',
                kernel='rbf',
                probability=True,
                random_state=self.config.random_state
            )

        try:
            scores = cross_val_score(model, X_sub, y_sub, cv=cv, scoring='accuracy', n_jobs=self.config.n_jobs)
            return scores.mean()
        except Exception:
            return 0.0

    def train(self, adata: ad.AnnData):
        """Train with optional hyperparameter optimization and calibration."""
        if 'X_autoencoder' not in adata.obsm or 'cell_type_ground_truth' not in adata.obs:
            logger.error("Missing embeddings or ground truth")
            return

        valid_indices = adata.obs['cell_type_ground_truth'].dropna().index
        if len(valid_indices) == 0:
            logger.error("No valid ground truth labels")
            return

        X = adata.obsm['X_autoencoder'][adata.obs_names.get_indexer(valid_indices)]
        y = adata.obs.loc[valid_indices, 'cell_type_ground_truth']

        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.config.random_state, stratify=y_encoded
        )

        if self.smote and len(np.unique(y_train)) > 1:
            try:
                X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
                logger.info("Applied SMOTE for class balancing")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}")
                X_train_resampled, y_train_resampled = X_train, y_train
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True,
                             random_state=self.config.random_state)

        if self.config.use_hyperparameter_optimization:
            logger.info(f"Starting hyperparameter optimization with {self.config.optuna_trials} trials...")

            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.random_state)
            )

            study.optimize(
                lambda trial: self._objective(trial, X_train_resampled, y_train_resampled, cv),
                n_trials=self.config.optuna_trials,
                show_progress_bar=True
            )

            best_params = study.best_params
            model_name = best_params.pop('model')

            if model_name == 'xgb':
                import xgboost as xgb
                base_model = xgb.XGBClassifier(
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    **{k.replace('xgb_', ''): v for k, v in best_params.items()}
                )
            elif model_name == 'lr':
                solver = 'liblinear' if best_params.get('lr_penalty') == 'l1' else 'lbfgs'
                base_model = LogisticRegression(
                    solver=solver,
                    random_state=self.config.random_state,
                    max_iter=1000,
                    **{k.replace('lr_', ''): v for k, v in best_params.items()}
                )
            else:
                base_model = SVC(
                    probability=True,
                    random_state=self.config.random_state,
                    **{k.replace('svc_', ''): v for k, v in best_params.items()}
                )

            self.best_model_name = model_name
            logger.info(f"Best model found by Optuna: {model_name} with score: {study.best_value:.4f}")

        else:
            import xgboost as xgb
            models = {
                'xgb': xgb.XGBClassifier(n_estimators=100, random_state=self.config.random_state, n_jobs=self.config.n_jobs),
                'lr': LogisticRegression(random_state=self.config.random_state, max_iter=1000),
                'svc': SVC(probability=True, random_state=self.config.random_state)
            }

            best_score = 0
            base_model = None
            for name, model in models.items():
                try:
                    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
                    mean_score = cv_scores.mean()
                    logger.info(f"{name} CV accuracy: {mean_score:.4f} ± {cv_scores.std():.4f}")

                    if mean_score > best_score:
                        best_score = mean_score
                        base_model = model
                        self.best_model_name = name
                except Exception as e:
                    logger.warning(f"Error training {name}: {e}")

            if base_model is None:
                logger.error("No default model could be trained.")
                return

        if base_model:
            logger.info(f"Training best model ({self.best_model_name}) on resampled data")
            base_model.fit(X_train_resampled, y_train_resampled)

            logger.info("Calibrating the model using the hold-out test set")
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
            calibrated_model.fit(X_test, y_test)

            self.best_model = calibrated_model

            val_probs = self.best_model.predict_proba(X_test)
            max_probs = np.max(val_probs, axis=1)
            self.confidence_threshold = np.quantile(max_probs, self.config.adaptive_quantile)
            logger.info(f"Set adaptive confidence threshold to: {self.confidence_threshold:.4f}")
        else:
            logger.error("No best model was selected or trained.")

    def predict(self, adata: ad.AnnData):
        """Predict with calibrated confidence scores."""
        if not self.best_model or 'X_autoencoder' not in adata.obsm:
            logger.error("Model not trained or embeddings missing")
            adata.obs['autoencoder_predictions'] = 'Unknown'
            return

        X = adata.obsm['X_autoencoder']

        y_pred_encoded = self.best_model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        adata.obs['autoencoder_predictions'] = y_pred

        y_prob = self.best_model.predict_proba(X)
        prob_df = pd.DataFrame(
            y_prob,
            index=adata.obs_names,
            columns=self.label_encoder.classes_
        )
        adata.obsm['autoencoder_probabilities'] = prob_df

        max_probs = np.max(y_prob, axis=1)
        adata.obs['autoencoder_confidence'] = max_probs

        low_conf_mask = max_probs < self.confidence_threshold
        adata.obs.loc[low_conf_mask, 'autoencoder_predictions'] = 'Low_confidence'

        logger.info(f"Predictions completed. {low_conf_mask.sum()} low-confidence predictions")


def create_visualizations(adata: ad.AnnData, config: Config):
    """Create comprehensive visualizations with reproducible UMAP."""
    sc.tl.umap(adata, random_state=config.random_state)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_configs = [
        ('cell_type_ground_truth', 'Ground Truth'),
        ('autoencoder_predictions', 'Predictions'),
        ('leiden_labels', 'Leiden Clusters'),
        ('autoencoder_confidence', 'Prediction Confidence')
    ]

    for i, (col, title) in enumerate(plot_configs):
        if col in adata.obs.columns:
            sc.pl.umap(adata, color=col, ax=axes[i//2, i%2], show=False, frameon=False)
            axes[i//2, i%2].set_title(title)

    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/umap_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    if 'autoencoder_confidence' in adata.obs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        adata.obs['autoencoder_confidence'].hist(bins=50, ax=ax1)
        ax1.axvline(config.confidence_threshold, color='red', linestyle='--',
                   label=f'Threshold: {config.confidence_threshold:.3f}')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Number of Cells')
        ax1.set_title('Confidence Score Distribution')
        ax1.legend()

        if 'cell_type_ground_truth' in adata.obs:
            conf_by_type = adata.obs.groupby('cell_type_ground_truth')['autoencoder_confidence'].mean()
            conf_by_type.plot(kind='bar', ax=ax2, rot=45)
            ax2.set_xlabel('Cell Type')
            ax2.set_ylabel('Average Confidence')
            ax2.set_title('Average Confidence by Cell Type')

        plt.tight_layout()
        plt.savefig(f"{config.output_dir}/confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_predictions(adata: ad.AnnData, config: Config):
    """Comprehensive evaluation of predictions."""
    eval_adata = adata[adata.obs['cell_type_ground_truth'].dropna().index].copy()

    if eval_adata.shape[0] == 0:
        logger.warning("No ground truth labels available for evaluation")
        return

    true_labels = eval_adata.obs['cell_type_ground_truth']
    pred_labels = eval_adata.obs['autoencoder_predictions']

    high_conf_mask = pred_labels != 'Low_confidence'
    if high_conf_mask.sum() == 0:
        logger.warning("All predictions are low confidence")
        return

    eval_true = true_labels[high_conf_mask]
    eval_pred = pred_labels[high_conf_mask]

    accuracy = accuracy_score(eval_true, eval_pred)
    kappa = cohen_kappa_score(eval_true, eval_pred)

    logger.info(f"High-confidence Accuracy: {accuracy:.4f}")
    logger.info(f"Cohen's Kappa: {kappa:.4f}")
    logger.info(f"High-confidence predictions: {high_conf_mask.sum()}/{len(pred_labels)} ({high_conf_mask.mean()*100:.1f}%)")

    report = classification_report(eval_true, eval_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{config.output_dir}/classification_report.csv")

    unique_labels = sorted(list(set(eval_true.unique()) | set(eval_pred.unique())))
    cm = confusion_matrix(eval_true, eval_pred, labels=unique_labels)

    plt.figure(figsize=(max(12, len(unique_labels)), max(10, len(unique_labels))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.title('Confusion Matrix (High Confidence Predictions)')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    if 'autoencoder_confidence' in eval_adata.obs:
        plt.figure(figsize=(10, 6))

        confidence_scores = eval_adata.obs.loc[high_conf_mask, 'autoencoder_confidence']
        correct_predictions = (eval_true == eval_pred).astype(int)

        n_bins = 10
        bin_boundaries = np.linspace(confidence_scores.min(), confidence_scores.max(), n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidence_scores >= bin_boundaries[i]) & (confidence_scores < bin_boundaries[i+1])
            if i == n_bins - 1:
                mask = (confidence_scores >= bin_boundaries[i]) & (confidence_scores <= bin_boundaries[i+1])

            if mask.sum() > 0:
                bin_accuracy = correct_predictions[mask].mean()
                bin_confidence = confidence_scores[mask].mean()
                bin_count = mask.sum()

                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)

        plt.scatter(bin_confidences, bin_accuracies, s=[c*5 for c in bin_counts], alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Accuracy')
        plt.title('Confidence Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{config.output_dir}/calibration_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    metrics = {
        'accuracy': float(accuracy),
        'kappa': float(kappa),
        'n_cells_total': int(len(eval_adata)),
        'n_cells_high_confidence': int(high_conf_mask.sum()),
        'high_confidence_ratio': float(high_conf_mask.mean()),
        'n_true_labels': int(len(eval_true.unique())),
        'n_pred_labels': int(len(eval_pred.unique())),
        'confidence_threshold': float(config.confidence_threshold)
    }

    with open(f"{config.output_dir}/evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


def run_annotation_pipeline(config: Config, data_path: str = None, annotations_path: str = None):
    """Run the complete optimized annotation pipeline."""
    logger.info("Starting optimized annotation pipeline...")

    if data_path is None:
        download_data()
        data_path = './data/10x-Multiome-Pbmc10k-RNA.h5'
        annotations_path = './data/pbmc10k_annotations.csv'

    adata = load_and_prepare_data(data_path)

    if annotations_path and os.path.exists(annotations_path):
        try:
            df = pd.read_csv(annotations_path, index_col=0)
            df.index = df.index.str.split('_').str[0].astype(str)
            adata.obs_names = adata.obs_names.astype(str)
            aligned_df = df.reindex(adata.obs_names)

            annotation_col = 'seurat_new_annotation'
            if annotation_col in aligned_df.columns:
                adata.obs['cell_type_ground_truth'] = aligned_df[annotation_col]
                valid_count = adata.obs['cell_type_ground_truth'].dropna().shape[0]
                logger.info(f"Loaded {valid_count} valid ground truth labels")
            else:
                logger.warning(f"Annotation column '{annotation_col}' not found")

        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")

    adata = enhanced_preprocessing(adata, config)
    adata, num_clusters = optimized_leiden_clustering(adata, config)
    adata = train_improved_vae(adata, config)

    annotator = EnhancedAutoencoderAnnotator(config)
    annotator.train(adata)
    annotator.predict(adata)

    evaluate_predictions(adata, config)
    create_visualizations(adata, config)

    adata.write(f"{config.output_dir}/annotated_data.h5ad")

    config_dict = {
        'leiden_resolution_range': config.leiden_resolution_range,
        'leiden_resolution_steps': config.leiden_resolution_steps,
        'autoencoder_embedding_dim': config.autoencoder_embedding_dim,
        'autoencoder_hidden_dims': config.autoencoder_hidden_dims,
        'autoencoder_epochs': config.autoencoder_epochs,
        'confidence_threshold': annotator.confidence_threshold,
        'adaptive_quantile': config.adaptive_quantile,
        'use_hyperparameter_optimization': config.use_hyperparameter_optimization,
        'optuna_trials': config.optuna_trials,
        'subsample_optuna_train': config.subsample_optuna_train,
        'random_state': config.random_state,
        'best_model': annotator.best_model_name if annotator.best_model_name else 'none'
    }

    with open(f"{config.output_dir}/config_used.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Optimized pipeline completed successfully!")
    return adata


def analyze_optimization_results(config: Config):
    """Analyze the results of the optimization pipeline."""
    results_dir = Path(config.output_dir)

    summary = {}

    if (results_dir / "vae_loss_history.csv").exists():
        loss_df = pd.read_csv(results_dir / "vae_loss_history.csv")
        summary['vae_epochs_trained'] = int(len(loss_df))
        summary['vae_final_train_loss'] = float(loss_df['train_loss'].iloc[-1])
        summary['vae_final_val_loss'] = float(loss_df['val_loss'].iloc[-1])
        summary['vae_early_stopped'] = bool(len(loss_df) < config.autoencoder_epochs)

    if (results_dir / "clustering_metrics.csv").exists():
        cluster_df = pd.read_csv(results_dir / "clustering_metrics.csv")
        best_idx = cluster_df['composite'].idxmax()
        summary['best_resolution'] = float(cluster_df.loc[best_idx, 'resolution'])
        summary['best_silhouette'] = float(cluster_df.loc[best_idx, 'silhouette'])
        summary['best_ari'] = float(cluster_df.loc[best_idx, 'ari'])
        summary['best_n_clusters'] = int(cluster_df.loc[best_idx, 'n_clusters'])

    if (results_dir / "evaluation_metrics.json").exists():
        with open(results_dir / "evaluation_metrics.json", 'r') as f:
            eval_metrics = json.load(f)
        summary.update(eval_metrics)

    if (results_dir / "config_used.json").exists():
        with open(results_dir / "config_used.json", 'r') as f:
            config_used = json.load(f)
        summary['optimization_used'] = bool(config_used['use_hyperparameter_optimization'])
        summary['best_model_type'] = config_used['best_model']
        summary['optuna_subsample'] = config_used.get('subsample_optuna_train')
        if summary['optuna_subsample'] is not None:
             summary['optuna_subsample'] = int(summary['optuna_subsample'])

    with open(results_dir / "optimization_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*50)

    if 'vae_early_stopped' in summary:
        logger.info(f"VAE Training: {summary['vae_epochs_trained']} epochs (Early stopped: {summary['vae_early_stopped']})")
        logger.info(f"Final Train/Val Loss: {summary['vae_final_train_loss']:.4f} / {summary['vae_final_val_loss']:.4f}")

    if 'best_resolution' in summary:
        logger.info(f"Best Clustering: Resolution {summary['best_resolution']:.4f}, {summary['best_n_clusters']} clusters")
        logger.info(f"Silhouette: {summary['best_silhouette']:.4f}, ARI: {summary['best_ari']:.4f}")

    if 'accuracy' in summary:
        logger.info(f"Final Accuracy: {summary['accuracy']:.4f} (Kappa: {summary['kappa']:.4f})")
        logger.info(f"High Confidence: {summary['high_confidence_ratio']*100:.1f}% of predictions")

    if 'best_model_type' in summary:
        logger.info(f"Best Model: {summary['best_model_type']}")

    if 'optuna_subsample' in summary:
        logger.info(f"Optuna Subsampling: {summary['optuna_subsample']}")

    logger.info("="*50)

    return summary


def create_optimized_config() -> Config:
    """Create an optimized configuration."""
    return Config(
        leiden_resolution_range=(0.01, 0.2),
        leiden_resolution_steps=15,
        leiden_k_neighbors=30,
        autoencoder_embedding_dim=32,
        autoencoder_hidden_dims=[512, 256, 128, 64],
        autoencoder_epochs=100,
        autoencoder_batch_size=128,
        autoencoder_lr=0.001,
        autoencoder_dropout=0.1,
        autoencoder_patience=7,
        confidence_threshold=0.7,
        adaptive_quantile=0.05,
        use_smote=True,
        cross_validation_folds=3,
        use_hyperparameter_optimization=True,
        optuna_trials=50,
        subsample_optuna_train=5000,
        n_top_genes=3000,
        min_genes_per_cell=200,
        max_mt_percent=20,
        min_ground_truth_ratio=0.7,
        random_state=42,
        n_jobs=-1
    )


if __name__ == "__main__":
    print("Torch CUDA available:", torch.cuda.is_available())
    
    config = create_optimized_config()

    logger.info("Starting pipeline with optimized configuration...")
    logger.info(f"Hyperparameter optimization: {config.use_hyperparameter_optimization}")
    logger.info(f"Optuna trials: {config.optuna_trials}")
    logger.info(f"Early stopping patience: {config.autoencoder_patience}")
    logger.info(f"Adaptive confidence quantile: {config.adaptive_quantile}")
    logger.info(f"Cross-validation folds: {config.cross_validation_folds}")
    logger.info(f"Optuna subsample: {config.subsample_optuna_train}")

    adata = run_annotation_pipeline(config)
    summary = analyze_optimization_results(config)

    logger.info("All optimizations successfully implemented!")

    if 'vae_early_stopped' in summary and summary['vae_early_stopped']:
        epochs_saved = config.autoencoder_epochs - summary['vae_epochs_trained']
        time_saved_pct = (epochs_saved / config.autoencoder_epochs) * 100
        logger.info(f"Early stopping saved ~{time_saved_pct:.1f}% training time")

    print("\n🎯 OPTIMIZATION BENEFITS:")
    print("✅ Adaptive marker gene discovery")
    print("✅ Smart ARI weighting based on ground truth coverage")
    print("✅ VAE early stopping with validation loss monitoring")
    print("✅ Automated hyperparameter optimization with Optuna")
    print("✅ Calibrated confidence thresholds on a hold-out set")
    print("✅ Reproducible UMAP with fixed random state")
    print("✅ Comprehensive evaluation and visualization")
