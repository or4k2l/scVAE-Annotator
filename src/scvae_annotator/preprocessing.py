"""
Preprocessing functions for scVAE-Annotator.
"""

import os
from typing import List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
from scipy.sparse import issparse
from sklearn.impute import KNNImputer

from .config import Config, logger


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
    from pathlib import Path
    
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
