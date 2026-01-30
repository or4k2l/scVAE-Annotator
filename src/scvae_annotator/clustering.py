"""
Clustering functions for scVAE-Annotator.
"""

from typing import Tuple, Dict, Any, List
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score, adjusted_rand_score

from .config import Config, logger


def optimized_leiden_clustering(adata: ad.AnnData, config: Config) -> Tuple[ad.AnnData, int]:
    """Optimized Leiden clustering with adaptive metrics."""
    logger.info("Starting Leiden clustering optimization...")

    if getattr(config, "_leiden_resolution_explicit", False) and config.leiden_resolution is not None:
        resolutions = np.array([config.leiden_resolution])
    else:
        resolutions = np.linspace(*config.leiden_resolution_range, config.leiden_resolution_steps)
    best_res, best_score = config.leiden_resolution_range[0], -np.inf
    best_n_clusters = 0

    use_ari = False
    if 'cell_type_ground_truth' in adata.obs:
        labeled_ratio = adata.obs['cell_type_ground_truth'].dropna().shape[0] / adata.shape[0]
        use_ari = labeled_ratio >= config.min_ground_truth_ratio
        logger.info(f"Ground truth coverage: {labeled_ratio:.1%}, using ARI: {use_ari}")

    metrics_history = []

    for res in resolutions:
        sc.tl.leiden(
            adata,
            resolution=res,
            random_state=config.random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        n_clusters = len(adata.obs['leiden'].unique())

        try:
            if n_clusters > 1:
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
            else:
                sil_score = -1.0
                ari_score = 0.0
                composite_score = -1.0

            metrics_history.append({
                'resolution': res,
                'n_clusters': n_clusters,
                'silhouette': sil_score,
                'ari': ari_score,
                'composite': composite_score
            })

            if composite_score >= best_score:
                best_res, best_score = res, composite_score
                best_n_clusters = n_clusters

        except Exception as e:
            logger.warning(f"Error computing metrics for resolution {res}: {e}")
            metrics_history.append({
                'resolution': res,
                'n_clusters': n_clusters,
                'silhouette': -1.0,
                'ari': 0.0,
                'composite': -1.0
            })

    sc.tl.leiden(
        adata,
        resolution=best_res,
        random_state=config.random_state,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    adata.obs['leiden'] = adata.obs['leiden'].astype(str)
    adata.obs['leiden_labels'] = adata.obs['leiden'].astype(str)

    if best_n_clusters <= 0:
        best_n_clusters = len(adata.obs['leiden'].unique())

    metrics_df = pd.DataFrame(metrics_history)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(f"{config.output_dir}/clustering_metrics.csv", index=False)

    logger.info(f"Best resolution: {best_res:.4f}, Clusters: {best_n_clusters}, Score: {best_score:.4f}")
    return adata, best_n_clusters
