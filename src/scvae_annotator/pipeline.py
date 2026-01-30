"""
Pipeline and Evaluation functions for scVAE-Annotator.
"""

import os
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, 
                              confusion_matrix, cohen_kappa_score)

from .config import Config, logger
from .preprocessing import download_data, load_and_prepare_data, enhanced_preprocessing
from .clustering import optimized_leiden_clustering
from .vae import train_improved_vae
from .annotator import EnhancedAutoencoderAnnotator
from .visualization import create_visualizations


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
