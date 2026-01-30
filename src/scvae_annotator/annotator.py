"""
Enhanced Annotator with Optuna optimization for scVAE-Annotator.
"""

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler

from .config import Config, logger


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
