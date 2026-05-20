"""
Anomaly detection models.
Combines: EnsembleAnomalyModel, GMMAnomalyDetector, PerCompanyModelRegistry (new pipeline)
+ GMM training from old pipeline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_models import (
    EnsembleAnomalyModel, GMMAnomalyDetector, PerCompanyModelRegistry
)

import os
import yaml
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score


def train_gmm_anomaly(config_path="config.yaml"):
    """Train GMM anomaly detector on Normal class only. Saves to models/gmm_anomaly.pkl."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    proc_dir = config["data"]["processed_dir"]
    X = np.load(os.path.join(proc_dir, "X_features.npy"))
    y = np.load(os.path.join(proc_dir, "y_labels.npy"))

    normal_idx = np.array([str(lbl).lower() == "normal" for lbl in y])
    if not np.any(normal_idx):
        print("No Normal data found.")
        return

    X_normal = X[normal_idx]
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=config["project"]["seed"])
    gmm.fit(X_normal)

    scores = gmm.score_samples(X)
    y_bin = normal_idx.astype(int)
    if len(np.unique(y_bin)) > 1:
        auc = roc_auc_score(y_bin, scores)
        print(f"GMM Anomaly AUC: {auc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(gmm, "models/gmm_anomaly.pkl")
    print("Saved models/gmm_anomaly.pkl")
