import os
import yaml
import json
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_anomaly_detector():
    config = load_config()
    proc_dir = config['data']['processed_dir']
    
    X_path = os.path.join(proc_dir, "X_features.npy")
    y_path = os.path.join(proc_dir, "y_labels.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Processed data not found at {proc_dir}")
        return
        
    X_all = np.load(X_path)
    y_all = np.load(y_path)
    
    # Train GMM only on "Normal" class
    normal_indices = np.array([str(y).lower() == "normal" for y in y_all])
    
    if not np.any(normal_indices):
        print("No 'Normal' data found to train anomaly detector.")
        return
        
    X_normal = X_all[normal_indices]
    
    print(f"Training GMM on {len(X_normal)} Normal samples...")
    
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=config['project']['seed'])
    gmm.fit(X_normal)
    
    # Evaluate anomaly detection (treating non-Normal as anomalies)
    scores = gmm.score_samples(X_all)
    
    # Binary labels: 1 for Normal, 0 for Anomaly
    y_binary = normal_indices.astype(int)
    
    if len(np.unique(y_binary)) > 1:
        auc = roc_auc_score(y_binary, scores)
        print(f"Anomaly Detection (GMM) AUC over all data: {auc:.4f}")
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gmm_anomaly.pkl")
    joblib.dump(gmm, model_path)
    print(f"Saved GMM to {model_path}")

if __name__ == "__main__":
    train_anomaly_detector()
