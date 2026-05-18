"""
Train pipeline — loads ingested data, trains CompanyClassifier + per-company
anomaly models, and persists them for the Flask app.
"""
import os
import sys
import json
import yaml
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ml_models import CompanyClassifier, PerCompanyModelRegistry

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
COMPANY_MODELS_DIR = os.path.join(MODELS_DIR, 'companies')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMPANY_MODELS_DIR, exist_ok=True)

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def load_processed_data():
    X_path = os.path.join(PROCESSED_DIR, 'X_all.npy')
    y_path = os.path.join(PROCESSED_DIR, 'y_all.npy')
    company_path = os.path.join(PROCESSED_DIR, 'company_labels.npy')
    meta_path = os.path.join(PROCESSED_DIR, 'metadata.json')

    if not all(os.path.exists(p) for p in [X_path, y_path, company_path, meta_path]):
        log.error("Processed data not found. Run data_ingestion.py first.")
        return None, None, None, None

    X = np.load(X_path)
    y = np.load(y_path)
    company_labels = np.load(company_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    log.info(f"Loaded {len(X)} samples, {len(metadata['companies'])} companies")
    return X, y, company_labels, metadata


def train_company_classifier(X, company_labels, metadata):
    log.info("=== Training Company Classifier ===")
    company_names = metadata['companies']

    # Read noise scale from config
    cfg = load_config()
    noise_scale = cfg.get('training', {}).get('company_noise_scale', 0.0)
    log.info(f"  Noise scale: {noise_scale} (0=clean, >0=overlap)")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, company_labels, test_size=0.2, random_state=42, stratify=company_labels
    )

    clf = CompanyClassifier(feature_dim=X.shape[1], num_companies=len(company_names))
    clf.train(X_train, y_train, company_names=company_names, noise_scale=noise_scale)

    # Evaluate
    predictions = []
    for i in range(len(X_test)):
        idx, name, conf = clf.predict(X_test[i])
        predictions.append(idx)

    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test, predictions)
    log.info(f"  Test accuracy: {acc:.4f}")
    report = classification_report(y_test, predictions, target_names=company_names, output_dict=True)
    for c in company_names:
        log.info(f"  {c}: F1={report[c]['f1-score']:.3f}, support={report[c]['support']}")

    report_path = os.path.join(MODELS_DIR, 'classifier_report.json')
    with open(report_path, 'w') as f:
        json.dump({'accuracy': round(acc, 4), 'report': report}, f, indent=2)

    return clf, company_names, acc


def train_anomaly_models(X, y, company_labels, metadata):
    log.info("=== Training Per-Company Anomaly Models ===")
    company_names = metadata['companies']
    classes_per_company = metadata['classes_per_company']

    registry = PerCompanyModelRegistry(models_dir=COMPANY_MODELS_DIR)

    for ci, name in enumerate(company_names):
        mask = company_labels == ci
        Xc = X[mask]
        yc = y[mask]

        # Anomaly model is trained on 'Normal' class only
        normal_mask = yc == 'Normal'
        if not np.any(normal_mask):
            log.warning(f"  {name}: No Normal data, skipping anomaly model")
            continue

        X_normal = Xc[normal_mask]
        log.info(f"  {name}: {len(X_normal)} Normal samples -> training anomaly model")

        try:
            registry.train_company_model(name, X_normal.tolist())
            log.info(f"  {name}: model saved to {COMPANY_MODELS_DIR}/{name}.pkl")
        except Exception as e:
            log.error(f"  {name}: training failed: {e}")

    # Show per-company class distribution
    for ci, name in enumerate(company_names):
        mask = company_labels == ci
        Xc = X[mask]
        yc = y[mask]
        classes, counts = np.unique(yc, return_counts=True)
        log.info(f"  {name} class distribution: {dict(zip(classes, counts))}")

    return registry


def main():
    X, y, company_labels, metadata = load_processed_data()
    if X is None:
        return

    clf, company_names, acc = train_company_classifier(X, company_labels, metadata)

    # Save classifier
    cls_path = os.path.join(MODELS_DIR, 'company_classifier.pth')
    clf.save(cls_path, company_names=company_names)
    log.info(f"CompanyClassifier saved -> {cls_path}")

    # Save manifest
    manifest = {'companies': company_names, 'accuracy': round(acc, 4)}
    manifest_path = os.path.join(COMPANY_MODELS_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved -> {manifest_path}")

    # Train per-company anomaly models
    train_anomaly_models(X, y, company_labels, metadata)

    log.info("\nDone. All models trained and saved.")
    log.info(f"  Classifier: {cls_path}")
    log.info(f"  Anomaly models: {COMPANY_MODELS_DIR}/")


if __name__ == '__main__':
    main()
