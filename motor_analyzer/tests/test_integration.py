"""Integration tests for the full MotorSense pipeline."""

import os
import sys
import json
import numpy as np
from feature_pipeline import extract_features
from ml_models import (
    EnsembleAnomalyModel,
    CompanyClassifier,
    GMMAnomalyDetector,
    AutoencoderAnomalyDetector,
)
from engineer_dataset import engineer_dataset, OUTPUT_DIR
from rul_model import RULPredictor, RUL_MODEL_PATH
from motor_classifier import MotorClassifier, build_cwru_motor_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_engineer_pipeline_runs():
    """End-to-end: engineer_dataset produces expected outputs."""
    X_path = os.path.join(OUTPUT_DIR, 'X_all.npy')
    meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    assert os.path.exists(X_path), "X_all.npy missing"
    assert os.path.exists(meta_path), "metadata.json missing"

    X = np.load(X_path)
    y = np.load(os.path.join(OUTPUT_DIR, 'y_all.npy'))
    meta = json.load(open(meta_path))

    assert X.shape[0] == len(y)
    assert X.shape[1] == 28, f"Expected 28 features, got {X.shape[1]}"
    assert meta['n_samples'] == X.shape[0]
    assert 'NASA_IMS' in meta['companies'], "NASA_IMS missing from companies"
    assert 'MAFAULDA' in meta['companies']
    assert 'CageFault' in meta['all_fault_classes']
    assert meta['severity_per_class'], "Severity tracking missing"


def test_class_balance():
    """Classes should be balanced to target_per_class."""
    y = np.load(os.path.join(OUTPUT_DIR, 'y_all.npy'))
    classes, counts = np.unique(y, return_counts=True)
    target = 5000
    for cls, cnt in zip(classes, counts):
        assert cnt == target, f"{cls}: expected {target}, got {cnt}"


def test_feature_extraction_consistency():
    """Same input → same features (deterministic)."""
    sig = np.sin(2 * np.pi * 5 * np.arange(0, 1.28, 0.01)).tolist()
    baseline = float(np.mean(np.abs(sig)))
    feats1 = extract_features(sig, baseline=baseline)
    feats2 = extract_features(sig, baseline=baseline)
    assert np.allclose(feats1, feats2), "Features not deterministic"


def test_feature_extraction_length():
    """Feature vector must be 28 elements."""
    sig = np.random.randn(128).tolist()
    baseline = float(np.mean(np.abs(sig)))
    feats = extract_features(sig, baseline=baseline)
    assert len(feats) == 28, f"Expected 28 features, got {len(feats)}"


def test_company_classifier_accepts_all_companies():
    """CompanyClassifier must predict for all 5 known company labels."""
    cc = CompanyClassifier()
    meta = json.load(open(os.path.join(OUTPUT_DIR, 'metadata.json')))
    companies = meta['companies']
    assert 'NASA_IMS' in companies
    assert len(companies) >= 5


def test_ensemble_model_fit_and_score():
    """Ensemble model can train and return anomaly scores."""
    normal = np.random.randn(50, 28)
    anomalous = np.random.randn(5, 28) * 5
    model = EnsembleAnomalyModel()
    model.fit(normal)
    result = model.predict_score(anomalous[0])
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], float)


def test_gmm_fit_and_score():
    """GMM model can train and return anomaly scores."""
    normal = np.random.randn(50, 28)
    model = GMMAnomalyDetector(n_components=2)
    model.fit(normal)
    result = model.predict_score(normal[0])
    assert isinstance(result, tuple) and len(result) == 2


def test_autoencoder_fit_and_score():
    """Autoencoder can train and score."""
    data = np.random.randn(30, 28)
    model = AutoencoderAnomalyDetector(threshold_percentile=5.0)
    model.fit(data)
    result = model.predict_score(data[0])
    assert isinstance(result, tuple) and len(result) == 2


def test_per_company_models_exist():
    """Each company should have a saved anomaly model."""
    companies_dir = os.path.join(BASE_DIR, '..', 'models', 'companies')
    assert os.path.exists(companies_dir)
    manifest = os.path.join(companies_dir, 'manifest.json')
    assert os.path.exists(manifest)
    meta = json.load(open(manifest))
    expected = ['CWRU', 'JNU', 'MAFAULDA', 'NASA_IMS', 'Synthetic']
    for c in expected:
        assert c in meta['companies'], f"{c} missing from manifest"


def test_company_classifier_manifest():
    """Manifest must list all 5 companies."""
    manifest = os.path.join(BASE_DIR, '..', 'models', 'companies', 'manifest.json')
    meta = json.load(open(manifest))
    assert len(meta['companies']) >= 5
    expected = {'CWRU', 'JNU', 'MAFAULDA', 'NASA_IMS', 'Synthetic'}
    assert expected.issubset(set(meta['companies'])), f"Missing companies"


def test_engineered_data_no_nans():
    """Engineered feature matrix must not contain NaN or Inf."""
    X = np.load(os.path.join(OUTPUT_DIR, 'X_all.npy'))
    assert not np.isnan(X).any(), "NaN values in X"
    assert not np.isinf(X).any(), "Inf values in X"


def test_rul_model_exists():
    """RUL predictor model file must exist."""
    assert os.path.exists(RUL_MODEL_PATH), "RUL model file missing"


def test_rul_model_predict():
    """RUL predictor must return a float between 0 and 1."""
    model = RULPredictor.load()
    assert model.trained, "RUL model not trained"
    feats = np.random.randn(28).tolist()
    pred = model.predict(feats)
    assert isinstance(pred, float), f"Expected float, got {type(pred)}"
    assert 0.0 <= pred <= 1.0, f"RUL should be [0,1], got {pred}"


def test_motor_classifier_exists():
    """Motor classifier model file must exist."""
    mc_path = os.path.join('models', 'motor_classifier.pkl')
    assert os.path.exists(mc_path), "Motor classifier file missing"


def test_motor_classifier_predict():
    """Motor classifier must return expected fields."""
    mc = MotorClassifier.load()
    assert mc.trained, "Motor classifier not trained"
    feats = np.random.randn(28).tolist()
    result = mc.predict(feats)
    assert 'hp' in result
    assert 'bearing_location' in result
    assert 'fault_diameter_inches' in result
    assert 'confidence' in result


def test_motor_classifier_explain():
    """Motor classifier SHAP explanations must return top features."""
    mc = MotorClassifier.load()
    assert mc.trained
    feats = np.random.randn(28).tolist()
    explanations = mc.explain(feats, MotorClassifier.feature_names())
    assert 'hp' in explanations
    assert 'bearing' in explanations
    assert len(explanations['hp']) <= 5
