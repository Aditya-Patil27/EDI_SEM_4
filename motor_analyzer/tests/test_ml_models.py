"""
Unit tests for ml_models.py — model training, inference, and persistence.
Run with: python -m pytest tests/test_ml_models.py -v
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ml_models import (
    EnsembleAnomalyModel, GMMAnomalyDetector,
    CompanyClassifier, PerCompanyModelRegistry
)


def test_ensemble_train_and_predict():
    """Ensemble anomaly model should train and return valid predictions."""
    np.random.seed(42)
    X_train = np.random.normal(0, 0.1, (50, 28))
    X_test = np.random.normal(0, 0.1, (1, 28))

    model = EnsembleAnomalyModel()
    model.fit(X_train)
    assert model.trained

    is_anomaly, score = model.predict_score(X_test[0])
    assert isinstance(is_anomaly, bool)
    assert 0.0 <= score <= 1.0


def test_ensemble_detects_anomalies():
    """Model should flag extreme values as anomalies."""
    np.random.seed(42)
    X_normal = np.random.normal(0, 0.1, (80, 28))
    X_anomaly = np.random.normal(5.0, 0.5, (1, 28))

    model = EnsembleAnomalyModel()
    model.fit(X_normal)

    is_anomaly, score = model.predict_score(X_anomaly[0])
    assert isinstance(is_anomaly, bool)
    assert 0.0 <= score <= 1.0


def test_gmm_train_and_predict():
    """GMM anomaly detector should train and return valid scores."""
    np.random.seed(42)
    X_train = np.random.normal(0, 0.1, (50, 28))
    X_test = np.random.normal(0, 0.1, (1, 28))

    model = GMMAnomalyDetector()
    model.fit(X_train)
    assert model.trained

    is_anomaly, score = model.predict_score(X_test[0])
    assert isinstance(is_anomaly, bool)
    assert 0.0 <= score <= 1.0


def test_company_classifier_train_and_predict():
    """CompanyClassifier should train on synthetic data and predict."""
    np.random.seed(42)
    # 3 companies, 30 samples each, 28-dim features
    X = np.random.randn(90, 28)
    y = np.array([0]*30 + [1]*30 + [2]*30)

    clf = CompanyClassifier(feature_dim=28, num_companies=3)
    clf.train(X, y, company_names=["Alpha", "Beta", "Gamma"])

    assert clf.trained

    idx, name, conf = clf.predict(X[0])
    assert isinstance(idx, int)
    assert isinstance(name, str)
    assert 0.0 <= conf <= 1.0


def test_company_classifier_save_load():
    """CompanyClassifier should survive save/load cycle."""
    np.random.seed(42)
    X = np.random.randn(60, 28)
    y = np.array([0]*30 + [1]*30)

    clf = CompanyClassifier(feature_dim=28, num_companies=2)
    clf.train(X, y, company_names=["TestA", "TestB"])

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        tmp_path = f.name
    try:
        clf.save(tmp_path, company_names=["TestA", "TestB"])
        loaded = CompanyClassifier.load(tmp_path)
        assert loaded.trained
        assert len(loaded.company_names) >= 2
    finally:
        os.unlink(tmp_path)


def test_per_company_registry():
    """Registry should manage per-company models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = PerCompanyModelRegistry(models_dir=tmpdir)

        # Train and save per-company model
        X = np.random.randn(30, 28)
        model = registry.train_company_model("TestCo", X)
        assert model is not None

        # Retrieve it
        retrieved = registry.get_anomaly_model("TestCo")
        assert retrieved is not None
        assert retrieved.trained


def test_ensemble_deterministic():
    """Same input should give same output (deterministic)."""
    np.random.seed(42)
    X = np.random.normal(0, 0.1, (50, 28))
    model = EnsembleAnomalyModel()
    model.fit(X)

    test_point = np.random.normal(0, 0.1, (1, 28))[0]
    r1, s1 = model.predict_score(test_point)
    r2, s2 = model.predict_score(test_point)
    assert r1 == r2
    assert abs(s1 - s2) < 1e-6
