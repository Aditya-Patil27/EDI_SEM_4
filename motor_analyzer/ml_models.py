"""
ML Models — CNNDetector, CompanyClassifier, ensemble anomaly detectors.
Integrates ml repo's CNN architecture with motor_analyzer's streaming workflow.
Torch is optional; falls back to sklearn RandomForest for company classification.
"""

import os
import json
import pickle
import logging
import warnings
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Falling back to sklearn-based models.")



if TORCH_AVAILABLE:
    class CNNDetector(nn.Module):
        """
        1D-CNN for vibration classification — adapted from ml repo's architecture.
        Input: (batch, 1, feature_dim) → Output: (batch, num_classes)
        """
        def __init__(self, feature_dim: int = 28, num_classes: int = 4):
            super().__init__()
            self.feature_dim = feature_dim
            self.num_classes = num_classes

            def conv1d_out(l_in, k, pad=0, dilation=1, stride=1):
                return (l_in + 2 * pad - dilation * (k - 1) - 1) // stride + 1

            l1 = conv1d_out(feature_dim, 3)
            p1 = conv1d_out(l1, 2, stride=2)
            l2 = conv1d_out(p1, 3)
            p2 = conv1d_out(l2, 2, stride=2)

            self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(16)
            self.pool1 = nn.MaxPool1d(2)
            self.drop1 = nn.Dropout(0.2)

            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(32)
            self.pool2 = nn.MaxPool1d(2)
            self.drop2 = nn.Dropout(0.2)

            conv_out_dim = 32 * p2
            self.fc1 = nn.Linear(conv_out_dim, 64)
            self.drop3 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.drop1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.drop2(x)

            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.drop3(x)
            x = self.fc2(x)
            return x

        @torch.no_grad()
        def predict(self, x: np.ndarray) -> np.ndarray:
            self.eval()
            if x.ndim == 1:
                x = x[np.newaxis, :]
            if x.ndim == 2:
                x = np.expand_dims(x, axis=1)
            t = torch.tensor(x, dtype=torch.float32)
            out = self.forward(t)
            return torch.argmax(out, dim=1).numpy()

        @torch.no_grad()
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            self.eval()
            if x.ndim == 1:
                x = x[np.newaxis, :]
            if x.ndim == 2:
                x = np.expand_dims(x, axis=1)
            t = torch.tensor(x, dtype=torch.float32)
            out = self.forward(t)
            return F.softmax(out, dim=1).numpy()

    _CNN_CLS = CNNDetector
else:
    class _SklearnCNN:
        """Placeholder — sklearn RandomForest used instead when torch unavailable."""
        def __init__(self, feature_dim=28, num_classes=4):
            self.feature_dim = feature_dim
            self.num_classes = num_classes
        def predict(self, x): raise NotImplementedError
        def predict_proba(self, x): raise NotImplementedError
        def load_state_dict(self, *a, **kw): pass
        def to(self, *a, **kw): return self
        def eval(self): pass
        def state_dict(self): return {}

    _CNN_CLS = _SklearnCNN


class TransferLearningAdapter:
    """
    Wraps pretrained 39-dim CNN with a 28→39 expansion layer for transfer learning.

    Architecture:
        Input (28) → Linear(28, 39) → ReLU → PretrainedCNN(39→3) → Output (3)

    The pretrained CNN stays frozen; only the expansion layer trains initially.
    When torch is unavailable, trains an sklearn RandomForest as fallback.

    Usage:
        adapter = TransferLearningAdapter(weights_path='model_best.pth')
        adapter.train(X_28dim, y_company_labels)
        idx, name, conf = adapter.predict(features)
    """
    PRETRAINED_CLASSES = ["Normal", "Unbalanced", "Bearing Fault"]

    def __init__(self, weights_path: str = None, adapt_dim: int = 28,
                 pretrain_dim: int = 39, num_classes: int = 3):
        self.adapt_dim = adapt_dim
        self.pretrain_dim = pretrain_dim
        self.num_classes = num_classes
        self.class_names = list(self.PRETRAINED_CLASSES)
        self.trained = False
        self.weights_path = weights_path

        if TORCH_AVAILABLE and weights_path and os.path.exists(weights_path):
            self._init_torch(weights_path)
        else:
            self._init_sklearn()

    def _init_torch(self, weights_path):
        self.pretrained_cnn = CNNDetector(
            feature_dim=self.pretrain_dim, num_classes=self.num_classes)
        data = torch.load(weights_path, map_location='cpu')
        sd = data.get('model_state', data)
        self.pretrained_cnn.load_state_dict(sd, strict=False)
        for param in self.pretrained_cnn.parameters():
            param.requires_grad = False
        self.expansion = nn.Linear(self.adapt_dim, self.pretrain_dim)
        logging.info(f"TransferLearningAdapter: loaded pretrained weights from {weights_path}")

    def _init_sklearn(self):
        if self.weights_path:
            logging.warning(f"PyTorch unavailable — cannot load {self.weights_path}. "
                            "Training sklearn fallback without transfer learning.")
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
        ])

    if TORCH_AVAILABLE:
        def _forward_torch(self, X):
            x = torch.tensor(X, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            if x.ndim == 2:
                x = x.unsqueeze(1)
            x = x.squeeze(1)
            x = F.relu(self.expansion(x))
            x = x.unsqueeze(1)
            return self.pretrained_cnn(x)

        @torch.no_grad()
        def _predict_torch(self, X):
            self.pretrained_cnn.eval()
            out = self._forward_torch(X)
            proba = F.softmax(out, dim=1).numpy()
            idx = int(np.argmax(proba, axis=1)[0])
            conf = float(np.max(proba, axis=1)[0])
            return idx, conf, proba[0]

        def _train_torch(self, X, y):
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42)
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.long)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.expansion.parameters(), lr=0.001)

            best_loss = float('inf')
            for epoch in range(50):
                self.pretrained_cnn.eval()
                self.expansion.train()
                optimizer.zero_grad()
                x = X_train_t
                x = F.relu(self.expansion(x))
                x = x.unsqueeze(1)
                out = self.pretrained_cnn(x)
                loss = criterion(out, y_train_t)
                loss.backward()
                optimizer.step()

                self.expansion.eval()
                with torch.no_grad():
                    x_val = F.relu(self.expansion(X_val_t))
                    x_val = x_val.unsqueeze(1)
                    val_out = self.pretrained_cnn(x_val)
                    val_loss = criterion(val_out, y_val_t)
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()

    def train(self, X, y, class_names=None):
        if class_names:
            self.class_names = list(class_names)
        if TORCH_AVAILABLE and hasattr(self, 'expansion'):
            self._train_torch(X, y)
        else:
            self.model.fit(X, y)
        self.trained = True

    def predict(self, features):
        if not self.trained:
            return 0, "Unknown", 0.0
        feats = np.array(features)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        if TORCH_AVAILABLE and hasattr(self, 'expansion'):
            idx, conf, proba = self._predict_torch(feats)
        else:
            proba = self.model.predict_proba(feats)
            idx = int(np.argmax(proba, axis=1)[0])
            conf = float(np.max(proba, axis=1)[0])
        name = self.class_names[idx] if idx < len(self.class_names) else "Unknown"
        return idx, name, conf

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if TORCH_AVAILABLE and hasattr(self, 'expansion'):
            torch.save({
                'expansion': self.expansion.state_dict(),
                'pretrained_cnn': self.pretrained_cnn.state_dict(),
                'class_names': self.class_names,
            }, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'class_names': self.class_names,
                }, f)

    @classmethod
    def load(cls, path, adapt_dim=28, pretrain_dim=39, num_classes=3):
        if not os.path.exists(path):
            return cls()
        try:
            if TORCH_AVAILABLE:
                data = torch.load(path, map_location='cpu')
                obj = cls.__new__(cls)
                obj.adapt_dim = adapt_dim
                obj.pretrain_dim = pretrain_dim
                obj.num_classes = num_classes
                obj.class_names = data.get('class_names', list(cls.PRETRAINED_CLASSES))
                obj._init_torch(obj.weights_path or '')
                if 'expansion' in data:
                    obj.expansion.load_state_dict(data['expansion'])
                if 'pretrained_cnn' in data:
                    obj.pretrained_cnn.load_state_dict(data['pretrained_cnn'])
                obj.trained = True
                return obj
            else:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                obj = cls()
                obj.model = data['model']
                obj.class_names = data.get('class_names', list(cls.PRETRAINED_CLASSES))
                obj.trained = True
                return obj
        except Exception as e:
            logging.warning(f"Failed to load transfer adapter: {e}")
            return cls()


class CompanyClassifier:
    """
    Company/machine-type identifier.
    Uses sklearn RandomForest when torch unavailable, CNNDetector when available.
    First identifies the machine, then dispatches to a per-company anomaly model.
    """
    def __init__(self, feature_dim: int = 28, num_companies: int = 4, weights_path: str = None):
        self.feature_dim = feature_dim
        self.num_companies = num_companies
        self.company_names = ["Unknown"]
        self.trained = False
        self.use_torch = TORCH_AVAILABLE

        if self.use_torch:
            self.model = CNNDetector(feature_dim=feature_dim, num_classes=num_companies)
            if weights_path and os.path.exists(weights_path):
                try:
                    self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                    self.trained = True
                    logging.info(f"Loaded torch CNN weights from {weights_path}")
                except Exception as e:
                    logging.warning(f"Could not load torch weights: {e}")
        else:
            # sklearn fallback — trained via train() call
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
            ])

    def train(self, X: np.ndarray, y: np.ndarray, company_names: list = None,
              noise_scale: float = 0.0):
        """Train the company classifier on labeled feature data.
        
        Args:
            noise_scale: Standard Gaussian noise multiplier (× feature std).
                         0=clean (trivially separable), 0.3-0.5=realistic overlap.
        """
        if company_names:
            self.company_names = list(company_names)
        if noise_scale > 0:
            noise = np.random.RandomState(42).randn(*X.shape) * X.std(axis=0) * noise_scale
            X = X + noise
        if self.use_torch:
            self._train_torch(X, y)
        else:
            self.model.fit(X, y)
        self.trained = True

    def _train_torch(self, X, y):
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_t = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(np.expand_dims(X_val, axis=1), dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        best_loss = float('inf')
        for epoch in range(30):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_t)
                val_loss = criterion(val_out, y_val_t)
            if val_loss < best_loss:
                best_loss = val_loss

    def predict(self, features: np.ndarray) -> tuple:
        """Returns (company_index, company_name, confidence)"""
        if not self.trained:
            return 0, "Unknown", 0.0
        feats = np.array(features)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        proba = self.model.predict_proba(feats)
        idx = int(np.argmax(proba, axis=1)[0])
        conf = float(np.max(proba, axis=1)[0])
        name = self.company_names[idx] if idx < len(self.company_names) else "Unknown"
        return idx, name, conf

    def save(self, path: str, company_names: list = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if company_names:
            self.company_names = company_names
        if self.use_torch:
            torch.save({
                'model_state': self.model.state_dict(),
                'feature_dim': self.feature_dim,
                'num_companies': self.num_companies,
                'company_names': self.company_names,
            }, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_dim': self.feature_dim,
                    'num_companies': self.num_companies,
                    'company_names': self.company_names,
                }, f)

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            return cls()
        try:
            if TORCH_AVAILABLE:
                data = torch.load(path, map_location='cpu')
                obj = cls(
                    feature_dim=data.get('feature_dim', 28),
                    num_companies=data.get('num_companies', 4),
                )
                obj.model.load_state_dict(data['model_state'])
                obj.company_names = data.get('company_names', ['Unknown'])
                obj.trained = True
                return obj
            else:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                obj = cls(
                    feature_dim=data.get('feature_dim', 28),
                    num_companies=data.get('num_companies', 4),
                )
                obj.model = data['model']
                obj.company_names = data.get('company_names', ['Unknown'])
                obj.trained = True
                return obj
        except Exception as e:
            logging.warning(f"Failed to load company classifier: {e}")
            return cls()


class EnsembleAnomalyModel:
    """
    IsolationForest + OC-SVM ensemble anomaly detector.
    Both must agree → anomaly (reduces false alarms).
    """
    def __init__(self):
        self.if_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', IsolationForest(n_estimators=300, contamination=0.03, random_state=42))
        ])
        self.svm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneClassSVM(kernel='rbf', nu=0.05, gamma='scale'))
        ])
        self.trained = False

    def fit(self, X):
        X = np.array(X)
        self.if_pipe.fit(X)
        self.svm_pipe.fit(X)
        self.trained = True

    def predict_score(self, x):
        x_in = np.array(x)
        if x_in.ndim == 1:
            x_in = x_in.reshape(1, -1)
        if_pred  = self.if_pipe.predict(x_in)[0]
        if_score = -self.if_pipe.score_samples(x_in)[0]

        svm_pred  = self.svm_pipe.predict(x_in)[0]
        svm_score = -self.svm_pipe.score_samples(x_in)[0]

        def normalise(s):
            return float(1 / (1 + np.exp(-s + 1.5)))

        combined = (normalise(if_score) + normalise(svm_score)) / 2.0
        is_anomaly = bool((if_pred == -1) and (svm_pred == -1))
        return is_anomaly, combined


class GMMAnomalyDetector:
    """
    GMM-based anomaly detector (from ml repo approach).
    Trained only on 'Normal' data for a specific company.
    """
    def __init__(self, n_components: int = 3):
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.trained = False
        self.threshold = None

    def fit(self, X):
        self.gmm.fit(np.array(X))
        scores = self.gmm.score_samples(np.array(X))
        self.threshold = np.percentile(scores, 5)
        self.trained = True

    def predict_score(self, x):
        x_in = np.array(x)
        if x_in.ndim == 1:
            x_in = x_in.reshape(1, -1)
        score = self.gmm.score_samples(x_in)[0]
        is_anomaly = bool(score < self.threshold)
        norm_score = float(1 / (1 + np.exp(score - self.threshold)))
        return is_anomaly, norm_score


class PerCompanyModelRegistry:
    """
    Registry that manages one anomaly model per company.
    First identifies company via CNN, then dispatches to correct anomaly model.
    """
    def __init__(self, models_dir: str = "models/companies"):
        self.models_dir = models_dir
        self.company_classifier = CompanyClassifier()
        self.anomaly_models = {}  # company_name -> EnsembleAnomalyModel
        self.company_names = []
        os.makedirs(models_dir, exist_ok=True)
        self._load_manifest()

    def _load_manifest(self):
        manifest_path = os.path.join(self.models_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            self.company_names = data.get('companies', [])
        cls_path = os.path.join(self.models_dir, "..", "company_classifier.pth")
        self.company_classifier = CompanyClassifier.load(cls_path)

    def _save_manifest(self):
        manifest_path = os.path.join(self.models_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump({'companies': self.company_names}, f, indent=2)

    def identify_company(self, features: np.ndarray) -> tuple:
        """Returns (company_index, company_name, confidence)"""
        return self.company_classifier.predict(features)

    def get_anomaly_model(self, company: str):
        if company not in self.anomaly_models:
            path = os.path.join(self.models_dir, f"{company}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.anomaly_models[company] = pickle.load(f)
            else:
                return None
        return self.anomaly_models.get(company)

    def train_company_model(self, company: str, features: list):
        if company == "Unknown":
            return None
        model = EnsembleAnomalyModel()
        model.fit(features)
        self.anomaly_models[company] = model
        path = os.path.join(self.models_dir, f"{company}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        if company not in self.company_names:
            self.company_names.append(company)
            self._save_manifest()
        return model

    def save_company_classifier(self, path: str, company_names: list):
        self.company_classifier.save(path, company_names)
        self.company_names = company_names
        self._save_manifest()
