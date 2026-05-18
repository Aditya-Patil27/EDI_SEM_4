"""
Model definitions for vibration fault detection.
Combines: CNNDetector (PyTorch, from old pipeline), CompanyClassifier (new pipeline),
TF CNN (from train_cnn.py).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# New pipeline models
from ml_models import CompanyClassifier, TransferLearningAdapter, TORCH_AVAILABLE

# PyTorch CNNDetector (old pipeline, 60-dim input) - optional
try:
    import torch
    import torch.nn as nn

    class CNNDetector(nn.Module):
        """
        1D-CNN for vibration fault classification (60-dim features).
        3 conv layers (32→64→128) + batch norm + dropout + 3 FC layers.
        """
        def __init__(self, num_classes, input_size=60):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.drop1 = nn.Dropout(0.2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.drop2 = nn.Dropout(0.2)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.MaxPool1d(2, 2)
            self.drop3 = nn.Dropout(0.3)
            self.flatten = nn.Flatten()
            self.fc1 = nn.LazyLinear(128)
            self.bn_fc1 = nn.BatchNorm1d(128)
            self.drop_fc = nn.Dropout(0.4)
            self.fc2 = nn.Linear(128, 64)
            self.drop_fc2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.pool1(self.drop1(torch.relu(self.bn1(self.conv1(x)))))
            x = self.pool2(self.drop2(torch.relu(self.bn2(self.conv2(x)))))
            x = self.pool3(self.drop3(torch.relu(self.bn3(self.conv3(x)))))
            x = self.flatten(x)
            x = self.drop_fc(torch.relu(self.bn_fc1(self.fc1(x))))
            x = self.drop_fc2(torch.relu(self.fc2(x)))
            return self.fc3(x)

except Exception:
    class CNNDetector:
        """Placeholder — torch unavailable."""
        def __init__(self, num_classes=3, input_size=60): pass
        def __call__(self, x): raise NotImplementedError("PyTorch unavailable")
        def to(self, *a, **kw): return self
        def eval(self): pass
        def load_state_dict(self, *a, **kw): pass
        def state_dict(self): return {}


def build_tf_cnn(input_shape, num_classes):
    """TensorFlow 1D CNN (< 12K parameters)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ])
    return model
