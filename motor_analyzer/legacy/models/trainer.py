"""
Training logic for PyTorch CNNDetector + GMM anomaly detector.
"""

import os
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE
from utils.gpu import configure_gpu
from models.detector import CNNDetector

logging.basicConfig(level=logging.INFO)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def augment_data(X, y, noise_std=0.05):
    noise = np.random.normal(0, noise_std, X.shape)
    return np.vstack((X, X + noise)), np.concatenate((y, y))


def train_gmm(X_train, X_val, y_val, n_components=5):
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    gmm.fit(X_train)
    scores = gmm.score_samples(X_val)
    threshold = np.percentile(scores, 5)
    logging.info(f"GMM threshold: {threshold:.2f}")
    return gmm, threshold


class FaultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def group_split(X, y, groups, test_size=0.15, random_state=42):
    np.random.seed(random_state)
    unique = np.unique(groups)
    np.random.shuffle(unique)
    n_test = int(len(unique) * test_size)
    test_g = set(unique[:n_test])
    train_idx = [i for i, g in enumerate(groups) if g not in test_g]
    test_idx = [i for i, g in enumerate(groups) if g in test_g]
    return train_idx, test_idx


def main():
    device = configure_gpu()
    config = load_config()
    proc_dir = config["data"]["processed_dir"]

    X = np.load(os.path.join(proc_dir, "X_features.npy"))
    y_labels = np.load(os.path.join(proc_dir, "y_labels.npy"))
    classes = config["data"]["classes"]
    class_map = {c: i for i, c in enumerate(classes)}
    y = np.array([class_map.get(lbl, 0) for lbl in y_labels])

    groups = np.arange(len(y)) // 100
    train_idx, test_idx = group_split(X, y, groups)
    train_idx, val_idx = group_split(X[train_idx], y[train_idx], np.array(groups)[train_idx], test_size=0.17)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    logging.info(f"Train/Val/Test: {len(X_tr)} / {len(X_val)} / {len(test_idx)}")

    smote = SMOTE(random_state=42)
    try:
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
    except ValueError:
        logging.warning("SMOTE failed, proceeding without.")

    X_tr, y_tr = augment_data(X_tr, y_tr)

    gmm, threshold = train_gmm(X_tr, X_val, y_val)

    model = CNNDetector(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["training"]["learning_rate"],
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"], eta_min=1e-6)

    train_loader = DataLoader(FaultDataset(np.expand_dims(X_tr, 1), y_tr),
                              batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(FaultDataset(np.expand_dims(X_val, 1), y_val),
                            batch_size=config["training"]["batch_size"])

    best_loss = float("inf")
    patience = config["training"]["patience"]
    counter = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            correct += (model(bx).argmax(1) == by).sum().item()
            total += by.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                loss = criterion(model(bx), by)
                val_loss += loss.item() * bx.size(0)
                val_correct += (model(bx).argmax(1) == by).sum().item()
                val_total += by.size(0)

        train_loss /= total
        val_loss /= val_total
        logging.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={correct/total:.4f} val_loss={val_loss:.4f} val_acc={val_correct/val_total:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "model_best.pth")
        else:
            counter += 1
            if counter >= patience:
                logging.info("Early stopping")
                break

        scheduler.step()

    logging.info("Training complete.")
