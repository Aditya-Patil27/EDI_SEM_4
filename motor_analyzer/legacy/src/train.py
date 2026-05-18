"""
DEPRECATED — Legacy audio pipeline (60-dim features, librosa-based).
NOT used by the ESP32 streaming pipeline (28-dim, numpy-only).
Replaced by:
  - feature_pipeline.py  (28-dim streaming feature extraction)
  - ml_models.py         (sklearn-based anomaly detection)
  - app.py               (Flask backend for ESP32)
  - evaluate_pipeline.py (end-to-end eval on 28-dim features)
"""
import os
import yaml
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from src.gpu_setup import configure_gpu

logging.basicConfig(level=logging.INFO)

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def augment_data(X_train, y_train):
    """
    Data augmentation logic (Gaussian noise/time jitter only on train)
    """
    noise = np.random.normal(0, 0.05, X_train.shape)
    X_aug = X_train + noise
    X_ret = np.vstack((X_train, X_aug))
    y_ret = np.concatenate((y_train, y_train))
    return X_ret, y_ret

def train_gmm(X_train, X_val, y_val):
    """
    Train a GMM for Anomaly Detection using training data.
    """
    logging.info("Training GMM Anomaly Detector...")
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(X_train)
    val_scores = gmm.score_samples(X_val)
    threshold = np.percentile(val_scores, 5)
    logging.info(f"GMM Log-Likelihood Threshold set at: {threshold:.2f}")
    return gmm, threshold

class FaultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNNDetector(nn.Module):
    """
    DEPRECATED — Legacy PyTorch CNN for 60-dim audio features.
    NOT used by ESP32 pipeline. See ml_models.py CNNDetector
    (28-dim, lightweight 2-conv architecture) for the deployed model.
    """
    def __init__(self, num_classes, input_size=60):
        super(CNNDetector, self).__init__()
        # 3-layer deep CNN with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.3)
        
        # Use LazyLinear to auto-detect flattened size
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc = nn.ReLU()
        self.drop_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU()
        self.drop_fc2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc(x)
        x = self.drop_fc(x)
        
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.drop_fc2(x)
        x = self.fc3(x)
        return x

def group_split(X, y, groups, test_size=0.15, random_state=42):
    unique_groups = np.unique(groups)
    np.random.seed(random_state)
    np.random.shuffle(unique_groups)
    
    n_test = int(len(unique_groups) * test_size)
    test_groups = unique_groups[:n_test]
    train_groups = unique_groups[n_test:]
    
    test_idx = [i for i, g in enumerate(groups) if g in test_groups]
    train_idx = [i for i, g in enumerate(groups) if g in train_groups]
    
    return train_idx, test_idx

def main():
    device = configure_gpu()
    config = load_config()
    proc_dir = config['data']['processed_dir']
    
    X_path = os.path.join(proc_dir, "X_features.npy")
    y_path = os.path.join(proc_dir, "y_labels.npy")
    
    if not os.path.exists(X_path):
        logging.error("Processed data not found. Run preprocessing.py first.")
        return
        
    X = np.load(X_path)
    y_labels = np.load(y_path)
    
    classes = config['data']['classes']
    class_map = {cls: i for i, cls in enumerate(classes)}
    y = np.array([class_map.get(lbl, 0) for lbl in y_labels])
    
    groups = np.arange(len(y)) // 100
    
    train_idx, test_idx = group_split(X, y, groups, test_size=0.15, random_state=42)
    train_idx, val_idx = group_split(X[train_idx], y[train_idx], np.array(groups)[train_idx], test_size=0.17, random_state=42)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    logging.info(f"Train/Val/Test sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    
    if len(X_train) == 0:
         logging.error("No training data. Pipeline cannot proceed.")
         return

    logging.info("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    try:
        result = smote.fit_resample(X_train, y_train)
        X_train_sm, y_train_sm = result[0], result[1]
    except ValueError:
        logging.warning("SMOTE failed due to too few samples. Proceeding without SMOTE.")
        X_train_sm, y_train_sm = X_train, y_train
        
    X_train_aug, y_train_aug = augment_data(X_train_sm, y_train_sm)
    
    gmm, threshold = train_gmm(X_train_aug, X_val, y_val)
    
    logging.info("Training 1D-CNN (PyTorch)...")
    # Reshape features to (batch, channels, seq_len) for PyTorch Conv1d
    X_train_cnn = np.expand_dims(X_train_aug, axis=1)
    X_val_cnn = np.expand_dims(X_val, axis=1)
    
    train_dataset = FaultDataset(X_train_cnn, y_train_aug)
    val_dataset = FaultDataset(X_val_cnn, y_val)
    
    batch_size = config['training'].get('batch_size', 32)
    # Optimize batch size for GPU: use larger batches for better throughput
    if device == 'cuda':
        batch_size = 128  # Force larger batch size for GPU
        logging.info(f"GPU detected: using batch size {batch_size} for maximum throughput")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    
    model = CNNDetector(num_classes=len(classes), input_size=config['dsp']['num_features']).to(device)
    criterion = nn.CrossEntropyLoss()
    # Use SGD with momentum for better generalization
    optimizer = optim.SGD(model.parameters(), lr=config['training'].get('learning_rate', 0.01), momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=1e-6)
    
    epochs = config['training']['epochs']
    patience = config['training'].get('patience', 10)
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Use non_blocking=True for asynchronous GPU transfer
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        scheduler.step()
        
        # Log GPU memory usage if CUDA is available
        if device == 'cuda':
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
            logging.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | GPU Mem: {gpu_mem_allocated:.2f}/{gpu_mem_reserved:.2f}GB")
        else:
            logging.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'model_best.pth')
            logging.info("Saved new best model.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    logging.info("Training complete. Exporting parameters...")

if __name__ == "__main__":
    main()
