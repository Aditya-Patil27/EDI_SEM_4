"""
Class-level diagnostic and confusion analysis tool.
"""

import os
import yaml
import logging
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from models.detector import CNNDetector
from utils.gpu import configure_gpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def analyze_confusion(X, y, y_labels, classes, model_path="model_best.pth"):
    device = configure_gpu()
    config = load_config()

    logging.info("=" * 70)
    logging.info("CLASS CONFUSION ANALYSIS")
    logging.info("=" * 70)

    try:
        model = CNNDetector(num_classes=len(classes), input_size=config.get("dsp", {}).get("num_features", 60))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Could not load model: {e}")
        return

    X_expanded = np.expand_dims(X, axis=1)
    batch_size = 4096
    y_pred_proba_list = []

    with torch.no_grad():
        for i in range(0, len(X_expanded), batch_size):
            batch = torch.tensor(X_expanded[i:i+batch_size], dtype=torch.float32).to(device)
            outputs = model(batch)
            proba = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            y_pred_proba_list.append(proba)

    y_pred_proba = np.concatenate(y_pred_proba_list, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred_conf = np.max(y_pred_proba, axis=1)

    overall_acc = np.mean(y_pred == y)
    logging.info(f"Overall Accuracy: {overall_acc:.4f}")

    logging.info("\nPer-Class Accuracies:")
    for class_idx, class_name in enumerate(classes):
        mask = y == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y[mask])
            n_correct = np.sum(y_pred[mask] == y[mask])
            logging.info(f"  {class_name:20s}: {class_acc:.4f} ({n_correct}/{np.sum(mask)} correct)")

    cm = confusion_matrix(y, y_pred, labels=list(range(len(classes))))
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)
    logging.info("\nConfusion Matrix (normalized):")
    for i, name in enumerate(classes):
        row = "  ".join(f"{cm_norm[i,j]:.2%}" for j in range(len(classes)))
        logging.info(f"  {name:15s} | {row}")

    mis_mask = y_pred != y
    n_mis = np.sum(mis_mask)
    if n_mis > 0:
        pairs = {}
        for ti, pi in zip(y[mis_mask], y_pred[mis_mask]):
            p = f"{classes[ti]} -> {classes[pi]}"
            pairs[p] = pairs.get(p, 0) + 1
        logging.info(f"\nMisclassifications ({n_mis}/{len(y)}):")
        for pair, count in sorted(pairs.items(), key=lambda x: -x[1])[:5]:
            logging.info(f"  {pair:35s}: {count}")

    correct_mask = y_pred == y
    if np.sum(correct_mask) > 0:
        logging.info(f"\nAvg confidence (correct):   {np.mean(y_pred_conf[correct_mask]):.4f}")
    if np.sum(mis_mask) > 0:
        logging.info(f"Avg confidence (incorrect): {np.mean(y_pred_conf[mis_mask]):.4f}")

    report = classification_report(y, y_pred, labels=list(range(len(classes))), target_names=classes, digits=4)
    logging.info(f"\n{report}")

    if len(classes) > 2:
        y_bin = label_binarize(y, classes=list(range(len(classes))))
        try:
            auc = roc_auc_score(y_bin, y_pred_proba, multi_class="ovr", average="weighted")
            logging.info(f"Weighted ROC-AUC: {auc:.4f}")
        except Exception as e:
            logging.warning(f"ROC-AUC failed: {e}")
