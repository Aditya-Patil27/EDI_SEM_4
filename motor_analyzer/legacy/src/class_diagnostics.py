"""
Class-level diagnostic and confusion analysis tool.
Run this after training to understand which classes are being confused.
"""

import os
import yaml
import logging
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from src.train import CNNDetector
from src.gpu_setup import configure_gpu

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def analyze_class_confusion(X, y, y_labels, classes, model_path='model_best.pth'):
    """
    Analyze which classes are being confused by the model.
    
    Args:
        X: Feature matrix (N, num_features)
        y: Integer labels (N,)
        y_labels: Original string labels (N,)
        classes: List of class names
        model_path: Path to trained model
    """
    device = configure_gpu()
    config = load_config()
    
    logging.info("=" * 70)
    logging.info("CLASS CONFUSION ANALYSIS")
    logging.info("=" * 70)
    
    # Load model
    try:
        model = CNNDetector(num_classes=len(classes), input_size=config['dsp']['num_features'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Could not load model: {e}")
        return
    
    # Prepare data
    X_expanded = np.expand_dims(X, axis=1)
    batch_size = 4096
    y_pred_proba_list = []
    
    logging.info(f"Evaluating {len(X)} samples...")
    with torch.no_grad():
        for i in range(0, len(X_expanded), batch_size):
            batch = torch.tensor(X_expanded[i:i+batch_size], dtype=torch.float32).to(device)
            outputs = model(batch)
            proba = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            y_pred_proba_list.append(proba)
    
    y_pred_proba = np.concatenate(y_pred_proba_list, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred_conf = np.max(y_pred_proba, axis=1)
    
    logging.info("\n" + "=" * 70)
    logging.info("OVERALL METRICS")
    logging.info("=" * 70)
    
    # Overall accuracy
    overall_acc = np.mean(y_pred == y)
    logging.info(f"Overall Accuracy: {overall_acc:.4f} ({100*overall_acc:.2f}%)")
    
    # Per-class accuracy
    logging.info("\n" + "=" * 70)
    logging.info("PER-CLASS ACCURACIES")
    logging.info("=" * 70)
    
    per_class_acc = {}
    for class_idx, class_name in enumerate(classes):
        mask = y == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y[mask])
            per_class_acc[class_name] = class_acc
            n_samples = np.sum(mask)
            n_correct = np.sum(y_pred[mask] == y[mask])
            logging.info(f"{class_name:20s}: {class_acc:.4f} ({100*class_acc:.2f}%) | {n_correct}/{n_samples} correct")
    
    # Confusion matrix
    logging.info("\n" + "=" * 70)
    logging.info("CONFUSION MATRIX")
    logging.info("=" * 70)
    
    cm = confusion_matrix(y, y_pred, labels=list(range(len(classes))))
    
    # Normalized confusion matrix for better visibility
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    # Print header
    header = "Predicted ->"
    for cls in classes:
        header += f" {cls:12s}"
    logging.info(header)
    logging.info("-" * len(header))
    
    # Print each row
    for i, true_class in enumerate(classes):
        row = f"{true_class:12s} |"
        for j in range(len(classes)):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            row += f" {pct:11.2%}"
        row += f" | n={cm[i].sum()}"
        logging.info(row)
    
    # Detailed misclassification analysis
    logging.info("\n" + "=" * 70)
    logging.info("DETAILED MISCLASSIFICATION ANALYSIS")
    logging.info("=" * 70)
    
    misclassified_mask = y_pred != y
    n_misclassified = np.sum(misclassified_mask)
    logging.info(f"Total Misclassified: {n_misclassified}/{len(y)} ({100*n_misclassified/len(y):.2f}%)")
    
    # Initialize sorted_pairs to avoid unbounded variable error
    sorted_pairs = []
    
    if n_misclassified > 0:
        logging.info("\nMisclassification Pairs (True -> Predicted):")
        logging.info("-" * 70)
        
        # Count each type of misclassification
        misclass_pairs = {}
        for true_idx, pred_idx in zip(y[misclassified_mask], y_pred[misclassified_mask]):
            pair = f"{classes[true_idx]} -> {classes[pred_idx]}"
            misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)
        for pair, count in sorted_pairs:
            logging.info(f"  {pair:35s}: {count:4d} times ({100*count/n_misclassified:5.1f}%)")
    
    # Confidence analysis
    logging.info("\n" + "=" * 70)
    logging.info("CONFIDENCE ANALYSIS")
    logging.info("=" * 70)
    
    correct_mask = y_pred == y
    if np.sum(correct_mask) > 0:
        avg_conf_correct = np.mean(y_pred_conf[correct_mask])
        logging.info(f"Avg Confidence (Correct): {avg_conf_correct:.4f}")
    
    if np.sum(misclassified_mask) > 0:
        avg_conf_incorrect = np.mean(y_pred_conf[misclassified_mask])
        logging.info(f"Avg Confidence (Incorrect): {avg_conf_incorrect:.4f}")
    
    # Per-class confidence
    logging.info("\nPer-Class Prediction Confidence:")
    logging.info("-" * 70)
    for class_idx, class_name in enumerate(classes):
        mask = y == class_idx
        if np.sum(mask) > 0:
            avg_conf = np.mean(y_pred_conf[mask])
            logging.info(f"{class_name:20s}: {avg_conf:.4f}")
    
    # Classification report
    logging.info("\n" + "=" * 70)
    logging.info("DETAILED CLASSIFICATION REPORT")
    logging.info("=" * 70)
    
    report = classification_report(y, y_pred, labels=list(range(len(classes))), 
                                   target_names=classes, digits=4)
    logging.info("\n" + report)
    
    # ROC-AUC for multi-class (one-vs-rest)
    if len(classes) > 2:
        try:
            y_bin = label_binarize(y, classes=list(range(len(classes))))
            roc_auc = roc_auc_score(y_bin, y_pred_proba, multi_class='ovr', average='weighted')
            logging.info(f"Weighted ROC-AUC (One-vs-Rest): {roc_auc:.4f}")
        except Exception as e:
            logging.warning(f"Could not compute ROC-AUC: {e}")
    
    logging.info("\n" + "=" * 70)
    logging.info("RECOMMENDATIONS")
    logging.info("=" * 70)
    
    # Identify problem areas
    problems = []
    for class_name, acc in per_class_acc.items():
        if acc < 0.90:
            problems.append(f"  • {class_name}: Low accuracy ({100*acc:.1f}%) - may need more training or better features")
    
    if len(problems) > 0:
        logging.info("Areas to improve:")
        for p in problems:
            logging.info(p)
    else:
        logging.info("Model performance is good across all classes!")
    
    # Check for confused pairs
    high_confusion_pairs = [(pair, count) for pair, count in sorted_pairs if count > 5]
    if len(high_confusion_pairs) > 0:
        logging.info("\nHigh confusion between class pairs:")
        for pair, count in high_confusion_pairs[:3]:
            logging.info(f"  • {pair}: {count} errors - consider data augmentation or feature engineering")

if __name__ == "__main__":
    config = load_config()
    proc_dir = config['data']['processed_dir']
    classes = config['data']['classes']
    
    X_path = os.path.join(proc_dir, "X_features.npy")
    y_path = os.path.join(proc_dir, "y_labels.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        logging.error("Processed data not found. Run preprocessing.py first.")
        exit(1)
    
    X = np.load(X_path)
    y_labels = np.load(y_path)
    
    class_map = {cls: i for i, cls in enumerate(classes)}
    y = np.array([class_map.get(lbl, 0) for lbl in y_labels])
    
    logging.info(f"Loaded {len(X)} samples, {len(classes)} classes")
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Classes: {classes}")
    
    analyze_class_confusion(X, y, y_labels, classes)
