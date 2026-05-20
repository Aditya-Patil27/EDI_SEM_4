"""
End-to-end evaluation pipeline for the motor analyzer system.
Tests feature extraction, company classification, and anomaly detection
against synthetic ground-truth data. Generates a model card.

Usage:
  python evaluate_pipeline.py                          # Full eval
  python evaluate_pipeline.py --quick                  # Quick smoke test
  python evaluate_pipeline.py --company                # Test company classifier only
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_synthetic_data import (
    generate_companies_dataset, generate_fault_dataset,
    normal_vibration, unbalanced_vibration, bearing_fault_vibration
)
from feature_pipeline import extract_features
from ml_models import EnsembleAnomalyModel, CompanyClassifier, TransferLearningAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def evaluate_feature_extraction():
    """Verify features correctly distinguish different fault types."""
    logging.info("=== Evaluating Feature Extraction ===")

    SAMPLE_RATE = 100
    WINDOW = 128
    results = {}

    classes = {
        'normal': normal_vibration,
        'unbalanced': unbalanced_vibration,
        'bearing_fault': bearing_fault_vibration,
    }

    all_feats = {}
    for name, gen_fn in classes.items():
        windows = []
        raw = gen_fn(10.0)
        for start in range(0, len(raw) - WINDOW, WINDOW):
            chunk = raw[start:start + WINDOW]
            feats = extract_features(chunk, np.mean(chunk), SAMPLE_RATE)
            windows.append(feats)
        arr = np.array(windows)
        all_feats[name] = arr
        avg_rms = float(np.mean(arr[:, 0])) if len(arr) > 0 else 0.0
        logging.info(f"  {name}: {len(arr)} windows, avg RMS={avg_rms:.4f}")

    # Separation score: RMS-based distinguishability
    rms_normal = np.mean(all_feats['normal'][:, 0])
    rms_unbal = np.mean(all_feats['unbalanced'][:, 0])
    rms_bearing = np.mean(all_feats['bearing_fault'][:, 0])

    results['rms_separation'] = {
        'normal_mean_rms': float(round(rms_normal, 4)),
        'unbalanced_mean_rms': float(round(rms_unbal, 4)),
        'bearing_mean_rms': float(round(rms_bearing, 4)),
        'normal_vs_unbalanced_ratio': float(round(rms_unbal / (rms_normal + 1e-8), 2)),
        'normal_vs_bearing_ratio': float(round(rms_bearing / (rms_normal + 1e-8), 2)),
    }

    # All features should be finite
    all_valid = all(np.all(np.isfinite(v)) for v in all_feats.values())
    results['all_features_finite'] = all_valid
    logging.info(f"  All features finite: {all_valid}")

    return results


def evaluate_company_classifier(overlap=0.0):
    """Train company classifier on synthetic data with controlled overlap.
    
    Args:
        overlap: 0=well-separated, 0.5=moderate, 1.0=heavy overlap.
    """
    label = f"OVERLAP={overlap:.1f}"
    logging.info(f"=== Evaluating Company Classifier ({label}) ===")

    X, y, companies = generate_companies_dataset(n_companies=3, samples_per_company=40,
                                                  seed=42, overlap=overlap)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = CompanyClassifier(feature_dim=X.shape[1], num_companies=len(companies))
    clf.train(X_train, y_train, company_names=companies)

    predictions = []
    for i in range(len(X_test)):
        idx, name, conf = clf.predict(X_test[i])
        predictions.append(idx)

    acc = accuracy_score(y_test, predictions)
    logging.info(f"  Accuracy ({label}): {acc:.4f}")

    report = classification_report(y_test, predictions, target_names=companies, output_dict=True)
    logging.info(f"  F1 per company: { {c: round(report[c]['f1-score'], 3) for c in companies} }")

    return {
        'accuracy': round(acc, 4),
        'num_companies': len(companies),
        'num_samples': len(X),
        'classification_report': report,
        'overlap': overlap,
    }


def evaluate_company_classifier_sweep():
    """Sweep overlap values to show accuracy curve — realistic benchmark."""
    logging.info("=== Company Classifier Overlap Sweep (Realistic Benchmark) ===")
    results = {}
    for overlap in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = evaluate_company_classifier(overlap=overlap)
        results[f"overlap_{overlap:.1f}"] = r
        logging.info(f"  >> overlap={overlap:.1f}: accuracy={r['accuracy']:.4f}")
    # Also evaluate on real datasets to show it's trivial
    logging.info("")
    logging.info("=== Company Classifier on REAL Datasets (CWRU+JNU+Synthetic) ===")
    eval_real_company_classifier()

    # Check for trained transfer model
    transfer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'models', 'company_classifier_transfer.pth')
    if os.path.exists(transfer_path):
        logging.info("=== Transfer Learning Adapter Evaluation ===")
        eval_transfer_classifier(transfer_path)
    else:
        logging.info("(No transfer model found at models/company_classifier_transfer.pth)")
        logging.info("  Train with: python train_pipeline.py --transfer")

    return results


def eval_transfer_classifier(model_path):
    """Evaluate saved TransferLearningAdapter on real dataset."""
    import json
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
    X = np.load(os.path.join(data_dir, 'X_all.npy'))
    company_labels = np.load(os.path.join(data_dir, 'company_labels.npy'))
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        meta = json.load(f)
    company_names = meta['companies']

    from sklearn.model_selection import train_test_split
    _, X_te, _, y_te = train_test_split(X, company_labels, test_size=0.2, random_state=42, stratify=company_labels)

    adapter = TransferLearningAdapter.load(model_path)
    if not adapter.trained:
        logging.warning("  Transfer adapter not trained, skipping")
        return

    preds = []
    for i in range(len(X_te)):
        idx, name, conf = adapter.predict(X_te[i])
        preds.append(idx)

    acc = accuracy_score(y_te, preds)
    report = classification_report(y_te, preds, target_names=company_names, output_dict=True)
    logging.info(f"  Accuracy: {acc:.4f}")
    logging.info(f"  F1: { {c: round(report[c]['f1-score'], 3) for c in company_names} }")
    logging.info(f"  NOTE: Uses TransferLearningAdapter with 28->39 expansion + pretrained CNN")


def eval_real_company_classifier():
    """Evaluate on the real CWRU/JNU/Synthetic data — trivially 100%."""
    import json
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
    X = np.load(os.path.join(data_dir, 'X_all.npy'))
    company_labels = np.load(os.path.join(data_dir, 'company_labels.npy'))
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        meta = json.load(f)
    company_names = meta['companies']

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, company_labels, test_size=0.2, random_state=42, stratify=company_labels)

    clf = CompanyClassifier(feature_dim=X.shape[1], num_companies=len(company_names))
    clf.train(X_tr, y_tr, company_names=company_names)

    preds = []
    for i in range(len(X_te)):
        idx, name, conf = clf.predict(X_te[i])
        preds.append(idx)

    acc = accuracy_score(y_te, preds)
    report = classification_report(y_te, preds, target_names=company_names, output_dict=True)
    logging.info(f"  Accuracy: {acc:.4f}")
    logging.info(f"  F1: { {c: round(report[c]['f1-score'], 3) for c in company_names} }")
    logging.info(f"  NOTE: 100% expected — datasets are from different research labs,")
    logging.info(f"  not real companies. This is NOT representative of real-world performance.")


def evaluate_anomaly_detection():
    """Evaluate how well the anomaly detector separates normal vs faulty."""
    logging.info("=== Evaluating Anomaly Detection ===")

    SAMPLE_RATE = 100
    WINDOW = 128

    # Generate normal training data
    np.random.seed(42)
    X_normal = []
    for _ in range(30):
        raw = normal_vibration(5.0)
        for start in range(0, len(raw) - WINDOW, WINDOW):
            chunk = raw[start:start + WINDOW]
            feats = extract_features(chunk, np.mean(chunk), SAMPLE_RATE)
            X_normal.append(feats)

    # Generate test data (mix of normal + faults)
    X_test = []
    y_test = []
    for label, gen_fn in [('normal', normal_vibration), ('fault', unbalanced_vibration),
                          ('fault', bearing_fault_vibration)]:
        for _ in range(10):
            raw = gen_fn(5.0)
            for start in range(0, len(raw) - WINDOW, WINDOW):
                chunk = raw[start:start + WINDOW]
                feats = extract_features(chunk, np.mean(chunk), SAMPLE_RATE)
                X_test.append(feats)
                y_test.append(0 if label == 'normal' else 1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = EnsembleAnomalyModel()
    model.fit(X_normal)

    y_pred = []
    y_score = []
    for i in range(len(X_test)):
        is_anomaly, score = model.predict_score(X_test[i])
        y_pred.append(1 if is_anomaly else 0)
        y_score.append(score)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logging.info(f"  Accuracy:  {acc:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall:    {recall:.4f}")
    logging.info(f"  F1 Score:  {f1:.4f}")

    return {
        'accuracy': round(acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'test_samples': len(X_test),
        'anomaly_ratio': float(round(np.mean(y_pred), 4)),
    }


def evaluate_live_simulation():
    """Simulate a live streaming session and measure end-to-end latency."""
    logging.info("=== Evaluating Live Simulation ===")

    SAMPLE_RATE = 100
    WINDOW = 128
    num_points = 2000

    # Generate streaming data: 10s normal + 5s unbalanced + 5s bearing
    raw = np.concatenate([
        normal_vibration(10.0),
        unbalanced_vibration(5.0),
        bearing_fault_vibration(5.0),
    ])[:num_points]

    # Train model on first portion (normal only)
    baseline = np.mean(raw[:500])
    normal_feats = []
    for start in range(0, 500 - WINDOW, WINDOW):
        chunk = raw[start:start + WINDOW]
        normal_feats.append(extract_features(chunk, baseline, SAMPLE_RATE))

    model = EnsembleAnomalyModel()
    model.fit(normal_feats)

    # Simulate streaming
    buffer = []
    anomaly_count = 0
    total_windows = 0
    latencies = []
    t0 = time.time()

    for i in range(len(raw)):
        buffer.append(raw[i])
        if len(buffer) >= WINDOW:
            chunk = buffer[-WINDOW:]
            t1 = time.time()
            feats = extract_features(chunk, baseline, SAMPLE_RATE)
            is_anomaly, score = model.predict_score(feats)
            latencies.append((time.time() - t1) * 1000)  # ms
            total_windows += 1
            if is_anomaly:
                anomaly_count += 1

    elapsed = time.time() - t0
    results = {
        'total_windows': total_windows,
        'anomaly_windows': anomaly_count,
        'anomaly_rate': round(anomaly_count / max(total_windows, 1), 4),
        'avg_inference_ms': round(float(np.mean(latencies)), 2),
        'p99_inference_ms': round(float(np.percentile(latencies, 99)), 2),
        'total_simulation_sec': round(elapsed, 3),
    }

    logging.info(f"  Total windows: {total_windows}")
    logging.info(f"  Anomalies detected: {anomaly_count}")
    logging.info(f"  Avg inference: {results['avg_inference_ms']}ms")
    logging.info(f"  P99 inference: {results['p99_inference_ms']}ms")

    return results


def generate_model_card(all_results: dict):
    """Generate a model card JSON (similar to ml repo's model_card.json)."""
    card = {
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system': 'MotorSense Evaluation Pipeline',
        'feature_dimension': 28,
        'sample_rate_hz': 100,
        'components': all_results,
        'verdict': 'PASS' if all_results.get('features', {}).get('all_features_finite', False) else 'INCOMPLETE',
    }
    with open('evaluation_report.json', 'w') as f:
        json.dump(card, f, indent=2)
    logging.info(f"\nModel card saved to evaluation_report.json")
    return card


def main():
    parser = argparse.ArgumentParser(description='MotorSense Evaluation Pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick smoke test only')
    parser.add_argument('--company', action='store_true', help='Company classifier only')
    parser.add_argument('--fault', action='store_true', help='Anomaly detection only')
    parser.add_argument('--live', action='store_true', help='Live simulation only')
    parser.add_argument('--transfer', action='store_true', help='Evaluate transfer model on real data')
    args = parser.parse_args()

    all_results = {}

    if args.transfer:
        logging.info("=== TRANSFER MODEL EVALUATION ===")
        transfer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'models', 'company_classifier_transfer.pth')
        if os.path.exists(transfer_path):
            all_results['transfer'] = eval_transfer_classifier(transfer_path)
        else:
            logging.error("No transfer model found. Train with: python train_pipeline.py --transfer")
            return
    elif args.quick:
        logging.info("=== QUICK SMOKE TEST ===")
        all_results['features'] = evaluate_feature_extraction()
        all_results['company_benchmark'] = evaluate_company_classifier_sweep()
    elif args.company:
        all_results['company_benchmark'] = evaluate_company_classifier_sweep()
    elif args.fault:
        all_results['anomaly'] = evaluate_anomaly_detection()
    elif args.live:
        all_results['live'] = evaluate_live_simulation()
    else:
        logging.info("=== FULL EVALUATION ===")
        all_results['features'] = evaluate_feature_extraction()
        all_results['company_benchmark'] = evaluate_company_classifier_sweep()
        all_results['anomaly'] = evaluate_anomaly_detection()
        all_results['live'] = evaluate_live_simulation()

    card = generate_model_card(all_results)
    print(f"\nEvaluation complete. Verdict: {card['verdict']}")


if __name__ == '__main__':
    main()
