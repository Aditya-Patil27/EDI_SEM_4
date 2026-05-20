"""
Modular data ingestion — adapters for CWRU, JNU, and Synthetic datasets.
Each adapter produces consistent dicts with raw signals + metadata.
Top-level function ingests all, normalizes, and returns feature vectors.
"""

import os
import re
import json
import yaml
import logging
import numpy as np
from scipy import signal as scipy_signal
from collections import defaultdict

from feature_pipeline import extract_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')

WINDOW_SIZE = 128
TARGET_FS = 100

CLASS_MAP_CWRU = {
    'Normal': 'Normal',
    'IR': 'InnerRace',
    'OR': 'OuterRace',
    'B': 'BallFault',
}

CLASS_MAP_JNU = {
    'n': 'Normal',
    'ib': 'InnerRace',
    'ob': 'OuterRace',
    'tb': 'BallFault',
}


def _resample_to_100hz(data, original_fs):
    duration = len(data) / original_fs
    target_len = int(round(duration * TARGET_FS))
    if target_len < 4:
        return np.array([])
    return scipy_signal.resample(data, target_len)


def _segment_into_windows(data, window_size=WINDOW_SIZE, step=None):
    if step is None:
        step = window_size // 2
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start:start + window_size])
    return windows


def _features_from_segment(segment, fs=TARGET_FS):
    baseline = float(np.mean(segment))
    return extract_features(segment.tolist(), baseline, fs)


def _process_cwru():
    """Ingest CWRU dataset from GitHub mirror structure."""
    log.info("Processing CWRU dataset...")
    cwru_dir = os.path.join(RAW_DIR, 'cwru')
    if not os.path.exists(cwru_dir):
        log.warning("CWRU directory not found, skipping")
        return []

    records = []
    import scipy.io

    for root, dirs, files in os.walk(cwru_dir):
        # Determine sample rate from parent folder name
        rel = os.path.relpath(root, cwru_dir)
        parts = rel.replace('\\', '/').split('/')

        fs = None
        for p in parts:
            pl = p.lower()
            if '12k' in pl:
                fs = 12000
                break
            elif '48k' in pl:
                fs = 48000
                break

        fault_type = None
        for p in parts:
            if p in CLASS_MAP_CWRU:
                fault_type = CLASS_MAP_CWRU[p]
                break

        for fname in files:
            if not fname.endswith('.mat'):
                continue
            fpath = os.path.join(root, fname)
            try:
                data = scipy.io.loadmat(fpath)
            except Exception as e:
                log.error(f"  Failed to load {fname}: {e}")
                continue

            # Find DE_time key
            de_key = None
            for k in data:
                if k.endswith('_DE_time'):
                    de_key = k
                    break
            if de_key is None:
                continue

            sig = data[de_key][:, 0].astype(np.float64)

            if fs is None:
                # infer from file size: Normal files
                fs = 12000 if len(sig) < 200000 else 48000

            sig_100 = _resample_to_100hz(sig, fs)
            if len(sig_100) < WINDOW_SIZE:
                continue

            windows = _segment_into_windows(sig_100)
            for win in windows:
                feats = _features_from_segment(win)
                records.append({
                    'filepath': fname,
                    'raw_class': fault_type or 'Unknown',
                    'class': fault_type or 'Unknown',
                    'dataset': 'CWRU',
                    'format': 'mat',
                    'features': feats,
                })

    log.info(f"  CWRU: {len(records)} windows processed")
    return records


def _process_jnu():
    """Ingest JNU bearing dataset from CSV files."""
    log.info("Processing JNU dataset...")
    jnu_dir = os.path.join(RAW_DIR, 'jnu')
    if not os.path.exists(jnu_dir):
        log.warning("JNU directory not found, skipping")
        return []

    records = []
    JNU_FS = 50000

    for fname in sorted(os.listdir(jnu_dir)):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(jnu_dir, fname)

        # Parse class from filename
        cls = None
        for prefix, mapped in CLASS_MAP_JNU.items():
            if fname.startswith(prefix):
                cls = mapped
                break
        if cls is None:
            log.warning(f"  Unknown class prefix in {fname}, skipping")
            continue

        try:
            sig = np.loadtxt(fpath, dtype=np.float64)
        except Exception as e:
            log.error(f"  Failed to load {fname}: {e}")
            continue

        if len(sig) < 100:
            continue

        sig_100 = _resample_to_100hz(sig, JNU_FS)
        if len(sig_100) < WINDOW_SIZE:
            continue

        windows = _segment_into_windows(sig_100)
        for win in windows:
            feats = _features_from_segment(win)
            records.append({
                'filepath': fname,
                'raw_class': cls,
                'class': cls,
                'dataset': 'JNU',
                'format': 'csv',
                'features': feats,
            })

    log.info(f"  JNU: {len(records)} windows processed")
    return records


def _process_synthetic(n_companies=1, samples_per_company=100):
    """Generate synthetic vibration data for dev/testing."""
    log.info("Processing Synthetic dataset...")
    from generate_synthetic_data import normal_vibration, unbalanced_vibration, bearing_fault_vibration, misalignment_vibration

    records = []
    patterns = [
        ('Normal', normal_vibration),
        ('Unbalanced', unbalanced_vibration),
        ('BearingFault', bearing_fault_vibration),
        ('Misalignment', misalignment_vibration),
    ]

    for cls_name, gen_fn in patterns:
        for _ in range(samples_per_company):
            raw = gen_fn(5.0)
            windows = _segment_into_windows(raw)
            for win in windows:
                feats = _features_from_segment(win)
                records.append({
                    'filepath': f'synthetic_{cls_name}',
                    'raw_class': cls_name,
                    'class': cls_name,
                    'dataset': 'Synthetic',
                    'format': 'generated',
                    'features': feats,
                })

    log.info(f"  Synthetic: {len(records)} windows processed")
    return records


def _compute_rms(records):
    vals = []
    for r in records:
        f = r.get('features', [])
        if f and len(f) > 0:
            vals.append(abs(f[0]))
    return float(np.mean(vals)) if vals else 1.0


def ingest_all_datasets(include_synthetic=True, output_dir=None):
    """Ingest all available datasets and return (features, labels, company_labels)."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    all_records = []

    all_records.extend(_process_cwru())
    all_records.extend(_process_jnu())
    if include_synthetic:
        all_records.extend(_process_synthetic())

    if not all_records:
        log.warning("No data ingested from any dataset!")
        return np.array([]), np.array([]), [], {}

    # Cross-dataset RMS normalization
    datasets = set(r['dataset'] for r in all_records)
    dataset_rms = {}
    for ds in datasets:
        ds_records = [r for r in all_records if r['dataset'] == ds]
        dataset_rms[ds] = _compute_rms(ds_records)
        log.info(f"  RMS({ds}) = {dataset_rms[ds]:.6f}")

    if len(datasets) > 1:
        # Normalize feature[0] (RMS) across datasets
        rms_vals = [v for v in dataset_rms.values() if v > 1e-10]
        if rms_vals:
            target_rms = np.mean(rms_vals)
            for r in all_records:
                ds = r['dataset']
                ds_rms = dataset_rms.get(ds, 1.0)
                if ds_rms > 1e-10:
                    scale = target_rms / ds_rms
                    r['features'] = [f * scale for f in r['features']]

    # Build feature matrix and labels
    X = []
    y = []
    company_labels = []

    # Collect unique company names
    company_names = sorted(set(r['dataset'] for r in all_records))
    company_to_idx = {name: i for i, name in enumerate(company_names)}

    # Collect class names per company
    class_sets = defaultdict(set)
    for r in all_records:
        class_sets[r['dataset']].add(r['class'])

    for r in all_records:
        X.append(r['features'])
        y.append(r['class'])
        company_labels.append(company_to_idx[r['dataset']])

    X = np.array(X)
    y = np.array(y)
    company_labels = np.array(company_labels)

    # Save metadata
    metadata = {
        'n_samples': len(X),
        'feature_dim': X.shape[1] if X.ndim > 1 else 0,
        'companies': company_names,
        'classes_per_company': {k: sorted(v) for k, v in class_sets.items()},
        'dataset_rms': dataset_rms,
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata saved -> {meta_path}")
    log.info(f"Total: {len(X)} windows, {len(company_names)} companies, {len(class_sets)} classes")

    return X, y, company_labels, metadata


def save_company_datasets(X, y, company_labels, metadata, output_dir=None):
    """Split by company and save separate datasets for per-company training."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    company_names = metadata['companies']
    for ci, name in enumerate(company_names):
        mask = company_labels == ci
        if not np.any(mask):
            continue
        Xc = X[mask]
        yc = y[mask]
        np.save(os.path.join(output_dir, f'X_{name}.npy'), Xc)
        np.save(os.path.join(output_dir, f'y_{name}.npy'), yc)
        classes, counts = np.unique(yc, return_counts=True)
        log.info(f"  {name}: {len(Xc)} samples, classes: {dict(zip(classes, counts))}")

    # Full dataset
    np.save(os.path.join(output_dir, 'X_all.npy'), X)
    np.save(os.path.join(output_dir, 'y_all.npy'), y)
    np.save(os.path.join(output_dir, 'company_labels.npy'), company_labels)
    log.info(f"Full dataset saved -> {output_dir}/")
    return output_dir


if __name__ == '__main__':
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    include_synthetic = True
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        include_synthetic = cfg.get('companies', {}).get('use_synthetic', True)
        log.info(f"Config: use_synthetic={include_synthetic}")

    X, y, company_labels, metadata = ingest_all_datasets(include_synthetic=include_synthetic)
    if len(X) > 0:
        save_company_datasets(X, y, company_labels, metadata)
        print(f"\nDone. {len(X)} total windows across {len(metadata['companies'])} companies.")
    else:
        print("No data ingested.")
