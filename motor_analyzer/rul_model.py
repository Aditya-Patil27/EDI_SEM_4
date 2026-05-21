"""
RUL (Remaining Useful Life) prediction model for run-to-failure bearing data.
Trains a regression model on NASA IMS full degradation sequences.
"""

import os
import re
import json
import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from feature_pipeline import extract_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMS_FS = 20000
TARGET_FS = 100
WINDOW_SIZE = 128

RUL_MODEL_PATH = os.path.join(MODELS_DIR, 'rul_predictor.pkl')
MAX_RUL_SAMPLES = 5000


class RULPredictor:
    """Regression model that predicts remaining useful life from vibration features."""

    def __init__(self, model_type='rf'):
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
            )
        elif model_type == 'gbr':
            self.model = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, features):
        if not self.trained:
            return 0.5
        features = np.array(features).reshape(1, -1)
        return float(self.model.predict(features)[0])

    def save(self, path=None):
        path = path or RUL_MODEL_PATH
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'trained': self.trained}, f)
        log.info(f"RULPredictor saved -> {path}")

    @staticmethod
    def load(path=None):
        path = path or RUL_MODEL_PATH
        if not os.path.exists(path):
            log.warning(f"RULPredictor not found at {path}")
            return RULPredictor()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        p = RULPredictor()
        p.model = data['model']
        p.trained = data['trained']
        log.info(f"RULPredictor loaded <- {path}")
        return p


def _resample_to_fs(data, orig_fs, target_fs=TARGET_FS):
    from scipy import signal as scipy_signal
    duration = len(data) / orig_fs
    target_len = int(round(duration * target_fs))
    if target_len < 4:
        return np.array([])
    return scipy_signal.resample(data, target_len)


def _features_from_segment(segment, fs=TARGET_FS):
    baseline = float(np.mean(segment))
    return extract_features(segment.tolist(), baseline, fs)


def _parse_nasa_ims_file(dir_path, fname):
    """Parse a single NASA IMS text file and return vibration signal (channel 0)."""
    fpath = os.path.join(dir_path, fname)
    with open(fpath, 'r') as f:
        content = f.read()
    floats = re.findall(r'-?\d+\.?\d*', content)
    if len(floats) < 4:
        return None
    data = np.array([float(x) for x in floats], dtype=np.float64)
    n_cols = 4
    usable = (len(data) // n_cols) * n_cols
    data = data[:usable]
    if data.size == 0:
        return None
    data = data.reshape(-1, n_cols)
    return data[:, 0]


def _load_nasa_ims_test(test_dir, expected_fault, max_files=1000):
    """Load a full NASA IMS test with RUL labels.
    
    RUL label = file_position / total_files (0 = imminent failure, 1 = brand new).
    Each file is one 2-second reading → extract features directly (no windowing).
    """
    if not os.path.isdir(test_dir):
        log.warning(f"  Test dir not found: {test_dir}, skipping")
        return []

    files = sorted([f for f in os.listdir(test_dir)
                    if os.path.isfile(os.path.join(test_dir, f))])
    if not files:
        return []

    # Subsample to keep runtime tractable
    if len(files) > max_files:
        idx = np.linspace(0, len(files) - 1, max_files, dtype=int)
        files = [files[i] for i in idx]
        log.info(f"  {os.path.basename(test_dir)}: subsampled {max_files} files")

    log.info(f"  {os.path.basename(test_dir)}: {len(files)} files")
    test_name = os.path.basename(os.path.dirname(test_dir))
    ds_name = f"NASA_IMS/{test_name}"

    records = []
    n = len(files)
    for i, fname in enumerate(files):
        sig = _parse_nasa_ims_file(test_dir, fname)
        if sig is None or len(sig) < 100:
            continue
        sig = _resample_to_fs(sig, IMS_FS)
        if len(sig) < 50:
            continue
        rul = (n - i) / n
        feats = _features_from_segment(sig)
        records.append({
            'features': feats,
            'rul': rul,
            'class': 'Normal' if rul > 0.5 else expected_fault,
            'dataset': ds_name,
            'file_index': i,
            'file_count': n,
        })

    if records:
        log.info(f"    -> {len(records)} samples, RUL range [{records[-1]['rul']:.3f}, {records[0]['rul']:.3f}]")
    else:
        log.warning(f"    -> No valid samples from {len(files)} files")
    return records


def prepare_rul_dataset():
    """Load all NASA IMS tests with RUL labels and return feature matrix + RUL targets."""
    ims_base = os.path.join(RAW_DIR, 'nasa_ims')
    if not os.path.exists(ims_base):
        log.warning("NASA IMS directory not found")
        return None, None, None

    TESTS = [
        {'dir': '1st_test/1st_test', 'fault': 'OuterRace'},
        {'dir': '2nd_test/2nd_test', 'fault': 'OuterRace'},
        {'dir': '3rd_test/4th_test/txt', 'fault': 'InnerRace'},
    ]

    all_records = []
    for test in TESTS:
        test_dir = os.path.join(ims_base, test['dir'])
        records = _load_nasa_ims_test(test_dir, test['fault'])
        all_records.extend(records)

    if not all_records:
        return None, None, None

    if len(all_records) > MAX_RUL_SAMPLES:
        rng = np.random.default_rng(42)
        all_records = list(rng.choice(all_records, MAX_RUL_SAMPLES, replace=False))
        log.info(f"  Subsampled to {MAX_RUL_SAMPLES} records")

    X = np.array([r['features'] for r in all_records])
    y = np.array([r['rul'] for r in all_records])
    meta = {
        'n_samples': len(X),
        'feature_dim': X.shape[1],
        'test_sources': list(set(r['dataset'] for r in all_records)),
    }

    log.info(f"RUL dataset: {len(X)} samples, {meta['feature_dim']} features")
    return X, y, meta


def train_rul_model(model_type='rf'):
    """Full pipeline: load data, train RULPredictor, evaluate, save."""
    X, y, meta = prepare_rul_dataset()
    if X is None:
        log.error("No RUL data available")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RULPredictor(model_type=model_type)
    model.train(X_train, y_train)

    y_pred = model.model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    log.info(f"RUL model evaluation:")
    log.info(f"  MAE:  {mae:.4f} (avg RUL error)")
    log.info(f"  R²:   {r2:.4f}")
    log.info(f"  MAE%: {mae * 100:.1f}% of full life")

    results = {
        'mae': round(mae, 4),
        'r2': round(r2, 4),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'model_type': model_type,
    }
    results_path = os.path.join(MODELS_DIR, 'rul_report.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved -> {results_path}")

    model.save()
    return model


if __name__ == '__main__':
    train_rul_model()
