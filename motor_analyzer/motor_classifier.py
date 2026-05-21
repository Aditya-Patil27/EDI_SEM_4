"""
Motor Characteristics Classifier — identifies physical motor properties
(HP rating, bearing type, RPM band, fault diameter) from vibration features.
Includes data augmentation, ensemble models, k-fold cross-validation,
and SHAP-based feature importance explanations.
"""

import os
import re
import json
import pickle
import yaml
import logging
import numpy as np
from collections import defaultdict
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from feature_pipeline import extract_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CWRU_DIR = os.path.join(RAW_DIR, 'cwru')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

def _load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}

_cfg = _load_config().get('motor', {})
TARGET_FS = 100
WINDOW_SIZE = 128
AUGMENT_MULTIPLIER = _cfg.get('augment_multiplier', 2)
RPM_SCALE_RANGE = tuple(_cfg.get('rpm_scale_range', [0.85, 1.15]))
NOISE_LEVEL_RANGE = tuple(_cfg.get('noise_level_range', [0.0, 0.12]))
AMP_SCALE_RANGE = tuple(_cfg.get('amplitude_scale_range', [0.7, 1.3]))
N_ESTIMATORS = _cfg.get('n_estimators', 150)
GB_ESTIMATORS = _cfg.get('gb_estimators', 50)
CV_FOLDS = _cfg.get('cv_folds', 3)
MAX_TRAIN_SAMPLES = _cfg.get('max_train_samples', 15000)

def _file_num_to_motor_specs(num):
    if num <= 96:
        return (0, 1797)
    if num <= 99:
        return (num - 97, 1797)
    if num <= 119:
        return (0, 1797)
    if num <= 139:
        return (1, 1772)
    if num <= 159:
        return (2, 1750)
    if num <= 179:
        return (3, 1730)
    if num <= 199:
        return (0, 1797)
    if num <= 299:
        return (0, 1797)
    if num <= 399:
        return (1, 1772)
    return (0, 1797)

def _parse_cwru_path(rel_path):
    parts = rel_path.replace('\\', '/').split('/')
    fname = parts[-1]
    m = re.match(r'(\d+)(?:@(\d+))?_(\d+)\.mat$', fname)
    if not m:
        m = re.match(r'(\d+)_Normal_(\d+)\.mat$', fname)
        if not m:
            return None
    file_num = int(m.group(1))
    hp, rpm = _file_num_to_motor_specs(file_num)
    bearing_loc = 'Normal_Baseline'
    fault_type = 'Normal'
    fault_diameter = None
    for p in parts[:-1]:
        pl = p.lower()
        if 'drive_end' in pl:
            bearing_loc = 'Drive_End'
        elif 'fan_end' in pl:
            bearing_loc = 'Fan_End'
    if 'normal' in parts:
        return {'hp': hp, 'rpm': rpm, 'bearing_location': 'Normal_Baseline',
                'fault_type': 'Normal', 'fault_diameter': None}
    for p in parts:
        if p == 'B':
            fault_type = 'BallFault'
        elif p == 'IR':
            fault_type = 'InnerRace'
        elif p == 'OR':
            fault_type = 'OuterRace'
        elif re.match(r'^\d{3}$', p):
            fault_diameter = int(p)
    return {'hp': hp, 'rpm': rpm, 'bearing_location': bearing_loc,
            'fault_type': fault_type, 'fault_diameter': fault_diameter}

def _load_cwru_signal(file_path):
    import scipy.io
    try:
        data = scipy.io.loadmat(file_path)
    except Exception:
        return None
    de_key = next((k for k in data if k.endswith('_DE_time')), None)
    if de_key is None:
        return None
    return data[de_key][:, 0].astype(np.float64)

def _resample_to_fs(data, orig_fs, target_fs=TARGET_FS):
    duration = len(data) / orig_fs
    target_len = int(round(duration * target_fs))
    if target_len < 4:
        return np.array([])
    return scipy_signal.resample(data, target_len)

def _detect_fs(parts):
    for p in parts:
        if '12k' in p.lower():
            return 12000
        if '48k' in p.lower():
            return 48000
    return 12000

def _features_from_segment(segment, fs=TARGET_FS):
    return extract_features(segment.tolist(), float(np.mean(segment)), fs)

def _segment_into_windows(data, window_size=WINDOW_SIZE, step=None):
    if step is None:
        step = window_size // 2
    return [data[s:s + window_size] for s in range(0, len(data) - window_size + 1, step)]

def _rpm_vary(signal, rpm_scale):
    if abs(rpm_scale - 1.0) < 0.01:
        return signal
    n = len(signal)
    orig_t = np.linspace(0, n / TARGET_FS, n)
    new_n = max(4, int(n * rpm_scale))
    new_t = np.linspace(0, n / TARGET_FS, new_n)
    f = interp1d(orig_t, signal, kind='linear', bounds_error=False, fill_value=0)
    return _resample_to_fs(f(new_t), TARGET_FS * rpm_scale)

def _add_noise(signal, noise_level):
    rms = np.std(signal)
    if rms < 1e-10:
        return signal
    return signal + np.random.normal(0, rms * noise_level, len(signal))

def _scale_amplitude(signal, scale):
    return signal * scale

def _augment_records(records):
    """Generate augmented copies of each record with RPM/noise/amplitude variation."""
    aug_count = 0
    for r in list(records):
        sig = np.array(r.get('_raw_signal'))
        if sig is None or len(sig) < WINDOW_SIZE * 2:
            continue
        for _ in range(AUGMENT_MULTIPLIER):
            rpm_s = np.random.uniform(*RPM_SCALE_RANGE)
            noise_lvl = np.random.uniform(*NOISE_LEVEL_RANGE)
            amp_s = np.random.uniform(*AMP_SCALE_RANGE)
            aug = _rpm_vary(sig, rpm_s)
            aug = _add_noise(aug, noise_lvl)
            aug = _scale_amplitude(aug, amp_s)
            if len(aug) < WINDOW_SIZE:
                continue
            windows = _segment_into_windows(aug)
            for win in windows:
                records.append({
                    'features': _features_from_segment(win),
                    'hp': r['hp'], 'bearing_location': r['bearing_location'],
                    'fault_diameter': r['fault_diameter'], 'rpm': r['rpm'],
                })
            aug_count += 1
    log.info(f"  Augmentation: generated {aug_count} copies ({len(records)} total windows)")
    return records

def _build_ensemble():
    """Build a VotingClassifier ensemble of RF + GB."""
    return VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=20, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(
            n_estimators=GB_ESTIMATORS, max_depth=5,
            learning_rate=0.05, subsample=0.6, random_state=42)),
    ], voting='soft')

class MotorClassifier:
    """Multi-output ensemble classifier with SHAP explanations.

    Trains three independent ensemble VotingClassifiers for:
    - HP rating (0–3 HP)
    - Bearing location (Drive_End / Fan_End / Normal_Baseline)
    - Fault diameter (0, 7, 14, 21, 28 in thousandths of an inch)
    """

    def __init__(self):
        self.hp_model = _build_ensemble()
        self.bearing_model = _build_ensemble()
        self.diameter_model = _build_ensemble()
        self.trained = False

    def train(self, X, y_hp, y_bearing, y_diameter):
        self.hp_model.fit(X, y_hp)
        self.bearing_model.fit(X, y_bearing)
        self.diameter_model.fit(X, y_diameter)
        self.trained = True

    def predict(self, features):
        features = np.array(features).reshape(1, -1)
        hp = int(self.hp_model.predict(features)[0])
        bearing = self.bearing_model.predict(features)[0]
        diameter = int(self.diameter_model.predict(features)[0])
        hp_proba = max(self.hp_model.predict_proba(features)[0])
        bearing_proba = max(self.bearing_model.predict_proba(features)[0])
        diameter_proba = max(self.diameter_model.predict_proba(features)[0])
        return {
            'hp': hp,
            'bearing_location': str(bearing),
            'fault_diameter_inches': diameter / 1000 if diameter > 0 else None,
            'rpm_estimate': self._hp_to_rpm(hp),
            'confidence': {
                'hp': round(float(hp_proba), 3),
                'bearing_location': round(float(bearing_proba), 3),
                'fault_diameter': round(float(diameter_proba), 3),
            },
        }

    def explain(self, features, feature_names=None):
        import shap
        features = np.array(features).reshape(1, -1)
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(features.shape[1])]

        explanations = {}
        for name, model in [('hp', self.hp_model), ('bearing', self.bearing_model),
                             ('diameter', self.diameter_model)]:
            explainer = shap.TreeExplainer(model.named_estimators_['rf'])
            shap_values = explainer.shap_values(features)
            if shap_values.ndim == 3:
                pred_class = model.predict(features)[0]
                classes = list(model.classes_)
                class_idx = classes.index(pred_class) if pred_class in classes else 0
                sv = shap_values[0, :, class_idx]
            elif isinstance(shap_values, list):
                pred_class = model.predict(features)[0]
                classes = list(model.classes_)
                class_idx = classes.index(pred_class) if pred_class in classes else 0
                sv = shap_values[class_idx][0]
            else:
                sv = shap_values[0]
            top_idx = np.argsort(np.abs(sv))[-5:][::-1]
            explanations[name] = [
                {'feature': feature_names[i], 'importance': round(float(sv[i]), 4)}
                for i in top_idx
            ]
        return explanations

    def save(self, path=None):
        path = path or os.path.join(MODELS_DIR, 'motor_classifier.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'hp_model': self.hp_model, 'bearing_model': self.bearing_model,
                'diameter_model': self.diameter_model, 'trained': self.trained,
            }, f)
        log.info(f"MotorClassifier saved -> {path}")

    @staticmethod
    def load(path=None):
        path = path or os.path.join(MODELS_DIR, 'motor_classifier.pkl')
        mc = MotorClassifier()
        if not os.path.exists(path):
            log.warning(f"MotorClassifier not found at {path}")
            return mc
        with open(path, 'rb') as f:
            data = pickle.load(f)
        mc.hp_model = data['hp_model']
        mc.bearing_model = data['bearing_model']
        mc.diameter_model = data['diameter_model']
        mc.trained = data['trained']
        log.info(f"MotorClassifier loaded <- {path}")
        return mc

    @staticmethod
    def _hp_to_rpm(hp):
        return {0: 1797, 1: 1772, 2: 1750, 3: 1730}.get(hp, 1797)

    @staticmethod
    def feature_names():
        return ['rms', 'p2p', 'variance', 'skewness', 'kurtosis',
                'crest_factor', 'shape_factor', 'zcr',
                'band_0', 'band_1', 'band_2', 'band_3', 'band_4',
                'band_5', 'band_6', 'band_7', 'band_8', 'band_9',
                'dom_freq', 'centroid', 'spread', 'flatness',
                'e_low', 'e_mid', 'e_high',
                'peak_freq_0', 'peak_freq_1', 'peak_freq_2']

def build_cwru_motor_dataset(augment=True):
    """Load CWRU files (with optional augmentation), return features + labels."""
    if not os.path.exists(CWRU_DIR):
        log.warning("CWRU directory not found")
        return None, None, None, None, None, None

    records = []
    for root, dirs, files in os.walk(CWRU_DIR):
        rel = os.path.relpath(root, CWRU_DIR)
        parts = rel.split(os.sep)
        fs = _detect_fs(parts)
        for fname in files:
            if not fname.endswith('.mat'):
                continue
            meta = _parse_cwru_path(os.path.join(rel, fname))
            if meta is None:
                continue
            sig = _load_cwru_signal(os.path.join(root, fname))
            if sig is None or len(sig) < 100:
                continue
            sig_100 = _resample_to_fs(sig, fs)
            if len(sig_100) < WINDOW_SIZE:
                continue
            windows = _segment_into_windows(sig_100)
            for win in windows:
                records.append({
                    'features': _features_from_segment(win),
                    'hp': meta['hp'], 'bearing_location': meta['bearing_location'],
                    'fault_diameter': meta['fault_diameter'] if meta['fault_diameter'] else 0,
                    'rpm': meta['rpm'], '_raw_signal': sig_100.copy(),
                })

    if augment:
        records = _augment_records(records)

    if not records:
        return None, None, None, None, None, None

    X = np.array([r['features'] for r in records])
    y_hp = np.array([r['hp'] for r in records])
    y_bearing = np.array([r['bearing_location'] for r in records])
    y_diameter = np.array([r['fault_diameter'] for r in records])

    meta_out = {
        'n_samples': len(X), 'feature_dim': X.shape[1],
        'hp_values': sorted(set(r['hp'] for r in records)),
        'bearing_values': sorted(set(r['bearing_location'] for r in records)),
        'diameter_values': sorted(set([r['fault_diameter'] for r in records])),
    }
    log.info(f"CWRU motor dataset: {len(X)} samples, {meta_out['hp_values']} HP, "
             f"{meta_out['bearing_values']} bearing, {meta_out['diameter_values']} diameters")
    return X, y_hp, y_bearing, y_diameter, meta_out, records

def train_motor_classifier():
    """Full pipeline: load augmented data → ensemble → k-fold CV → save."""
    result = build_cwru_motor_dataset(augment=True)
    if result[0] is None:
        log.error("No motor data available")
        return None

    X, y_hp, y_bearing, y_diameter, meta_out, records = result

    # Subsample to keep training time tractable
    if len(X) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), MAX_TRAIN_SAMPLES, replace=False)
        X = X[idx]; y_hp = y_hp[idx]; y_bearing = y_bearing[idx]; y_diameter = y_diameter[idx]
        log.info(f"  Subsampled to {MAX_TRAIN_SAMPLES} for training")

    X_train, X_test, y_hp_train, y_hp_test, y_bear_train, y_bear_test, y_dia_train, y_dia_test = \
        train_test_split(X, y_hp, y_bearing, y_diameter, test_size=0.2,
                         random_state=42, stratify=y_hp)

    mc = MotorClassifier()
    mc.train(X_train, y_hp_train, y_bear_train, y_dia_train)

    hp_pred = mc.hp_model.predict(X_test)
    bear_pred = mc.bearing_model.predict(X_test)
    dia_pred = mc.diameter_model.predict(X_test)

    hp_acc = accuracy_score(y_hp_test, hp_pred)
    bear_acc = accuracy_score(y_bear_test, bear_pred)
    dia_acc = accuracy_score(y_dia_test, dia_pred)

    log.info(f"MotorClassifier evaluation (hold-out test set):")
    log.info(f"  HP accuracy:         {hp_acc:.4f}")
    log.info(f"  Bearing accuracy:    {bear_acc:.4f}")
    log.info(f"  Diameter accuracy:   {dia_acc:.4f}")

    # Stratified k-fold cross-validation on training portion only
    if CV_FOLDS > 1:
        log.info(f"  Cross-validation ({CV_FOLDS}-fold stratified):")
        cv_results = {'hp': [], 'bearing': [], 'diameter': []}
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        cv_sub = min(len(X_train), 5000)
        rng = np.random.default_rng(42)
        cv_idx = rng.choice(len(X_train), cv_sub, replace=False)
        X_cv = X_train[cv_idx]; hp_cv = y_hp_train[cv_idx]
        bear_cv = y_bear_train[cv_idx]; dia_cv = y_dia_train[cv_idx]
        for fold, (tr, te) in enumerate(skf.split(X_cv, hp_cv)):
            cv_model = MotorClassifier()
            cv_model.train(X_cv[tr], hp_cv[tr], bear_cv[tr], dia_cv[tr])
            cv_results['hp'].append(accuracy_score(hp_cv[te], cv_model.hp_model.predict(X_cv[te])))
            cv_results['bearing'].append(accuracy_score(bear_cv[te], cv_model.bearing_model.predict(X_cv[te])))
            cv_results['diameter'].append(accuracy_score(dia_cv[te], cv_model.diameter_model.predict(X_cv[te])))
            log.info(f"    Fold {fold+1}: HP={cv_results['hp'][-1]:.4f}, "
                     f"Bear={cv_results['bearing'][-1]:.4f}, Dia={cv_results['diameter'][-1]:.4f}")
        for target in ['hp', 'bearing', 'diameter']:
            vals = cv_results[target]
            log.info(f"  CV {target}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    report = {
        'test': {
            'hp': round(hp_acc, 4), 'bearing': round(bear_acc, 4),
            'diameter': round(dia_acc, 4),
        },
        'n_train': len(X_train), 'n_test': len(X_test), 'n_total': len(X),
        'augment_multiplier': AUGMENT_MULTIPLIER,
    }
    if CV_FOLDS > 1:
        report['cv'] = {
            target: {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'per_fold': [round(v, 4) for v in vals],
            }
            for target, vals in cv_results.items()
        }
    report_path = os.path.join(MODELS_DIR, 'motor_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    log.info(f"Results saved -> {report_path}")

    mc.save()
    return mc

if __name__ == '__main__':
    train_motor_classifier()
