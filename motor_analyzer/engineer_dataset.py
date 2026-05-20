"""
Engineer a unified, augmented dataset from CWRU + JNU + Synthetic.
Applies RPM variation, noise injection, and amplitude scaling to raw signals
before feature extraction — producing a more robust, realistic training set.
Output: data/engineered/ with consistent format.
"""

import os
import re
import json
import yaml
import logging
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from collections import defaultdict
from feature_pipeline import extract_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'engineered')

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

# ── Augmentation helpers ──

def _resample_to_fs(data, orig_fs, target_fs=TARGET_FS):
    duration = len(data) / orig_fs
    target_len = int(round(duration * target_fs))
    if target_len < 4:
        return np.array([])
    return scipy_signal.resample(data, target_len)


def _rpm_vary(signal, orig_fs, rpm_scale):
    """Time-stretch the signal to simulate RPM variation.
    
    rpm_scale: 0.8 = 20% slower, 1.2 = 20% faster.
    """
    if abs(rpm_scale - 1.0) < 0.01:
        return signal, orig_fs
    n = len(signal)
    orig_t = np.linspace(0, n / orig_fs, n)
    new_n = max(4, int(n * rpm_scale))
    new_t = np.linspace(0, n / orig_fs, new_n)
    f = interp1d(orig_t, signal, kind='linear', bounds_error=False, fill_value=0)
    stretched = f(new_t)
    new_fs = orig_fs * rpm_scale
    return stretched, new_fs


def _add_noise(signal, noise_level):
    """Add Gaussian noise. noise_level: std as fraction of signal RMS."""
    rms = np.std(signal)
    if rms < 1e-10:
        return signal
    noise = np.random.normal(0, rms * noise_level, len(signal))
    return signal + noise


def _scale_amplitude(signal, scale):
    """Multiply signal by random amplitude scale factor."""
    return signal * scale


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


def _augmentation_config():
    """Return augmentation parameters. Tweak these to control variability."""
    return {
        'rpm_scale_range': (0.85, 1.15),    # ±15% RPM variation
        'noise_level_range': (0.0, 0.15),   # up to 15% noise
        'amplitude_scale_range': (0.7, 1.3),# ±30% amplitude
        'augment_multiplier': 3,             # generate N augmented copies per original
    }


# ── Processors ──

def _process_cwru(augment_cfg):
    log.info("Processing CWRU dataset with augmentation...")
    cwru_dir = os.path.join(RAW_DIR, 'cwru')
    if not os.path.exists(cwru_dir):
        log.warning("  CWRU directory not found, skipping")
        return []

    import scipy.io
    records = []

    for root, dirs, files in os.walk(cwru_dir):
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

            de_key = None
            for k in data:
                if k.endswith('_DE_time'):
                    de_key = k
                    break
            if de_key is None:
                continue

            sig = data[de_key][:, 0].astype(np.float64)
            if fs is None:
                fs = 12000 if len(sig) < 200000 else 48000

            sig_100 = _resample_to_fs(sig, fs)
            if len(sig_100) < WINDOW_SIZE:
                continue

            # Original (no augmentation)
            windows = _segment_into_windows(sig_100)
            for win in windows:
                feats = _features_from_segment(win)
                records.append({
                    'features': feats,
                    'class': fault_type or 'Unknown',
                    'dataset': 'CWRU',
                    'augmented': False,
                })

            # Augmented copies
            aug_cfg = _augmentation_config()
            for _ in range(aug_cfg['augment_multiplier']):
                rpm_scale = np.random.uniform(*aug_cfg['rpm_scale_range'])
                noise_lvl = np.random.uniform(*aug_cfg['noise_level_range'])
                amp_scale = np.random.uniform(*aug_cfg['amplitude_scale_range'])

                aug_sig, _ = _rpm_vary(sig_100, TARGET_FS, rpm_scale)
                aug_sig = _resample_to_fs(aug_sig, TARGET_FS * rpm_scale)
                if len(aug_sig) < WINDOW_SIZE:
                    continue
                aug_sig = _add_noise(aug_sig, noise_lvl)
                aug_sig = _scale_amplitude(aug_sig, amp_scale)

                aug_windows = _segment_into_windows(aug_sig)
                for win in aug_windows:
                    feats = _features_from_segment(win)
                    records.append({
                        'features': feats,
                        'class': fault_type or 'Unknown',
                        'dataset': 'CWRU',
                        'augmented': True,
                        'rpm_scale': round(rpm_scale, 3),
                        'noise_level': round(noise_lvl, 3),
                        'amp_scale': round(amp_scale, 3),
                    })

    log.info(f"  CWRU: {len(records)} windows ({sum(1 for r in records if r.get('augmented'))} augmented)")
    return records


def _process_jnu(augment_cfg):
    log.info("Processing JNU dataset with augmentation...")
    jnu_dir = os.path.join(RAW_DIR, 'jnu')
    if not os.path.exists(jnu_dir):
        log.warning("  JNU directory not found, skipping")
        return []

    records = []
    JNU_FS = 50000

    for fname in sorted(os.listdir(jnu_dir)):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(jnu_dir, fname)

        cls = None
        for prefix, mapped in CLASS_MAP_JNU.items():
            if fname.startswith(prefix):
                cls = mapped
                break
        if cls is None:
            continue

        try:
            sig = np.loadtxt(fpath, dtype=np.float64)
        except Exception as e:
            log.error(f"  Failed to load {fname}: {e}")
            continue

        if len(sig) < 100:
            continue

        sig_100 = _resample_to_fs(sig, JNU_FS)
        if len(sig_100) < WINDOW_SIZE:
            continue

        # Original
        windows = _segment_into_windows(sig_100)
        for win in windows:
            feats = _features_from_segment(win)
            records.append({
                'features': feats,
                'class': cls,
                'dataset': 'JNU',
                'augmented': False,
            })

        # Augmented copies
        aug_cfg = _augmentation_config()
        for _ in range(aug_cfg['augment_multiplier']):
            rpm_scale = np.random.uniform(*aug_cfg['rpm_scale_range'])
            noise_lvl = np.random.uniform(*aug_cfg['noise_level_range'])
            amp_scale = np.random.uniform(*aug_cfg['amplitude_scale_range'])

            aug_sig, _ = _rpm_vary(sig_100, TARGET_FS, rpm_scale)
            aug_sig = _resample_to_fs(aug_sig, TARGET_FS * rpm_scale)
            if len(aug_sig) < WINDOW_SIZE:
                continue
            aug_sig = _add_noise(aug_sig, noise_lvl)
            aug_sig = _scale_amplitude(aug_sig, amp_scale)

            aug_windows = _segment_into_windows(aug_sig)
            for win in aug_windows:
                feats = _features_from_segment(win)
                records.append({
                    'features': feats,
                    'class': cls,
                    'dataset': 'JNU',
                    'augmented': True,
                    'rpm_scale': round(rpm_scale, 3),
                    'noise_level': round(noise_lvl, 3),
                    'amp_scale': round(amp_scale, 3),
                })

    log.info(f"  JNU: {len(records)} windows ({sum(1 for r in records if r.get('augmented'))} augmented)")
    return records


def _process_mafaulda(augment_cfg):
    """Process MAFAULDA Machinery Fault dataset.
    Structure: {fault_type}/[{severity}/]{rpm}.csv with 8-column CSV data.
    """
    log.info("Processing MAFAULDA dataset...")
    maf_dir = os.path.join(RAW_DIR, 'mafaulda')
    if not os.path.exists(maf_dir):
        log.warning("  MAFAULDA directory not found, skipping")
        return []

    # MAFAULDA sample rates are encoded in filenames like 12.288.csv = 12288 Hz
    # Map folder paths to fault classes
    FOLDER_CLASS_MAP = {
        'normal': 'Normal',
        'imbalance': 'Unbalanced',
        'horizontal-misalignment': 'Misalignment',
        'vertical-misalignment': 'Misalignment',
        'overhang': None,  # subfolder determines class
        'underhang': None,  # subfolder determines class
    }
    SUBFOLDER_CLASS_MAP = {
        'ball_fault': 'BallFault',
        'cage_fault': 'CageFault',
        'outer_race': 'OuterRace',
    }

    records = []
    MAFAULDA_FS = 12288  # base sample rate (12.288 kHz)

    for fault_folder in sorted(os.listdir(maf_dir)):
        fault_path = os.path.join(maf_dir, fault_folder)
        if not os.path.isdir(fault_path):
            continue

        base_class = FOLDER_CLASS_MAP.get(fault_folder.lower())
        if fault_folder.lower() not in FOLDER_CLASS_MAP:
            # Unknown folder, skip
            continue

        if base_class is None:
            # Subfolder-based classification: overhang/ball_fault → BallFault
            for sub in sorted(os.listdir(fault_path)):
                sub_path = os.path.join(fault_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                mapped_class = SUBFOLDER_CLASS_MAP.get(sub.lower())
                if mapped_class is None:
                    continue
                _process_mafaulda_csvs(sub_path, mapped_class, 'MAFAULDA',
                                       MAFAULDA_FS, records, augment_cfg, severity=sub)
            continue

        for entry in sorted(os.listdir(fault_path)):
            entry_path = os.path.join(fault_path, entry)
            if os.path.isdir(entry_path):
                # Severity subfolder (e.g. 6g, 0.5mm, 10g)
                mapped_class = SUBFOLDER_CLASS_MAP.get(entry.lower(), base_class)
                _process_mafaulda_csvs(entry_path, mapped_class,
                                       'MAFAULDA', MAFAULDA_FS, records, augment_cfg, severity=entry)
            elif entry.endswith('.csv'):
                _process_mafaulda_csv(fault_path, entry, base_class,
                                      'MAFAULDA', MAFAULDA_FS, records, augment_cfg)

    log.info(f"  MAFAULDA: {len(records)} windows ({sum(1 for r in records if r.get('augmented'))} augmented)")
    return records


def _process_mafaulda_csvs(dir_path, fault_class, dataset_name, fs, records, aug_cfg, max_files=15, nested=False, severity=None):
    """Process CSVs in a directory (walking into subdirs). Limits per dir to keep runtime tractable."""
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    if csv_files:
        limit = max_files if not nested else 3
        if len(csv_files) > limit:
            rng = np.random.default_rng(42)
            csv_files = list(rng.choice(sorted(csv_files), limit, replace=False))
        for fname in sorted(csv_files):
            _process_mafaulda_csv(dir_path, fname, fault_class, dataset_name, fs, records, aug_cfg, severity=severity)
    else:
        # No direct CSVs — recurse into severity subdirectories (e.g. 0g/, 6g/)
        for sub in sorted(os.listdir(dir_path)):
            sub_path = os.path.join(dir_path, sub)
            if os.path.isdir(sub_path):
                # Propagate severity with subfolder name appended for nested contexts
                child_severity = f"{severity}/{sub}" if severity else sub
                _process_mafaulda_csvs(sub_path, fault_class, dataset_name, fs,
                                       records, aug_cfg, max_files, nested=True, severity=child_severity)


def _process_mafaulda_csv(dir_path, fname, fault_class, dataset_name, base_fs, records, aug_cfg, severity=None):
    """Process a single MAFAULDA CSV file with augmentation."""
    # Parse sample rate from filename: "12.288.csv" → 12288 Hz
    fs_match = re.match(r'^(\d+)\.(\d+)\.csv$', fname)
    fs = base_fs
    if fs_match:
        try:
            fs = int(f'{fs_match.group(1)}{fs_match.group(2)}')
        except ValueError:
            fs = base_fs
    if fs < 100:
        fs = base_fs

    fpath = os.path.join(dir_path, fname)
    try:
        df = np.loadtxt(fpath, delimiter=',', dtype=np.float64)
    except Exception as e:
        log.error(f"  Failed to load {fname}: {e}")
        return

    if df.ndim == 1:
        sig = df
    else:
        sig = df[:, 0]  # use first column as primary vibration channel

    if len(sig) < 100:
        return

    # Downsample to target FS
    sig_100 = _resample_to_fs(sig, fs)
    if len(sig_100) < WINDOW_SIZE:
        return

    entry = {
        'features': None, 'class': fault_class, 'dataset': dataset_name, 'augmented': False,
    }
    if severity:
        entry['severity'] = severity

    # Original
    windows = _segment_into_windows(sig_100)
    for win in windows:
        feats = _features_from_segment(win)
        records.append({**entry, 'features': feats})

    # Augmented copies
    for _ in range(aug_cfg['augment_multiplier']):
        rpm_scale = np.random.uniform(*aug_cfg['rpm_scale_range'])
        noise_lvl = np.random.uniform(*aug_cfg['noise_level_range'])
        amp_scale = np.random.uniform(*aug_cfg['amplitude_scale_range'])

        aug_sig, _ = _rpm_vary(sig_100, TARGET_FS, rpm_scale)
        aug_sig = _resample_to_fs(aug_sig, TARGET_FS * rpm_scale)
        if len(aug_sig) < WINDOW_SIZE:
            continue
        aug_sig = _add_noise(aug_sig, noise_lvl)
        aug_sig = _scale_amplitude(aug_sig, amp_scale)

        aug_windows = _segment_into_windows(aug_sig)
        for win in aug_windows:
            feats = _features_from_segment(win)
            records.append({
                **entry, 'features': feats, 'augmented': True,
                'rpm_scale': round(rpm_scale, 3),
                'noise_level': round(noise_lvl, 3),
                'amp_scale': round(amp_scale, 3),
            })


def _process_synthetic(samples_per_class=50):
    """Generate synthetic data with RPM overlap, matching engineered format."""
    log.info("Processing Synthetic dataset...")
    from generate_synthetic_data import (
        normal_vibration, unbalanced_vibration,
        bearing_fault_vibration, misalignment_vibration
    )

    records = []
    patterns = [
        ('Normal', normal_vibration),
        ('InnerRace', bearing_fault_vibration),   # bearing fault as inner race
        ('Unbalanced', unbalanced_vibration),
        ('Misalignment', misalignment_vibration),
    ]

    for cls_name, gen_fn in patterns:
        for _ in range(samples_per_class):
            raw = gen_fn(5.0)
            windows = _segment_into_windows(raw)
            for win in windows:
                feats = _features_from_segment(win)
                records.append({
                    'features': feats,
                    'class': cls_name,
                    'dataset': 'Synthetic',
                    'augmented': False,
                })

    log.info(f"  Synthetic: {len(records)} windows")
    return records


# ── Merge & Normalize ──

def _normalize_rms_across_datasets(all_records):
    """Cross-dataset RMS normalization so no dataset dominates by amplitude."""
    datasets = set(r['dataset'] for r in all_records)
    dataset_rms = {}
    for ds in datasets:
        ds_records = [r for r in all_records if r['dataset'] == ds]
        vals = [abs(r['features'][0]) for r in ds_records if r['features']]
        dataset_rms[ds] = float(np.mean(vals)) if vals else 1.0
        log.info(f"  RMS({ds}) = {dataset_rms[ds]:.6f}")

    if len(datasets) > 1:
        rms_vals = [v for v in dataset_rms.values() if v > 1e-10]
        if rms_vals:
            target_rms = np.mean(rms_vals)
            for r in all_records:
                ds = r['dataset']
                ds_rms = dataset_rms.get(ds, 1.0)
                if ds_rms > 1e-10:
                    scale = target_rms / ds_rms
                    r['features'] = [f * scale for f in r['features']]
    return all_records


def _process_nasa_ims(aug_cfg, max_files_per_class=50):
    """Process NASA IMS run-to-failure bearing dataset.
    Format: tab-separated text, 4 channels × 40960 samples at 20kHz per file.
    First N% of each test → Normal, last N% → fault class.
    """
    log.info("Processing NASA IMS dataset (run-to-failure)...")
    ims_base = os.path.join(RAW_DIR, 'nasa_ims')
    if not os.path.exists(ims_base):
        log.warning("  NASA IMS directory not found, skipping")
        return []

    TESTS = [
        {'dir': '1st_test/1st_test', 'fault': 'OuterRace', 'pct': 0.10},
        {'dir': '2nd_test/2nd_test', 'fault': 'OuterRace', 'pct': 0.10},
        {'dir': '3rd_test/4th_test/txt', 'fault': 'InnerRace', 'pct': 0.10},
    ]

    records = []
    IMS_FS = 20000

    for test in TESTS:
        test_dir = os.path.join(ims_base, test['dir'])
        if not os.path.isdir(test_dir):
            log.warning(f"  Test dir not found: {test['dir']}, skipping")
            continue

        files = sorted([f for f in os.listdir(test_dir)
                        if os.path.isfile(os.path.join(test_dir, f))])
        if not files:
            continue

        n_normal = max(1, int(len(files) * test['pct']))
        n_fault = max(1, int(len(files) * test['pct']))
        normal_files = files[:n_normal]
        fault_files = files[-n_fault:]

        # Limit per class
        rng = np.random.default_rng(42)
        if len(normal_files) > max_files_per_class:
            normal_files = list(rng.choice(normal_files, max_files_per_class, replace=False))
        if len(fault_files) > max_files_per_class:
            fault_files = list(rng.choice(fault_files, max_files_per_class, replace=False))

        fault_class = test['fault']

        for fname in normal_files:
            _process_nasa_ims_file(test_dir, fname, 'Normal', 'NASA_IMS',
                                   IMS_FS, records, aug_cfg)
        for fname in fault_files:
            _process_nasa_ims_file(test_dir, fname, fault_class, 'NASA_IMS',
                                   IMS_FS, records, aug_cfg)

        log.info(f"  {test['dir']}: {len(normal_files)} Normal + {len(fault_files)} {fault_class}")

    log.info(f"  NASA IMS: {len(records)} windows ({sum(1 for r in records if r.get('augmented'))} augmented)")
    return records


def _process_nasa_ims_file(dir_path, fname, fault_class, dataset_name, fs, records, aug_cfg):
    """Process a single NASA IMS text-format file."""
    fpath = os.path.join(dir_path, fname)
    try:
        with open(fpath, 'r') as f:
            content = f.read()
    except Exception as e:
        log.error(f"  Failed to read {fname}: {e}")
        return

    # Parse tab-separated 4-column data
    import re
    floats = re.findall(r'-?\d+\.?\d*', content)
    if len(floats) < 4:
        return
    data = np.array([float(x) for x in floats], dtype=np.float64)

    # Reshape to channels (4 columns) — files have 163840 values = 4 × 40960
    n_cols = 4
    usable = (len(data) // n_cols) * n_cols
    data = data[:usable]
    if data.size == 0:
        return
    data = data.reshape(-1, n_cols)

    # Use channel 0 as primary vibration
    sig = data[:, 0]

    if len(sig) < 100:
        return

    # Downsample to target FS
    sig_100 = _resample_to_fs(sig, fs)
    if len(sig_100) < WINDOW_SIZE:
        return

    # Original
    windows = _segment_into_windows(sig_100)
    for win in windows:
        feats = _features_from_segment(win)
        records.append({
            'features': feats,
            'class': fault_class,
            'dataset': dataset_name,
            'augmented': False,
        })

    # Augmented copies
    for _ in range(aug_cfg['augment_multiplier']):
        rpm_scale = np.random.uniform(*aug_cfg['rpm_scale_range'])
        noise_lvl = np.random.uniform(*aug_cfg['noise_level_range'])
        amp_scale = np.random.uniform(*aug_cfg['amplitude_scale_range'])

        aug_sig, _ = _rpm_vary(sig_100, TARGET_FS, rpm_scale)
        aug_sig = _resample_to_fs(aug_sig, TARGET_FS * rpm_scale)
        if len(aug_sig) < WINDOW_SIZE:
            continue
        aug_sig = _add_noise(aug_sig, noise_lvl)
        aug_sig = _scale_amplitude(aug_sig, amp_scale)

        aug_windows = _segment_into_windows(aug_sig)
        for win in aug_windows:
            feats = _features_from_segment(win)
            records.append({
                'features': feats,
                'class': fault_class,
                'dataset': dataset_name,
                'augmented': True,
            })


def _resample_classes(records, target_per_class=5000):
    """Resample to balance class distribution via oversampling minority + undersampling majority."""
    from collections import Counter
    class_counts = Counter(r['class'] for r in records)
    log.info(f"Pre-resample distribution: {dict(class_counts)}")

    by_class = defaultdict(list)
    for r in records:
        by_class[r['class']].append(r)

    balanced = []
    for cls, samples in by_class.items():
        n = len(samples)
        if n == 0:
            continue
        rng = np.random.default_rng(42)
        if n >= target_per_class:
            # Undersample
            idx = rng.choice(n, target_per_class, replace=False)
            balanced.extend([samples[i] for i in idx])
        else:
            # Oversample with replacement
            idx = rng.choice(n, target_per_class, replace=True)
            balanced.extend([samples[i] for i in idx])

    log.info(f"Post-resample distribution: {dict(Counter(r['class'] for r in balanced))}")
    log.info(f"  Total after resampling: {len(balanced)}")
    return balanced


def engineer_dataset(include_synthetic=True):
    """Run full engineering pipeline: load, augment, merge, normalize, save."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aug_cfg = _augmentation_config()

    all_records = []
    all_records.extend(_process_cwru(aug_cfg))
    all_records.extend(_process_jnu(aug_cfg))
    all_records.extend(_process_mafaulda(aug_cfg))
    all_records.extend(_process_nasa_ims(aug_cfg))
    if include_synthetic:
        all_records.extend(_process_synthetic())

    if not all_records:
        log.warning("No data from any dataset!")
        return None, None, None, None

    # Cross-dataset RMS normalization
    log.info("Normalizing RMS across datasets...")
    all_records = _normalize_rms_across_datasets(all_records)

    # Resample to balance classes
    all_records = _resample_classes(all_records, target_per_class=5000)

    # Build feature matrix and labels
    X, y, company_labels, fault_labels = [], [], [], []
    company_names = sorted(set(r['dataset'] for r in all_records))
    company_to_idx = {name: i for i, name in enumerate(company_names)}

    class_sets = defaultdict(set)
    for r in all_records:
        class_sets[r['dataset']].add(r['class'])

    # Unified fault class mapping
    ALL_FAULT_CLASSES = ['Normal', 'InnerRace', 'OuterRace', 'BallFault', 'CageFault', 'Unbalanced', 'Misalignment']
    fault_to_idx = {c: i for i, c in enumerate(ALL_FAULT_CLASSES)}

    for r in all_records:
        X.append(r['features'])
        y.append(r['class'])
        company_labels.append(company_to_idx[r['dataset']])
        fault_labels.append(fault_to_idx.get(r['class'], -1))

    X = np.array(X)
    y = np.array(y)
    company_labels = np.array(company_labels)
    fault_labels = np.array(fault_labels)

    # Collect severity info per class (from MAFAULDA records that have it)
    severity_per_class = defaultdict(set)
    for r in all_records:
        if 'severity' in r and r.get('severity'):
            severity_per_class[r['class']].add(r['severity'])

    # Metadata
    metadata = {
        'n_samples': len(X),
        'feature_dim': X.shape[1] if X.ndim > 1 else 0,
        'companies': company_names,
        'classes_per_company': {k: sorted(v) for k, v in class_sets.items()},
        'all_fault_classes': ALL_FAULT_CLASSES,
        'severity_per_class': {k: sorted(v) for k, v in severity_per_class.items()},
        'dataset_rms': {},
        'augmentation_config': aug_cfg,
    }

    # Save
    np.save(os.path.join(OUTPUT_DIR, 'X_all.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_all.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'company_labels.npy'), company_labels)
    np.save(os.path.join(OUTPUT_DIR, 'fault_labels.npy'), fault_labels)

    meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Per-company splits
    for ci, name in enumerate(company_names):
        mask = company_labels == ci
        if not np.any(mask):
            continue
        Xc = X[mask]
        yc = y[mask]
        np.save(os.path.join(OUTPUT_DIR, f'X_{name}.npy'), Xc)
        np.save(os.path.join(OUTPUT_DIR, f'y_{name}.npy'), yc)
        classes, counts = np.unique(yc, return_counts=True)
        log.info(f"  {name}: {len(Xc)} samples, classes: {dict(zip(classes, counts))}")

    log.info(f"\n=== Engineered Dataset Summary ===")
    log.info(f"  Total samples: {len(X)}")
    log.info(f"  Feature dim: {X.shape[1]}")
    log.info(f"  Companies: {company_names}")
    log.info(f"  Fault classes: {ALL_FAULT_CLASSES}")
    log.info(f"  Augmentation: {aug_cfg}")
    log.info(f"  Saved to: {OUTPUT_DIR}/")

    return X, y, company_labels, metadata


if __name__ == '__main__':
    engineer_dataset()
