"""
Unified data ingestion adapters for all datasets:
CWRU, JNU, MAFAULDA, NASA IMS, Unbalance, Synthetic, Generic.
"""

import os
import re
import json
import glob
import yaml
import hashlib
import struct
import logging
import numpy as np
from scipy import signal as sp_signal
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

WINDOW_SIZE = 128
TARGET_FS = 100
CLASS_MAP_CWRU = {"Normal": "Normal", "IR": "InnerRace", "OR": "OuterRace", "B": "BallFault"}
CLASS_MAP_JNU = {"n": "Normal", "ib": "InnerRace", "ob": "OuterRace", "tb": "BallFault"}


# ─── Shared Utilities ────────────────────────────────────

def compute_checksum(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _resample_to_100hz(data, original_fs):
    duration = len(data) / original_fs
    target_len = int(round(duration * TARGET_FS))
    if target_len < 4:
        return np.array([])
    return sp_signal.resample(data, target_len)


def _segment_into_windows(data, window_size=WINDOW_SIZE, step=None):
    if step is None:
        step = window_size // 2
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start:start + window_size])
    return windows


def _features_from_segment(segment, fs=TARGET_FS):
    from features.pipeline import extract_features
    baseline = float(np.mean(segment))
    return extract_features(segment.tolist(), baseline, fs)


def _map_label(raw_class, class_mapping):
    lower = raw_class.lower().strip()
    for key, val in class_mapping.items():
        if key in lower:
            return val
    return raw_class


# ─── CWRU Adapter ────────────────────────────────────────

def process_cwru():
    logging.info("Processing CWRU dataset...")
    cwru_dir = os.path.join(RAW_DIR, "cwru")
    if not os.path.exists(cwru_dir):
        logging.warning("CWRU directory not found")
        return []

    import scipy.io
    records = []
    for root, dirs, files in os.walk(cwru_dir):
        rel = os.path.relpath(root, cwru_dir).replace("\\", "/")
        parts = rel.split("/")

        fs = None
        for p in parts:
            if "12k" in p.lower(): fs = 12000; break
            if "48k" in p.lower(): fs = 48000; break

        fault_type = None
        for p in parts:
            if p in CLASS_MAP_CWRU:
                fault_type = CLASS_MAP_CWRU[p]; break

        for fname in files:
            if not fname.endswith(".mat"): continue
            try:
                data = scipy.io.loadmat(os.path.join(root, fname))
            except Exception:
                continue
            de_key = next((k for k in data if k.endswith("_DE_time")), None)
            if de_key is None: continue
            sig = data[de_key][:, 0].astype(np.float64)
            if fs is None:
                fs = 12000 if len(sig) < 200000 else 48000
            sig_100 = _resample_to_100hz(sig, fs)
            if len(sig_100) < WINDOW_SIZE: continue
            for win in _segment_into_windows(sig_100):
                records.append({
                    "filepath": fname, "raw_class": fault_type or "Unknown",
                    "class": fault_type or "Unknown", "dataset": "CWRU",
                    "format": "mat", "features": _features_from_segment(win),
                })

    logging.info(f"  CWRU: {len(records)} windows")
    return records


# ─── JNU Adapter ─────────────────────────────────────────

def process_jnu():
    logging.info("Processing JNU dataset...")
    jnu_dir = os.path.join(RAW_DIR, "jnu")
    if not os.path.exists(jnu_dir):
        logging.warning("JNU directory not found")
        return []

    records = []
    for fname in sorted(os.listdir(jnu_dir)):
        if not fname.endswith(".csv"): continue
        cls = next((CLASS_MAP_JNU[p] for p in CLASS_MAP_JNU if fname.startswith(p)), None)
        if cls is None: continue
        try:
            sig = np.loadtxt(os.path.join(jnu_dir, fname), dtype=np.float64)
        except Exception:
            continue
        sig_100 = _resample_to_100hz(sig, 50000)
        if len(sig_100) < WINDOW_SIZE: continue
        for win in _segment_into_windows(sig_100):
            records.append({
                "filepath": fname, "raw_class": cls, "class": cls,
                "dataset": "JNU", "format": "csv",
                "features": _features_from_segment(win),
            })

    logging.info(f"  JNU: {len(records)} windows")
    return records


# ─── Synthetic Adapter ───────────────────────────────────

def process_synthetic(n_companies=1, samples_per_company=100):
    logging.info("Processing Synthetic dataset...")
    from generate_synthetic_data import normal_vibration, unbalanced_vibration, bearing_fault_vibration, misalignment_vibration
    records = []
    patterns = [
        ("Normal", normal_vibration), ("Unbalanced", unbalanced_vibration),
        ("BearingFault", bearing_fault_vibration), ("Misalignment", misalignment_vibration),
    ]
    for cls_name, gen_fn in patterns:
        for _ in range(samples_per_company):
            raw = gen_fn(5.0)
            for win in _segment_into_windows(raw):
                records.append({
                    "filepath": f"synthetic_{cls_name}", "raw_class": cls_name, "class": cls_name,
                    "dataset": "Synthetic", "format": "generated",
                    "features": _features_from_segment(win),
                })
    logging.info(f"  Synthetic: {len(records)} windows")
    return records


# ─── MAFAULDA Adapter ────────────────────────────────────

def process_mafaulda(dataset_dir, class_mapping, exclude_classes):
    logging.info("Processing MAFAULDA...")
    entries = []
    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        wav_files = glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
        logging.info(f"  Found {len(wav_files)} WAV files")
        for fp in wav_files:
            parts = fp.replace("\\", "/").split("/")
            raw_cls = parts[-2] if len(parts) >= 2 else "Unknown"
            if any(ex in raw_cls.lower() for ex in exclude_classes): continue
            entries.append({"filepath": fp, "raw_class": raw_cls,
                            "class": _map_label(raw_cls, class_mapping),
                            "dataset": "mafaulda", "format": "wav"})
    else:
        logging.info(f"  Found {len(csv_files)} CSV files")
        for fp in csv_files:
            parts = fp.replace("\\", "/").split("/")
            raw_cls = parts[-2] if len(parts) >= 2 else "Unknown"
            if any(ex in raw_cls.lower() for ex in exclude_classes): continue
            entries.append({"filepath": fp, "raw_class": raw_cls,
                            "class": _map_label(raw_cls, class_mapping),
                            "dataset": "mafaulda", "format": "csv"})

    logging.info(f"  MAFAULDA: {len(entries)} entries")
    return entries


# ─── NASA IMS Adapter ────────────────────────────────────

def process_nasa_ims(dataset_dir, class_mapping, exclude_classes):
    logging.info("Processing NASA IMS...")
    entries = []
    for test_dir in sorted(glob.glob(os.path.join(dataset_dir, "*"))):
        if not os.path.isdir(test_dir): continue
        inner = [d for d in glob.glob(os.path.join(test_dir, "*")) if os.path.isdir(d)]
        search_dir = inner[0] if inner else test_dir
        files = sorted(f for f in glob.glob(os.path.join(search_dir, "*"))
                       if os.path.isfile(f) and not f.endswith((".json", ".md")))
        if not files: continue
        split = int(len(files) * 0.80)
        test_name = os.path.basename(test_dir)
        for i, fp in enumerate(files):
            raw_cls = "normal" if i < split else "bearing_fault"
            entries.append({"filepath": fp, "raw_class": raw_cls,
                            "class": _map_label(raw_cls, class_mapping),
                            "dataset": f"nasa_ims_{test_name}", "format": "raw_binary"})

    logging.info(f"  NASA IMS: {len(entries)} entries")
    return entries


# ─── Unbalance Adapter ───────────────────────────────────

def process_unbalance(dataset_dir, class_mapping, exclude_classes):
    logging.info("Processing Unbalance...")
    entries = []
    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    for fp in csv_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        raw_cls = "normal" if name.startswith("0") else "unbalance"
        entries.append({"filepath": fp, "raw_class": raw_cls,
                        "class": _map_label(raw_cls, class_mapping),
                        "dataset": "unbalance", "format": "csv_vibration"})
    logging.info(f"  Unbalance: {len(entries)} entries")
    return entries


# ─── Generic Adapter ─────────────────────────────────────

def process_generic(dataset_dir, class_mapping, exclude_classes):
    entries = []
    for pattern in ["**/*.wav", "**/*.csv"]:
        for fp in glob.glob(os.path.join(dataset_dir, pattern), recursive=True):
            parts = fp.replace("\\", "/").split("/")
            raw_cls = parts[-2] if len(parts) >= 2 else "Unknown"
            if any(ex in raw_cls.lower() for ex in exclude_classes): continue
            fmt = "wav" if fp.endswith(".wav") else "csv"
            entries.append({"filepath": fp, "raw_class": raw_cls,
                            "class": _map_label(raw_cls, class_mapping),
                            "dataset": os.path.basename(dataset_dir), "format": fmt})
    return entries


# ─── Registry ────────────────────────────────────────────

DATASET_ADAPTERS = {
    "mafaulda": process_mafaulda,
    "nasa_ims": process_nasa_ims,
    "unbalance": process_unbalance,
    "generic": process_generic,
}


# ─── Unified Orchestrator ────────────────────────────────

def ingest_all(config_path=None):
    """Main entry point: ingest all configured datasets and return manifest."""
    if config_path is None:
        config_path = os.path.join(BASE_DIR, "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_dir = config.get("data", {}).get("raw_dir", RAW_DIR)
    excluded = config.get("data", {}).get("exclude_classes", [])
    class_mapping = config.get("data", {}).get("class_mapping", {})
    datasets_config = config.get("datasets", {})
    use_synthetic = config.get("companies", {}).get("use_synthetic", True)

    os.makedirs(raw_dir, exist_ok=True)
    manifest = []

    if datasets_config:
        for ds_name, ds_conf in datasets_config.items():
            ds_dir = ds_conf.get("path", os.path.join(raw_dir, ds_name))
            adapter = ds_conf.get("adapter", ds_name)
            if not os.path.exists(ds_dir):
                logging.warning(f"Dataset dir not found: {ds_dir}")
                continue
            fn = DATASET_ADAPTERS.get(adapter, process_generic)
            entries = fn(ds_dir, class_mapping, excluded)
            manifest.extend(entries)
    else:
        # Legacy scan mode
        if os.path.exists(raw_dir):
            for subdir in sorted(os.listdir(raw_dir)):
                sp = os.path.join(raw_dir, subdir)
                if os.path.isdir(sp):
                    fn = DATASET_ADAPTERS.get(subdir, process_generic)
                    manifest.extend(fn(sp, class_mapping, excluded))

    logging.info(f"Computing checksums for {len(manifest)} files...")
    for entry in tqdm(manifest, desc="Checksums"):
        entry["checksum"] = compute_checksum(entry["filepath"])

    manifest_path = os.path.join(raw_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    counts = {}
    for e in manifest:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    logging.info(f"Total: {len(manifest)} files. Classes: {json.dumps(counts, indent=2)}")
    return manifest
