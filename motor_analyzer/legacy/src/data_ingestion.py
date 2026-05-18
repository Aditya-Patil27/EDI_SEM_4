import os
import glob
import hashlib
import json
import yaml
import struct
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def compute_checksum(filepath):
    """Compute MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _process_mafaulda(dataset_dir, class_mapping, exclude_classes):
    """
    MAFAULDA: WAV files organized in subdirectories by fault type.
    Structure: mafaulda/<class_folder>/<recording>.wav
    Classes: normal, imbalance, horizontal-misalignment, vertical-misalignment,
             underhang (bearing), overhang (bearing)
    """
    entries = []
    wav_files = glob.glob(os.path.join(dataset_dir, '**', '*.wav'), recursive=True)

    if not wav_files:
        csv_files = glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)
        if csv_files:
            logging.info(f"MAFAULDA: Found {len(csv_files)} CSV files (will be processed as time-series)")
            for filepath in csv_files:
                normalized = filepath.replace('\\', '/')
                parts = normalized.split('/')
                raw_class = parts[-2] if len(parts) >= 2 else "Unknown"

                if any(ex in raw_class.lower() for ex in exclude_classes):
                    continue

                mapped = _map_label(raw_class, class_mapping)
                entries.append({
                    "filepath": filepath,
                    "raw_class": raw_class,
                    "class": mapped,
                    "dataset": "mafaulda",
                    "format": "csv"
                })
            return entries

    logging.info(f"MAFAULDA: Found {len(wav_files)} WAV files")
    for filepath in wav_files:
        normalized = filepath.replace('\\', '/')
        parts = normalized.split('/')
        raw_class = parts[-2] if len(parts) >= 2 else "Unknown"

        if any(ex in raw_class.lower() for ex in exclude_classes):
            continue

        mapped = _map_label(raw_class, class_mapping)
        entries.append({
            "filepath": filepath,
            "raw_class": raw_class,
            "class": mapped,
            "dataset": "mafaulda",
            "format": "wav"
        })

    return entries


def _process_nasa_ims(dataset_dir, class_mapping, exclude_classes):
    """
    NASA IMS Bearing: Raw binary accelerometer data or text files.
    Structure: bearing-dataset/<test_N>/<test_N>/<timestamp_files>
    Labels derived from position in run-to-failure sequence:
      - First 80% of files -> Normal
      - Last 20% of files -> Bearing Fault
    """
    entries = []
    all_files = []

    for test_dir in sorted(glob.glob(os.path.join(dataset_dir, '*'))):
        if not os.path.isdir(test_dir):
            continue

        inner_dirs = [d for d in glob.glob(os.path.join(test_dir, '*')) if os.path.isdir(d)]
        search_dir = inner_dirs[0] if inner_dirs else test_dir

        data_files = sorted([
            f for f in glob.glob(os.path.join(search_dir, '*'))
            if os.path.isfile(f) and not f.endswith('.json') and not f.endswith('.md')
        ])

        if not data_files:
            continue

        split_point = int(len(data_files) * 0.80)
        test_name = os.path.basename(test_dir)

        for i, filepath in enumerate(data_files):
            raw_class = "normal" if i < split_point else "bearing_fault"
            mapped = _map_label(raw_class, class_mapping)

            entries.append({
                "filepath": filepath,
                "raw_class": raw_class,
                "class": mapped,
                "dataset": f"nasa_ims_{test_name}",
                "format": "raw_binary"
            })

    logging.info(f"NASA IMS: Found {len(entries)} data files across test sets")
    return entries


def _process_unbalance(dataset_dir, class_mapping, exclude_classes):
    """
    Unbalance Vibration: CSV files with vibration columns.
    Structure: <dataset_dir>/<NnD.csv or NnE.csv>
    Files starting with '0' = Normal (no unbalance), others = Unbalanced.
    Columns: V_in, Measured_RPM, Vibration_1, Vibration_2, Vibration_3
    """
    entries = []
    csv_files = glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)

    logging.info(f"Unbalance: Found {len(csv_files)} CSV files")
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        name_no_ext = os.path.splitext(filename)[0]

        if name_no_ext.startswith('0'):
            raw_class = "normal"
        else:
            raw_class = "unbalance"

        mapped = _map_label(raw_class, class_mapping)
        entries.append({
            "filepath": filepath,
            "raw_class": raw_class,
            "class": mapped,
            "dataset": "unbalance",
            "format": "csv_vibration"
        })

    return entries


def _process_generic(dataset_dir, class_mapping, exclude_classes):
    """
    Generic adapter: scans for WAV/CSV files organized in class subfolders.
    Structure: <dataset_dir>/<class_folder>/<file>.wav|csv
    """
    entries = []
    audio_files = glob.glob(os.path.join(dataset_dir, '**', '*.wav'), recursive=True)
    audio_files += glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)

    logging.info(f"Generic dataset: Found {len(audio_files)} files in {dataset_dir}")
    for filepath in audio_files:
        normalized = filepath.replace('\\', '/')
        parts = normalized.split('/')
        raw_class = parts[-2] if len(parts) >= 2 else "Unknown"

        if any(ex in raw_class.lower() for ex in exclude_classes):
            continue

        fmt = "wav" if filepath.endswith('.wav') else "csv"
        mapped = _map_label(raw_class, class_mapping)
        entries.append({
            "filepath": filepath,
            "raw_class": raw_class,
            "class": mapped,
            "dataset": os.path.basename(dataset_dir),
            "format": fmt
        })

    return entries


def _map_label(raw_class, class_mapping):
    """Map a raw class label to standardized vocabulary using config mapping."""
    lower = raw_class.lower().strip()
    for key, val in class_mapping.items():
        if key in lower:
            return val
    return raw_class


DATASET_ADAPTERS = {
    "mafaulda": _process_mafaulda,
    "nasa_ims": _process_nasa_ims,
    "unbalance": _process_unbalance,
}


def ingest_data(config_path="config.yaml"):
    """
    Scans all configured dataset directories, processes each with the
    appropriate adapter, and creates a unified manifest.json.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_dir = config['data']['raw_dir']
    excluded = config['data'].get('exclude_classes', [])
    class_mapping = config['data'].get('class_mapping', {})
    datasets_config = config.get('datasets', {})

    os.makedirs(raw_dir, exist_ok=True)

    manifest = []

    if datasets_config:
        for ds_name, ds_conf in datasets_config.items():
            ds_dir = ds_conf.get('path', os.path.join(raw_dir, ds_name))
            adapter_name = ds_conf.get('adapter', ds_name)

            if not os.path.exists(ds_dir):
                logging.warning(f"Dataset directory not found: {ds_dir} — skipping '{ds_name}'")
                continue

            adapter_fn = DATASET_ADAPTERS.get(adapter_name, _process_generic)
            logging.info(f"Processing dataset '{ds_name}' with adapter '{adapter_name}' from {ds_dir}")

            entries = adapter_fn(ds_dir, class_mapping, excluded)
            manifest.extend(entries)
    else:
        logging.info("No datasets configured. Scanning raw_dir with generic adapter...")
        if os.path.exists(raw_dir):
            for subdir in sorted(os.listdir(raw_dir)):
                subdir_path = os.path.join(raw_dir, subdir)
                if os.path.isdir(subdir_path):
                    adapter_fn = DATASET_ADAPTERS.get(subdir, _process_generic)
                    entries = adapter_fn(subdir_path, class_mapping, excluded)
                    manifest.extend(entries)

    logging.info(f"Computing checksums for {len(manifest)} files...")
    for entry in tqdm(manifest, desc="Computing checksums"):
        entry["checksum"] = compute_checksum(entry["filepath"])

    manifest_path = os.path.join(raw_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)

    class_counts = {}
    for entry in manifest:
        cls = entry["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    logging.info(f"Data ingestion complete. Total files: {len(manifest)}")
    logging.info(f"Class distribution: {json.dumps(class_counts, indent=2)}")
    logging.info(f"Manifest saved at {manifest_path}")

    return manifest


if __name__ == "__main__":
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
    ingest_data("config.yaml")
