"""
Legacy 60-dim feature extraction (from original audio pipeline).
Includes: 6 time-domain + 18 spectral bands + 10 advanced spectral + 13 MFCC + 13 Delta MFCC.
"""

import os
import json
import yaml
import logging
import librosa
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess_audio(y, sr, target_sr, bandpass=(50, 3500)):
    if sr != target_sr:
        y = signal.resample_poly(y, target_sr, sr)
    y = y - np.mean(y)
    nyq = 0.5 * target_sr
    low = bandpass[0] / nyq
    high = bandpass[1] / nyq
    if 0 < low < high < 1.0:
        b, a = signal.butter(4, [low, high], btype="bandpass")
        y = signal.filtfilt(b, a, y)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y


def extract_features_60(y_frame, sr, n_fft=256):
    features = []
    rms = np.sqrt(np.mean(y_frame ** 2) + 1e-6)
    kurt = kurtosis(y_frame)
    crest = np.max(np.abs(y_frame)) / (rms + 1e-6)
    skewness = float(skew(y_frame))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_frame)[0])
    energy = np.sum(y_frame ** 2)
    features.extend([float(rms), float(kurt), float(crest), float(skewness), float(zcr), float(energy)])

    window = np.hanning(len(y_frame))
    fft_spec = np.abs(np.fft.rfft(y_frame * window, n=n_fft))
    step = len(fft_spec) // 18
    spectral_bands = [np.sum(fft_spec[i*step:(i+1)*step]) for i in range(18)]
    features.extend([float(b) for b in spectral_bands])

    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    sc = np.sum(freqs * fft_spec) / np.sum(fft_spec + 1e-6)
    features.append(float(sc))
    variance = np.sum(((freqs - sc) ** 2) * fft_spec) / (np.sum(fft_spec) + 1e-6)
    features.append(float(np.sqrt(variance)))
    cumsum = np.cumsum(fft_spec)
    rolloff_bin = np.where(cumsum >= 0.95 * cumsum[-1])[0][0] if len(cumsum) > 0 else 0
    features.append(float(freqs[rolloff_bin]) if rolloff_bin < len(freqs) else 0)
    features.append(float(np.sum(np.abs(np.diff(fft_spec)))) if len(y_frame) > 1 else 0)
    features.append(float(freqs[np.argmax(fft_spec)]) if len(fft_spec) > 0 else 0)
    top_peaks = np.argsort(fft_spec)[-2:] if len(fft_spec) > 2 else [0, 1]
    features.append(float(np.mean([fft_spec[i] for i in top_peaks])))
    features.append(float(np.sum(fft_spec)))
    features.append(float(np.mean(fft_spec)))
    features.append(float(np.sqrt(np.sum(fft_spec) / np.std(fft_spec + 1e-6))))
    features.append(float(np.max(fft_spec) / (np.mean(fft_spec) + 1e-6)))

    mfccs = librosa.feature.mfcc(y=y_frame, sr=sr, n_fft=n_fft, hop_length=n_fft+1, n_mfcc=13).flatten()[:13]
    if len(mfccs) < 13:
        mfccs = np.pad(mfccs, (0, 13 - len(mfccs)))
    features.extend(mfccs.tolist())

    delta = librosa.feature.delta(mfccs.reshape(1, -1), order=1, mode="nearest").flatten()[:13]
    if len(delta) < 13:
        delta = np.pad(delta, (0, 13 - len(delta)))
    features.extend(delta.tolist())

    assert len(features) == 60, f"Expected 60 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


def sliding_window(y, sr, window_size=256, overlap=0.5):
    hop = int(window_size * (1 - overlap))
    frames = librosa.util.frame(y, frame_length=window_size, hop_length=hop).T
    results = []
    for frame in frames:
        if np.max(np.abs(frame)) < 1e-4:
            continue
        results.append(extract_features_60(frame, sr, n_fft=window_size))
    return np.array(results) if results else np.empty((0, 60))


def run_pipeline(config_path="config.yaml"):
    config = load_config(config_path)
    target_sr = config["data"]["target_sr"]
    bandpass = tuple(config["dsp"]["bandpass"])
    w_size = config["dsp"]["window_size"]
    w_overlap = config["dsp"]["overlap"]

    manifest_path = os.path.join(config["data"]["raw_dir"], "manifest.json")
    if not os.path.exists(manifest_path):
        print("No manifest. Run data ingestion first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    proc_dir = config["data"]["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    X_all, y_all = [], []
    for item in tqdm(manifest, desc="Processing"):
        fpath = item["filepath"]
        label = item["class"]
        fmt = item.get("format", "wav")

        try:
            if fmt == "wav":
                y, sr = librosa.load(fpath, sr=None, mono=True)
            elif fmt == "csv_vibration":
                import pandas as pd
                df = pd.read_csv(fpath)
                vib_cols = [c for c in df.columns if "vibration" in c.lower() or "vib" in c.lower()]
                y = df[vib_cols[0]].values.astype(np.float32) if vib_cols else df.iloc[:, -1].values.astype(np.float32)
                sr = 4096
            elif fmt == "csv":
                y = pd.read_csv(fpath, header=None).iloc[:, 0].values.astype(np.float32)
                sr = 50000
            elif fmt == "raw_binary":
                raw = np.fromfile(fpath, dtype=np.float64, sep="\t")
                if raw.size == 0:
                    raw = np.loadtxt(fpath)
                y = raw[:, 0].astype(np.float32) if raw.ndim > 1 else raw.astype(np.float32)
                sr = 20480
            else:
                y, sr = librosa.load(fpath, sr=None, mono=True)
        except Exception as e:
            logging.warning(f"Skipping {fpath}: {e}")
            continue

        if len(y) < w_size:
            continue

        y = preprocess_audio(y, sr, target_sr, bandpass)
        X = sliding_window(y, target_sr, w_size, w_overlap)
        if len(X) > 0:
            X_all.append(X)
            y_all.extend([label] * len(X))

    if X_all:
        X_all = np.vstack(X_all)
        y_all = np.array(y_all)
        np.save(os.path.join(proc_dir, "X_features.npy"), X_all)
        np.save(os.path.join(proc_dir, "y_labels.npy"), y_all)
        params = {"target_sr": target_sr, "bandpass_hz": list(bandpass), "window_size": w_size, "overlap": w_overlap, "feature_dim": 60}
        with open(os.path.join(proc_dir, "preproc_params.json"), "w") as f:
            json.dump(params, f, indent=4)
        print(f"Saved {X_all.shape[0]} feature vectors -> {proc_dir}")
    else:
        print("No features generated.")
