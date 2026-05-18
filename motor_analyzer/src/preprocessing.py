"""
DEPRECATED — Legacy audio DSP pipeline (60-dim features, librosa-based).
NOT used by the ESP32 streaming pipeline (28-dim, numpy-only).
Replaced by:
  - feature_pipeline.py  (28-dim streaming feature extraction)
  - config.yaml streaming section
"""
import os
import yaml
import json
import logging
import librosa
import numpy as np
from scipy import signal
from scipy.stats import kurtosis
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_audio(y, sr, target_sr, bandpass_freq=(50, 3500)):
    """
    Resample with antialiasing Butterworth filter, DC removal, and bandpass.
    """
    # 1. Resample to 8000 Hz using resample_poly
    # resample_poly inherently applies optimal antialiasing filter unlike decimate
    if sr != target_sr:
        y_resampled = signal.resample_poly(y, target_sr, sr)
    else:
        y_resampled = y

    # 2. DC removal (subtract mean)
    y_dc_free = y_resampled - np.mean(y_resampled)
    
    # 3. 4th-order Butterworth bandpass filter
    nyq = 0.5 * target_sr
    low = bandpass_freq[0] / nyq
    high = bandpass_freq[1] / nyq
    # For very low frequencies like 50Hz, ensure low < high and high < 1
    if low > 0 and high < 1.0:
        b, a = signal.butter(4, [low, high], btype='bandpass')
        y_filtered = signal.filtfilt(b, a, y_dc_free)
    else:
        y_filtered = y_dc_free

    # 4. Amplitude Normalization (Peak Normalization)
    max_val = np.max(np.abs(y_filtered))
    if max_val > 0:
        y_normalized = y_filtered / max_val
    else:
        y_normalized = y_filtered

    return y_normalized

def extract_features(y_frame, sr, n_fft=256):
    """
    Extract 60 features from a single frame:
    - 6 time domain (RMS, Kurtosis, Crest, Skewness, ZCR, Energy)
    - 18 spectral bands
    - 13 MFCCs
    - 13 Delta MFCCs
    - 10 advanced spectral features (Centroid, Bandwidth, Rolloff, etc.)
    Total: 60 floats.
    """
    from scipy.stats import skew
    features = []
    
    # ===== Time-domain statistical features (6) =====
    rms = np.sqrt(np.mean(y_frame**2) + 1e-6)
    kurt = kurtosis(y_frame)
    crest = np.max(np.abs(y_frame)) / (rms + 1e-6)
    skewness = float(skew(y_frame))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_frame)[0])
    energy = np.sum(y_frame**2)
    features.extend([float(rms), float(kurt), float(crest), float(skewness), float(zcr), float(energy)])
    
    # ===== Spectral features from FFT (28 total) =====
    window = np.hanning(len(y_frame))
    y_windowed = y_frame * window
    fft_spec = np.abs(np.fft.rfft(y_windowed, n=n_fft))
    
    # 18 spectral sub-band energies
    step = len(fft_spec) // 18
    spectral_bands = [np.sum(fft_spec[i*step:(i+1)*step]) for i in range(18)]
    features.extend([float(b) for b in spectral_bands])
    
    # 10 advanced spectral features
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    # Spectral Centroid
    spectral_centroid = np.sum(freqs * fft_spec) / np.sum(fft_spec + 1e-6)
    features.append(float(spectral_centroid))
    
    # Spectral Bandwidth
    variance = np.sum(((freqs - spectral_centroid) ** 2) * fft_spec) / (np.sum(fft_spec) + 1e-6)
    spectral_bandwidth = np.sqrt(variance)
    features.append(float(spectral_bandwidth))
    
    # Spectral Rolloff (95% of energy)
    cumsum = np.cumsum(fft_spec)
    rolloff_bin = np.where(cumsum >= 0.95 * cumsum[-1])[0][0] if len(cumsum) > 0 else 0
    spectral_rolloff = freqs[rolloff_bin] if rolloff_bin < len(freqs) else 0
    features.append(float(spectral_rolloff))
    
    # Spectral Flux (change in spectrum)
    if len(y_frame) > 1:
        spectral_flux = np.sum(np.abs(np.diff(fft_spec)))
    else:
        spectral_flux = 0
    features.append(float(spectral_flux))
    
    # Peak Frequency
    peak_freq = freqs[np.argmax(fft_spec)] if len(fft_spec) > 0 else 0
    features.append(float(peak_freq))
    
    # Spectral Contrast (simplified - top 2 peaks)
    top_peaks_idx = np.argsort(fft_spec)[-2:] if len(fft_spec) > 2 else [0, 1]
    spectral_contrast = float(np.mean([fft_spec[i] for i in top_peaks_idx]) if len(fft_spec) > 0 else 0)
    features.append(float(spectral_contrast))
    
    # Sum of spectral energy
    total_energy_spectral = np.sum(fft_spec)
    features.append(float(total_energy_spectral))
    
    # Mean spectral magnitude
    mean_spec_mag = np.mean(fft_spec)
    features.append(float(mean_spec_mag))
    
    # Spectral Spread
    spectral_spread = np.sqrt(np.sum(fft_spec) / np.std(fft_spec + 1e-6))
    features.append(float(spectral_spread))
    
    # Spectral Crest Factor (max / mean)
    spectral_crest = np.max(fft_spec) / (np.mean(fft_spec) + 1e-6)
    features.append(float(spectral_crest))
    
    # ===== MFCC features (13) =====
    mfccs = librosa.feature.mfcc(y=y_frame, sr=sr, n_fft=n_fft, hop_length=n_fft+1, n_mfcc=13)
    mfccs_1d = mfccs.flatten()[:13]
    if len(mfccs_1d) < 13:
        mfccs_1d = np.pad(mfccs_1d, (0, 13-len(mfccs_1d)))
    features.extend(mfccs_1d.tolist())
    
    # ===== Delta MFCC features (13) =====
    delta_mfccs = librosa.feature.delta(mfccs, order=1, mode='nearest').flatten()[:13]
    if len(delta_mfccs) < 13:
        delta_mfccs = np.pad(delta_mfccs, (0, 13-len(delta_mfccs)))
    features.extend(delta_mfccs.tolist())
    
    # Total: 6 + 18 + 10 + 13 + 13 = 60
    assert len(features) == 60, f"Expected 60 features, got {len(features)}"
    return np.array(features, dtype=np.float32)

def sliding_window_extract(y, sr, window_size=256, overlap=0.5):
    """
    Extract sliding frames with given overlap.
    """
    hop_length = int(window_size * (1 - overlap))
    frames = librosa.util.frame(y, frame_length=window_size, hop_length=hop_length).T
    
    all_features = []
    for frame in frames:
        # Check for silence/padded zero fragments
        if np.max(np.abs(frame)) < 1e-4:
            continue
            
        feats = extract_features(frame, sr, n_fft=window_size)
        all_features.append(feats)
    
    return np.array(all_features) if all_features else np.empty((0, 60))

def process_pipeline():
    config = load_config()
    target_sr = config['data']['target_sr']
    bandpass = tuple(config['dsp']['bandpass'])
    w_size = config['dsp']['window_size']
    w_overlap = config['dsp']['overlap']
    
    # Read manifest
    manifest_path = os.path.join(config['data']['raw_dir'], 'manifest.json')
    if not os.path.exists(manifest_path):
        print("No manifest found. Please run data_ingestion.py")
        return
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    # Prepare processed dir
    proc_dir = config['data']['processed_dir']
    os.makedirs(proc_dir, exist_ok=True)
    
    dataset_features = []
    dataset_labels = []
    
    for item in tqdm(manifest, desc="Processing audio"):
        file_path = item['filepath']
        label = item['class']
        fmt = item.get('format', 'wav')

        try:
            if fmt == 'wav':
                y, sr = librosa.load(file_path, sr=None, mono=True)

            elif fmt == 'csv_vibration':
                import pandas as pd
                df = pd.read_csv(file_path)
                vib_cols = [c for c in df.columns if 'vibration' in c.lower() or 'vib' in c.lower()]
                if vib_cols:
                    y = df[vib_cols[0]].values.astype(np.float32)
                else:
                    y = df.iloc[:, -1].values.astype(np.float32)
                sr = 4096

            elif fmt == 'csv':
                import pandas as pd
                df = pd.read_csv(file_path, header=None)
                y = df.iloc[:, 0].values.astype(np.float32)
                sr = 50000

            elif fmt == 'raw_binary':
                raw = np.fromfile(file_path, dtype=np.float64, sep='\t')
                if raw.size == 0:
                    try:
                        raw = np.loadtxt(file_path)
                    except Exception:
                        continue
                if raw.ndim > 1:
                    y = raw[:, 0].astype(np.float32)
                else:
                    y = raw.astype(np.float32)
                sr = 20480

            else:
                y, sr = librosa.load(file_path, sr=None, mono=True)

        except Exception as e:
            logging.warning(f"Failed to load {file_path}: {e}")
            continue

        if len(y) < w_size:
            continue

        y_proc = preprocess_audio(y, sr, target_sr, bandpass_freq=bandpass)

        X = sliding_window_extract(y_proc, target_sr, window_size=w_size, overlap=w_overlap)

        if len(X) > 0:
            dataset_features.append(X)
            dataset_labels.extend([label] * len(X))
            
    if dataset_features:
        X_all = np.vstack(dataset_features)
        y_all = np.array(dataset_labels)
        np.save(os.path.join(proc_dir, "X_features.npy"), X_all)
        np.save(os.path.join(proc_dir, "y_labels.npy"), y_all)
        print(f"Produced {X_all.shape[0]} feature vectors. Saved to {proc_dir}.")
        
        # Save preproc params log
        params = {
            "target_sr": target_sr,
            "bandpass_hz": bandpass,
            "window_size": w_size,
            "overlap": w_overlap,
            "feature_dim": 39
        }
        with open(os.path.join(proc_dir, "preproc_params.json"), "w") as f:
            json.dump(params, f, indent=4)
    else:
        print("No feature vectors generated. Raw data might be empty.")

if __name__ == "__main__":
    process_pipeline()
