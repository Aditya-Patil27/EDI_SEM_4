"""
Synthetic vibration data generator.
Creates known vibration patterns to test the system without hardware.
Patterns: normal, unbalanced, bearing fault, misalignment.
"""

import numpy as np
import os
import pickle
import json
from feature_pipeline import extract_features


SAMPLE_RATE = 100  # Hz, matching ESP32 ADXL345 output


def normal_vibration(duration_sec: float, rpm: float = 1800, noise_level: float = 0.05) -> np.ndarray:
    """Healthy motor — clean sine at fundamental frequency + low noise."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    fund_freq = rpm / 60.0
    signal = np.sin(2 * np.pi * fund_freq * t)
    signal += 0.02 * np.sin(2 * np.pi * 2 * fund_freq * t)  # tiny 2nd harmonic
    signal += np.random.normal(0, noise_level, len(t))
    return signal


def unbalanced_vibration(duration_sec: float, rpm: float = 1800, severity: float = 0.3) -> np.ndarray:
    """Unbalanced rotor — strong 1x RPM harmonic + modulated amplitude."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    fund_freq = rpm / 60.0
    signal = np.sin(2 * np.pi * fund_freq * t)
    signal += severity * np.sin(2 * np.pi * 2 * fund_freq * t)  # strong 2nd harmonic
    signal += 0.15 * np.sin(2 * np.pi * 3 * fund_freq * t)
    # Amplitude modulation at RPM
    signal *= (1 + 0.2 * np.sin(2 * np.pi * fund_freq * t))
    signal += np.random.normal(0, 0.03, len(t))
    return signal


def bearing_fault_vibration(duration_sec: float, rpm: float = 1800) -> np.ndarray:
    """Bearing fault — impulse train (BPFI/BPFO) + high-frequency ringing."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    fund_freq = rpm / 60.0
    bpfo = fund_freq * 3.5  # typical ball-pass freq outer race

    signal = 0.3 * np.sin(2 * np.pi * fund_freq * t)  # fundamental
    # Impulse train at BPFO
    impulse_step = max(1, int(round(SAMPLE_RATE / max(bpfo, 1))))
    impulse_idx = np.arange(0, len(t), impulse_step)
    for idx in impulse_idx:
        if idx < len(t):
            remaining = len(t) - idx
            env_len = min(15, remaining)
            envelope = np.exp(-5 * np.arange(env_len) / SAMPLE_RATE)
            signal[idx:idx+env_len] += 0.5 * envelope

    signal += np.random.normal(0, 0.04, len(t))
    return signal


def misalignment_vibration(duration_sec: float, rpm: float = 1800) -> np.ndarray:
    """Misalignment — strong 2x RPM + 3x RPM harmonics."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    fund_freq = rpm / 60.0
    signal = 0.2 * np.sin(2 * np.pi * fund_freq * t)
    signal += 0.8 * np.sin(2 * np.pi * 2 * fund_freq * t)  # dominant 2x
    signal += 0.4 * np.sin(2 * np.pi * 3 * fund_freq * t)
    signal += np.random.normal(0, 0.03, len(t))
    return signal


def generate_companies_dataset(n_companies: int = 5, samples_per_company: int = 40,
                                seed: int = 42, overlap: float = 0.0):
    """
    Generate labeled synthetic data for training the company classifier.

    Each company has a unique acoustic fingerprint that varies per-sample
    (simulating unit-to-unit variation). At overlap=0.0, RPM bands are exclusive
    → ~100% accuracy. At overlap=1.0, same RPM band → classifier uses subtle
    but noisy fingerprints (~60-72% accuracy on real data).

    Yields exactly 1 feature vector per sample (no overlapping windows) so
    training instances are independent.

    Args:
        overlap: 0.0 = well-separated RPM bands, 1.0 = full RPM overlap.
    """
    rng = np.random.RandomState(seed)
    companies = [f"Company_{chr(65+i)}" for i in range(n_companies)]
    window_size = 128
    n_time = 256  # 2.56s of signal for spectral development; window 0:128 used

    per_company_span = 100
    base_rpm = 1300

    if overlap >= 1.0:
        shared_center = base_rpm + per_company_span * n_companies / 2
        centers = [shared_center for _ in range(n_companies)]
    else:
        gap = per_company_span * (1 - overlap * 0.95)
        spacing = max(10, per_company_span + gap)
        centers = [base_rpm + i * spacing + per_company_span / 2 for i in range(n_companies)]

    jitter = per_company_span * 0.5 * (1 + overlap * 2)

    base_profiles = []
    for ci in range(n_companies):
        prng = np.random.RandomState(42 + ci * 7)
        base_profiles.append({
            'harmonics': [1.0] + [prng.uniform(0.08, 0.18) for _ in range(3)],
            'pink_weight': prng.uniform(0.3, 0.6),
            'mod_freq': prng.uniform(2.0, 5.0),
            'phase': prng.uniform(0, 2 * np.pi, 4),
        })

    all_features = []
    all_labels = []

    for ci in range(n_companies):
        base = base_profiles[ci]
        for _ in range(samples_per_company):
            rpm = centers[ci] + rng.uniform(-jitter, jitter)
            rpm = max(300, rpm)
            freq = rpm / 60.0

            t = np.arange(n_time) / SAMPLE_RATE

            # Per-sample profile jitter (unit-to-unit variation)
            harm_jitter = rng.normal(0, 0.04, 4)
            harmonics = [max(0, base['harmonics'][h] + harm_jitter[h]) for h in range(4)]
            pink_w = np.clip(base['pink_weight'] + rng.normal(0, 0.15), 0.05, 0.95)
            mod_f = max(0.5, base['mod_freq'] + rng.normal(0, 0.5))

            sig = np.zeros(n_time)
            for h in range(4):
                sig += harmonics[h] * np.sin(2 * np.pi * freq * (h + 1) * t + base['phase'][h])

            pink = np.cumsum(rng.normal(0, 0.01, n_time))
            white = rng.normal(0, 0.02, n_time)
            noise = pink_w * pink + (1 - pink_w) * white

            mod = 0.02 * np.sin(2 * np.pi * mod_f * t)

            noise_std = 0.15 * (1 + overlap * 2)
            rand_noise = rng.normal(0, noise_std, n_time)

            sig = sig + noise + mod + rand_noise
            sig = sig / max(np.std(sig), 1e-8)

            # One window per sample (no overlap) -> independent observations
            for start in range(0, n_time, window_size):
                chunk = sig[start:start + window_size]
                if len(chunk) < window_size:
                    continue
                baseline = np.mean(chunk)
                feats = extract_features(chunk, baseline, SAMPLE_RATE)
                all_features.append(feats)
                all_labels.append(ci)

    return np.array(all_features), np.array(all_labels), companies


def generate_fault_dataset(samples_per_class: int = 30, seed: int = 42):
    """
    Generate labeled synthetic data for fault detection evaluation.
    Classes: Normal (0), Unbalanced (1), Bearing Fault (2), Misalignment (3)
    """
    np.random.seed(seed)
    generators = [normal_vibration, unbalanced_vibration, bearing_fault_vibration, misalignment_vibration]
    class_names = ["Normal", "Unbalanced", "Bearing Fault", "Misalignment"]

    all_features = []
    all_labels = []
    window_size = 128

    for ci, gen_fn in enumerate(generators):
        for _ in range(samples_per_class):
            raw = gen_fn(5.0)
            for start in range(0, len(raw) - window_size, window_size // 2):
                chunk = raw[start:start + window_size]
                baseline = np.mean(chunk)
                feats = extract_features(chunk, baseline, SAMPLE_RATE)
                all_features.append(feats)
                all_labels.append(ci)

    return np.array(all_features), np.array(all_labels), class_names


def save_synthetic_dataset(output_dir: str = "data/synthetic"):
    """Generate and save both company and fault datasets."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating company classification dataset...")
    X, y, companies = generate_companies_dataset()
    np.save(os.path.join(output_dir, "X_company.npy"), X)
    np.save(os.path.join(output_dir, "y_company.npy"), y)
    with open(os.path.join(output_dir, "companies.json"), "w") as f:
        json.dump(companies, f)
    print(f"  {len(X)} samples, {len(companies)} companies -> {output_dir}")

    print("Generating fault classification dataset...")
    Xf, yf, classes = generate_fault_dataset()
    np.save(os.path.join(output_dir, "X_fault.npy"), Xf)
    np.save(os.path.join(output_dir, "y_fault.npy"), yf)
    with open(os.path.join(output_dir, "fault_classes.json"), "w") as f:
        json.dump(classes, f)
    print(f"  {len(Xf)} samples, {len(classes)} classes -> {output_dir}")

    # Also create a live-stream simulation file (single long sequence)
    print("Generating live simulation sequence...")
    seq = []
    labels_seq = []
    for ci, gen_fn in enumerate([normal_vibration, unbalanced_vibration, bearing_fault_vibration]):
        raw = gen_fn(10.0)
        seq.append(raw)
        labels_seq.extend([ci] * len(raw))
    full_seq = np.concatenate(seq)
    np.save(os.path.join(output_dir, "live_sequence.npy"), full_seq)
    np.save(os.path.join(output_dir, "live_labels.npy"), np.array(labels_seq[:len(full_seq)]))
    print(f"  Live sequence: {len(full_seq)} samples -> {output_dir}")

    print(f"\nSaved to {output_dir}/")
    return output_dir


if __name__ == "__main__":
    save_synthetic_dataset()
    print("Done.")
