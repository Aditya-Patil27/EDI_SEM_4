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


def generate_companies_dataset(n_companies: int = 3, samples_per_company: int = 50,
                                seed: int = 42, overlap: float = 0.0):
    """
    Generate labeled synthetic data for training the company classifier.
    
    Args:
        overlap: 0=well-separated (F1~1.0), 0.5=moderate (F1~0.93), 1.0=heavy (F1~0.75).
                 Controls how much RPM ranges overlap between companies.
    """
    np.random.seed(seed)
    companies = [f"Company_{chr(65+i)}" for i in range(n_companies)]
    # Aggressive overlap: spread goes from 200 down to 10
    rpm_spread = max(10, int(200 * (1 - overlap * 0.95)))
    base_rpm = 1500
    # Per-company base RPM with overlap
    company_centers = [base_rpm + i * rpm_spread for i in range(n_companies)]
    # Noise levels
    noise_base = 0.05 + overlap * 0.08
    noise_spread = max(0.005, 0.03 * (1 - overlap))

    all_features = []
    all_labels = []
    window_size = 128

    for ci, company in enumerate(companies):
        for _ in range(samples_per_company):
            # Per-sample RPM jitter: ~30% of between-company spacing
            jitter_max = max(5, int(rpm_spread * 0.3))
            rpm = max(300, company_centers[ci] + np.random.randint(-jitter_max, jitter_max))
            nl = noise_base + np.random.uniform(-noise_spread, noise_spread)
            raw = normal_vibration(5.0, rpm=rpm, noise_level=max(0.01, nl))
            for start in range(0, len(raw) - window_size, window_size // 2):
                chunk = raw[start:start + window_size]
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
