"""
Unit tests for feature_pipeline.py — validates feature extraction correctness.
Run with: python -m pytest tests/test_feature_pipeline.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from feature_pipeline import extract_features, dominant_frequency, build_fft_payload

SAMPLE_RATE = 100


def test_extract_features_returns_28_values():
    """Feature extractor must return exactly 28 floats."""
    data = np.sin(2 * np.pi * 10 * np.linspace(0, 1.28, 128))
    feats = extract_features(data, np.mean(data), SAMPLE_RATE)
    assert len(feats) == 28, f"Expected 28 features, got {len(feats)}"


def test_extract_features_no_baseline():
    """Zero-baseline data should produce stable features."""
    np.random.seed(42)
    data = np.random.normal(0, 0.1, 128)
    feats = extract_features(data, 0.0, SAMPLE_RATE)
    assert all(np.isfinite(f) for f in feats), "Non-finite feature detected"
    assert feats[0] > 0, "RMS should be positive"  # RMS


def test_extract_features_baseline_correction():
    """Subtracting baseline should center the signal."""
    data = np.ones(128) * 5.0 + np.random.normal(0, 0.01, 128)
    feats = extract_features(data, 5.0, SAMPLE_RATE)
    assert all(np.isfinite(f) for f in feats)
    assert feats[0] < 0.1, "RMS should be near zero after baseline subtraction"  # RMS


def test_known_sine_wave():
    """A pure sine wave should have specific characteristics."""
    t = np.linspace(0, 1.28, 128, endpoint=False)
    freq = 10.0
    sine = np.sin(2 * np.pi * freq * t)
    feats = extract_features(sine, np.mean(sine), SAMPLE_RATE)

    # Dominant frequency should be near 10 Hz
    dom_freq = feats[18]  # spectral features start at index 8+10=18
    assert abs(dom_freq - freq) < 2.0, f"Dominant freq {dom_freq} != {freq}"


def test_different_amplitudes():
    """Louder signal = higher RMS, same frequency content."""
    t = np.linspace(0, 1.28, 128)
    quiet = 0.1 * np.sin(2 * np.pi * 10 * t)
    loud = 1.0 * np.sin(2 * np.pi * 10 * t)

    feats_q = extract_features(quiet, np.mean(quiet), SAMPLE_RATE)
    feats_l = extract_features(loud, np.mean(loud), SAMPLE_RATE)

    assert feats_l[0] > feats_q[0], "Louder signal should have higher RMS"


def test_extract_features_short_buffer():
    """Very short buffer should still work (fallback to zeros)."""
    data = [1.0, 2.0, 3.0]
    feats = extract_features(data, 0.0, SAMPLE_RATE)
    assert len(feats) == 28
    assert all(np.isfinite(f) for f in feats)


def test_extract_features_empty():
    """Empty buffer should produce all zeros."""
    feats = extract_features([], 0.0, SAMPLE_RATE)
    assert len(feats) == 28
    assert all(f == 0.0 for f in feats)


def test_dominant_frequency_sine():
    """Dominant frequency of a pure sine should match."""
    t = np.linspace(0, 1.28, 128, endpoint=False)
    sine = np.sin(2 * np.pi * 15 * t)
    dom = dominant_frequency(sine, np.mean(sine), SAMPLE_RATE)
    assert abs(dom - 15) < 2.0, f"Dominant freq {dom} != 15 Hz"


def test_dominant_frequency_empty():
    """Empty data should return 0.0."""
    assert dominant_frequency([], 0.0, SAMPLE_RATE) == 0.0


def test_build_fft_payload_structure():
    """FFT payload must have correct keys."""
    data = np.sin(2 * np.pi * 10 * np.linspace(0, 1.28, 128))
    payload = build_fft_payload(data, np.mean(data), SAMPLE_RATE)
    assert 'freqs' in payload
    assert 'magnitudes' in payload
    assert 'dominant' in payload
    assert len(payload['freqs']) == len(payload['magnitudes'])


def test_different_fault_patterns():
    """Normal vs Unbalanced vs Bearing should have distinguishable features."""
    from generate_synthetic_data import normal_vibration, unbalanced_vibration, bearing_fault_vibration

    np.random.seed(42)
    normal = extract_features(normal_vibration(2.0)[:128], 0, SAMPLE_RATE)
    unbalanced = extract_features(unbalanced_vibration(2.0)[:128], 0, SAMPLE_RATE)
    bearing = extract_features(bearing_fault_vibration(2.0)[:128], 0, SAMPLE_RATE)

    # Normal should have lower RMS than unbalanced/bearing
    assert normal[0] < unbalanced[0], "Unbalanced should have higher RMS than Normal"
    assert normal[0] < bearing[0], "Bearing fault should have higher RMS than Normal"
