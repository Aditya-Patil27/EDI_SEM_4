"""
Unified feature extractor — merges motor_analyzer + ml repo approaches.
28-dim feature vector for streaming vibration data.
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew


def extract_features(data: list, baseline: float, fs: float = 100.0, n_fft: int = 128, n_bands: int = 10) -> list:
    """
    28-dim feature vector:
      [0:8]   Time-domain stats
      [8:18]  10 spectral sub-band energies  (from ml repo approach)
      [18:28] Frequency-domain stats + peaks (from motor_analyzer)
    """
    arr = np.array(data, dtype=np.float64) - baseline

    if len(arr) < 16 or np.std(arr) < 1e-10:
        return [0.0] * (8 + n_bands + 10)

    # ── Time Domain (8 features) ──
    rms = float(np.sqrt(np.mean(arr ** 2)))
    p2p = float(np.ptp(arr))
    variance = float(np.var(arr))
    skewness = float(skew(arr))
    kurt = float(kurtosis(arr))
    crest = float(np.max(np.abs(arr)) / (rms + 1e-8))
    shape_factor = rms / (np.mean(np.abs(arr)) + 1e-8)
    zcr = float(np.sum(np.diff(np.sign(arr)) != 0) / len(arr))

    # ── Frequency Domain ──
    win = signal.windows.hann(len(arr))
    spectrum = np.abs(rfft(arr * win))
    freqs = rfftfreq(len(arr), d=1.0 / fs)

    if len(spectrum) < 2:
        return [rms, p2p, variance, skewness, kurt, crest, shape_factor, zcr] + [0.0] * (n_bands + 10)

    # Spectral sub-band energies (ml repo style)
    band_size = max(1, len(spectrum) // n_bands)
    bands = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else len(spectrum)
        bands.append(float(np.sum(spectrum[start:end] ** 2) / (len(spectrum) + 1e-8)))

    # Dominant frequency
    dom_idx = int(np.argmax(spectrum[1:])) + 1
    dom_freq = float(freqs[dom_idx]) if dom_idx < len(freqs) else 0.0

    # Spectral centroid
    spec_sum = spectrum.sum() + 1e-8
    centroid = float(np.sum(freqs * spectrum) / spec_sum)

    # Spectral spread
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spec_sum))

    # Spectral flatness
    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-8)))
    arith_mean = np.mean(spectrum) + 1e-8
    flatness = float(geo_mean / arith_mean)

    # Band energy ratios (low/mid/high)
    low_mask  = freqs < fs * 0.1
    mid_mask  = (freqs >= fs * 0.1) & (freqs < fs * 0.3)
    high_mask = freqs >= fs * 0.3
    total_e = np.sum(spectrum ** 2) + 1e-8
    e_low  = float(np.sum(spectrum[low_mask]  ** 2) / total_e) if np.any(low_mask) else 0.0
    e_mid  = float(np.sum(spectrum[mid_mask]  ** 2) / total_e) if np.any(mid_mask) else 0.0
    e_high = float(np.sum(spectrum[high_mask] ** 2) / total_e) if np.any(high_mask) else 0.0

    # Top-3 spectral peaks
    sorted_idx = np.argsort(spectrum)[::-1]
    peak_freqs = [float(freqs[i]) if i < len(freqs) else 0.0 for i in sorted_idx[:3]]

    freq_domain = [
        dom_freq, centroid, spread, flatness,
        e_low, e_mid, e_high,
        *peak_freqs,
    ]

    return [
        rms, p2p, variance, skewness, kurt, crest, shape_factor, zcr,
        *bands,
        *freq_domain,
    ]


def dominant_frequency(data: list, baseline: float, fs: float = 100.0) -> float:
    if len(data) < 8:
        return 0.0
    arr = np.array(data, dtype=np.float64) - baseline
    win = signal.windows.hann(len(arr))
    spectrum = np.abs(rfft(arr * win))
    freqs = rfftfreq(len(arr), d=1.0 / fs)
    dom_idx = int(np.argmax(spectrum[1:])) + 1
    return float(freqs[dom_idx]) if dom_idx < len(freqs) else 0.0


def build_fft_payload(data: list, baseline: float, fs: float = 100.0) -> dict:
    if len(data) < 8:
        return {'freqs': [], 'magnitudes': [], 'dominant': 0.0}
    arr = np.array(data, dtype=np.float64) - baseline
    win = signal.windows.hann(len(arr))
    spectrum = np.abs(rfft(arr * win))
    freqs = rfftfreq(len(arr), d=1.0 / fs)
    dom_idx = int(np.argmax(spectrum[1:])) + 1
    return {
        'freqs': freqs.tolist(),
        'magnitudes': (spectrum / (spectrum.max() + 1e-8)).tolist(),
        'dominant': float(freqs[dom_idx]) if dom_idx < len(freqs) else 0.0,
    }
