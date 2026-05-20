# Configuration Reference

Central config file: `motor_analyzer/config.yaml`

## Full Schema

```yaml
project:
  name: "MotorSense"          # Project identifier
  seed: 42                    # Global random seed for reproducibility

companies:
  classes:                    # Ordered list of company/machine IDs
    - "Unknown"               #   Index 0 = unknown/unidentified
    - "CWRU"
    - "JNU"
    - "Synthetic"
  fingerprint_samples: 512    # Raw samples needed before company ID triggers (~5s @ 100 Hz)
  use_synthetic: true         # Include synthetic data in training; set false once real ESP32 data collected

feature:
  window_size: 128            # Samples per feature window (~1.28s @ 100 Hz)
  sample_rate: 100            # Target sample rate (Hz) — ESP32 ADXL345 output
  n_fft: 128                  # FFT window size (must match window_size)
  dimension: 28               # Output feature vector length (8 time + 10 spectral + 10 frequency)
  spectral_bands: 10          # Number of spectral sub-band energy bins

model:
  cnn:
    conv1_channels: 16        # CNN first conv layer output channels
    conv2_channels: 32        # CNN second conv layer output channels
    kernel_size: 3            # Conv1D kernel size
    fc1_units: 64             # Fully connected hidden units
    dropout: 0.2              # Dropout rate
  anomaly:
    confidence_threshold: 0.90  # Min confidence for anomaly classification
    consensus_windows: 5        # Number of consecutive windows for consensus voting
  company_classifier_path: "models/company_classifier.pth"  # Saved classifier path

training:
  batch_size: 32              # Training batch size (PyTorch path only)
  epochs: 50                  # Max training epochs
  learning_rate: 0.001        # Adam optimizer learning rate
  patience: 10                # Early stopping patience
  company_noise_scale: 0.0    # Gaussian noise multiplier for overlap robustness
                              # 0=perfect separation on public datasets
                              # 0.3-0.5 = realistic overlap benchmark
                              # NOTE: Use evaluate_pipeline.py --company for overlap sweep

paths:
  models_dir: "models"
  company_models_dir: "models/companies"
  cwru_data_dir: "data/raw/cwru"

replay:
  target_fs: 100              # Target sample rate for CWRU replay (Hz)
  chunk_size: 100             # Samples per HTTP injection chunk
  app_url: "http://127.0.0.1:5050"  # Flask app URL

pretrained:
  model_best: "model_best.pth"         # Best CNN weights (from ml repo)
  model_quantized: "model_quantized.pth"  # Quantized weights
  source: "ml_repo"                    # Source identifier
  input_dim: 39                        # Pretrained model input dimension (MFCC)
  num_classes: 3                       # Pretrained model output classes
  classes: ["Normal", "Unbalanced", "Bearing Fault"]
  adapter:
    adapt_dim: 28                      # Current feature dimension to adapt
    enabled: true                      # Enable adapter in app.py
    fallback_to_sklearn: true          # Train sklearn RF when torch unavailable
```

## Important Flags

| Key | Effect |
|-----|--------|
| `companies.use_synthetic` | `true` = all 3 datasets; `false` = CWRU + JNU only (when real data collected) |
| `training.company_noise_scale` | `0` = 100% accuracy on public data; `0.3-0.5` = realistic per-company overlap |
| `pretrained.adapter.fallback_to_sklearn` | If `true`, trains `RandomForest` when `torch` import fails (`shm.dll` error) |

## Feature Dimensions

| Index | Domain | Feature |
|-------|--------|---------|
| 0 | Time | RMS (root mean square) |
| 1 | Time | Peak-to-peak amplitude |
| 2 | Time | Variance |
| 3 | Time | Skewness |
| 4 | Time | Kurtosis |
| 5 | Time | Crest factor |
| 6 | Time | Shape factor |
| 7 | Time | Zero-crossing rate |
| 8–17 | Spectral | 10 sub-band energies (evenly split FFT bins) |
| 18 | Frequency | Dominant frequency |
| 19 | Frequency | Spectral centroid |
| 20 | Frequency | Spectral spread |
| 21 | Frequency | Spectral flatness |
| 22 | Frequency | Low band energy ratio (< 10 Hz) |
| 23 | Frequency | Mid band energy ratio (10–30 Hz) |
| 24 | Frequency | High band energy ratio (> 30 Hz) |
| 25–27 | Frequency | Top-3 spectral peak frequencies |
