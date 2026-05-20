# Architecture

## High-Level Data Flow

```
┌──────────┐   Serial (115200 baud)   ┌──────────────────────┐
│  ESP32   │ ──────────────────────→ │   Flask + SocketIO   │
│ ADXL345  │   CSV lines of float     │   eventlet thread    │
└──────────┘                          └──────────┬───────────┘
           ┌──────────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────┐
│              serial_reader thread                    │
│                                                      │
│  1. Raw float buffer (rolling window, 128 samples)   │
│  2. Baseline calibration (first 80 samples → mean)   │
│  3. Downsampled plot buffer (every 3rd sample)       │
│  4. Feature extraction every FEATURE_WINDOW (128)    │
│  5. Company fingerprinting (first N windows)         │
│  6. Anomaly prediction via active model              │
│  7. SocketIO emit → Web UI                            │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              28-dim Feature Vector                    │
│   [0:8]   Time-domain: RMS, P2P, var, skew, kurt,    │
│            crest factor, shape factor, ZCR             │
│   [8:18]  10 spectral sub-band energies               │
│  [18:28]  Frequency-domain: dom freq, centroid,       │
│            spread, flatness, low/mid/high ratios,      │
│            top-3 peak frequencies                      │
└──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              Company Classification                    │
│                                                       │
│  After fingerprint_samples (512 raw samples):         │
│    1. Average N feature windows                       │
│    2. CompanyClassifier.predict(avg_feat)             │
│    3. → (company_idx, company_name, confidence)       │
│    4. Load per-company anomaly model from registry    │
└──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              Anomaly Detection                         │
│                                                       │
│  EnsembleAnomalyModel:                                 │
│    IsolationForest (300 estimators, 3% contamination) │
│    + OneClassSVM (RBF kernel, nu=0.05)                │
│    → Both must agree = anomaly (reduces false alarms) │
│                                                       │
│  Score: sigmoid-normalized ensemble of IF score + SVM │
│                                                       │
│  Alternative: GMMAnomalyDetector (3-component GMM)    │
└──────────────────────────────────────────────────────┘
```

## Threading Model

- **Main thread**: Flask request handlers (REST API)
- **Serial reader thread** (daemon): Continuously reads from `pyserial`, processes samples, emits SocketIO events
- **SocketIO**: Async via `eventlet` (monkey-patched)

All state is shared via a global `state` dict + `ser_lock` for serial port access.

## CWRU Replay Bridge

```
cwru_replay.py ──HTTP POST /api/inject──→ Flask app
                                           │
  .mat file → scipy.io.loadmat()           │
  → find DE_time signal key                 │
  → downsample 12/48 kHz → 100 Hz           │
  → chunk into 100-sample groups            │
  → POST to /api/inject at 1s intervals     ▼
                                      raw_buffer
                                      (bypasses serial)
```

In `--sequence` mode, plays 4 stages: Normal → Inner Race → Ball Fault → Outer Race, with console prompts at each transition.

## Dataset Ingestion Pipeline

```
data/raw/
  cwru/           ← GitHub mirror structure (.mat files)
    Normal/
    12k_Drive_End_Bearing_Fault_Data/
      IR/007/, OR/021_6oclock/, B/014/
    48k_Drive_End_Bearing_Fault_Data/...
  jnu/             ← CSV files (n, ib, ob, tb prefixes)
  synthetic/       ← Generated on the fly

data_ingestion.py:
  _process_cwru()    → recursive walk → DE_time key → resample(100Hz) → window(128, stride 64) → features(28-dim)
  _process_jnu()     → sorted CSV list → resample(50kHz → 100Hz) → window → features
  _process_synthetic() → 4 patterns × N samples → window → features

Cross-dataset RMS normalization: feature[0] (RMS) scaled to mean of all datasets.

data/processed/
  X_all.npy       → (4791, 28) feature matrix
  y_all.npy       → class labels (Normal, InnerRace, ...)
  company_labels.npy → company index (0=CWRU, 1=JNU, 2=Synthetic)
  metadata.json   → companies, classes per company, dataset RMS
```

## Transfer Learning Adapter

```
Pretrained CNN weights (39-dim audio MFCC features)
  ┌──────────────────────────────────────────────┐
  │  model_best.pth / model_quantized.pth        │
  │  Architecture: conv1→pool→conv2→pool→fc1→fc2 │
  │  3 classes: Normal, Unbalanced, Bearing Fault │
  │  Source: github.com/Aditya-Patil27/ml         │
  └──────────────────────────────────────────────┘

TransferLearningAdapter:
  28-dim vibration → Linear(28,39) → ReLU → FrozenPretrainedCNN(39→3) → 3-class output

  When PyTorch unavailable: train sklearn RF directly (no transfer)

  Train: python train_pipeline.py --transfer
  Persists to: models/company_classifier_transfer.pth
```

## Model Persistence

```
models/
  company_classifier.pth              ← CompanyClassifier (pickled sklearn pipeline)
  company_classifier_transfer.pth     ← TransferLearningAdapter
  companies/
    manifest.json                     ← company list
    CWRU.pkl                          ← per-company anomaly model
    JNU.pkl
    Synthetic.pkl
```

## Frontend Architecture

```
templates/index.html
  ├── Header (brand + status pills)
  ├── Connection Drawer (port/baud select)
  ├── Status Bar
  ├── Left Panel
  │   ├── Metric Cards (dominant freq, anomaly score)
  │   ├── Machine Identity (company badge)
  │   ├── Anomaly Alert
  │   ├── Train Section (name, duration, progress)
  │   └── Save Section
  ├── Charts Area
  │   ├── Waveform (line chart, 200 points)
  │   └── FFT Spectrum (bar chart, normalized magnitude)
  └── Right Panel
      ├── Motor Control (3 buttons + speed slider)
      ├── Evaluate Section
      ├── Saved Models List
      └── System Log

static/js/main.js
  - SocketIO client for real-time events
  - Chart.js for both waveform and FFT
  - API helper: fetch wrapper
  - Toast notification system
  - Motor control UI state management
```
