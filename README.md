# MotorSense — Real-Time Vibration Analysis with ML

ESP32 + ADXL345 accelerometer → real-time FFT + ML anomaly detection with per-company model adaptation.

## Quick Start

```bash
pip install -r motor_analyzer/requirements.txt
python motor_analyzer/data_ingestion.py    # ingest CWRU + JNU + Synthetic datasets
python motor_analyzer/train_pipeline.py     # train CompanyClassifier + per-company anomaly models
python motor_analyzer/app.py                # start Flask server → http://127.0.0.1:5050
```

## System Overview

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Sensor | ESP32 + ADXL345 | 3-axis accelerometer @ ~100 Hz |
| Backend | Flask + SocketIO + eventlet | Real-time streaming, REST API, WebSocket push |
| ML | scikit-learn | 28-dim feature extraction, RandomForest company classifier, IsolationForest + OC-SVM ensemble anomaly detection |
| Optional | PyTorch (unavailable on some Windows due to `shm.dll`) | CNNDetector + TransferLearningAdapter (falls back to sklearn) |
| Frontend | Chart.js | Real-time oscilloscope, FFT spectrum, anomaly alerts |
| Replay | CWRU dataset bridge | Downsample 12/48 kHz bearing data → 100 Hz, inject via HTTP/serial |

## Architecture

```
ESP32 ──serial──→ Flask (serial_reader thread)
                    │
                    ├─ feature_pipeline: 28-dim extractor
                    ├─ CompanyClassifier: identifies machine type
                    ├─ PerCompanyModelRegistry: loads per-company anomaly model
                    ├─ EnsembleAnomalyModel: IF + OC-SVM consensus
                    │
                    └─ SocketIO ──→ Web UI (Chart.js)
```

## Datasets

| Dataset | Source | Samples | Classes | Format |
|---------|--------|---------|---------|--------|
| CWRU | Case Western Reserve University | 2,130 | Normal, InnerRace, OuterRace, BallFault | `.mat` (12/48 kHz → resampled to 100 Hz) |
| JNU | Jiangnan University | 261 | Normal, InnerRace, OuterRace, BallFault | CSV (50 kHz → resampled to 100 Hz) |
| Synthetic | Generated | 2,400 | Normal, Unbalanced, BearingFault, Misalignment | Generated at 100 Hz |

**Company caveat**: "CWRU", "JNU", and "Synthetic" are different research labs — not real companies. 100% classification accuracy on public datasets is trivially achievable. For realistic benchmarks, use `evaluate_pipeline.py --company` which sweeps RPM overlap across synthetic companies (accuracy: 100% at 0 overlap → 68.5% at 1.0 overlap).

## Key Files

| File | Purpose |
|------|---------|
| `motor_analyzer/app.py` | Flask server with 23 REST endpoints + WebSocket |
| `motor_analyzer/feature_pipeline.py` | 28-dim feature extractor (8 time + 10 spectral + 10 frequency) |
| `motor_analyzer/ml_models.py` | CompanyClassifier, EnsembleAnomalyModel, GMM, TransferLearningAdapter, PerCompanyModelRegistry |
| `motor_analyzer/data_ingestion.py` | CWRU/JNU/Synthetic dataset ingestion with cross-dataset RMS normalization |
| `motor_analyzer/train_pipeline.py` | Training script (`--transfer` for TransferLearningAdapter) |
| `motor_analyzer/evaluate_pipeline.py` | End-to-end evaluation with overlap sweep, live simulation, model card |
| `motor_analyzer/cwru_replay.py` | CWRU dataset replay bridge (HTTP or serial mode) |
| `motor_analyzer/generate_synthetic_data.py` | Synthetic vibration generator with configurable overlap |
| `motor_analyzer/config.yaml` | Central configuration |
| `motor_analyzer/ESP32_DATA_GUIDE.md` | Guide for recording custom ESP32 data and retraining |

## API Routes

### Serial Connection
- `GET /api/ports` — List available serial ports
- `POST /api/connect` — Connect to serial port
- `POST /api/disconnect` — Disconnect serial port

### Training & Model Management
- `POST /api/train/start` — Start training capture
- `POST /api/train/cancel` — Cancel training
- `POST /api/model/save` — Save trained model to disk
- `POST /api/model/load` — Load saved model
- `GET /api/model/list` — List saved models
- `POST /api/model/delete` — Delete saved model

### Evaluation
- `POST /api/evaluate/stop` — Stop evaluation
- `GET /api/evaluate/report` — Get evaluation report
- `POST /api/evaluate/run` — Run evaluation pipeline
- `POST /api/evaluate/synthetic` — Generate synthetic evaluation data
- `POST /api/evaluate/run-tests` — Run pytest suite

### Company Awareness
- `GET /api/company/status` — Current company info
- `GET /api/company/models` — Per-company anomaly models
- `GET /api/company/classifier/status` — CompanyClassifier status
- `POST /api/company/reidentify` — Force re-identification

### System
- `GET /api/status` — Full system status
- `POST /api/motor/control` — Send motor command to ESP32

### Pretrained Weights (ml repo bridge)
- `GET /api/pretrained/info` — Transfer adapter + weights status
- `GET /api/pretrained/weights` — .pth file metadata

### Development
- `POST /api/inject` — Direct buffer injection (for CWRU replay bridge)

### WebSocket Events (SocketIO)
- `sensor_data` — Real-time waveform, FFT, anomaly score, company info
- `company_identified` — Company identification result
- `status_update` — System status changes
- `train_progress` — Training progress updates
- `train_complete` — Training completion
- `esp32_log` — ESP32 debug messages
- `motor_state` — Motor control state broadcasts

## Transfer Learning Adapter

Bridge pretrained 39-dim audio CNN weights (`model_best.pth` from [ml repo](https://github.com/Aditya-Patil27/ml)) to 28-dim vibration features:

- `Linear(28, 39)` expansion layer wraps frozen pretrained CNN
- sklearn RandomForest fallback when PyTorch unavailable (`shm.dll` error on Windows)
- Train with: `python train_pipeline.py --transfer`
- Evaluate with: `python evaluate_pipeline.py --transfer`

## CWRU Replay

Validate anomaly detection on real bearing data without hardware:

```bash
python cwru_replay.py --sequence --mode http   # recommended: Normal→train→fault sequence
python cwru_replay.py --file 97 --mode http     # single file replay
```

Requires app running at `http://127.0.0.1:5050`.

## Testing

```bash
python -m pytest motor_analyzer/tests/ -v    # 18 unit tests
python motor_analyzer/evaluate_pipeline.py    # full evaluation
python motor_analyzer/evaluate_pipeline.py --quick   # smoke test
```
