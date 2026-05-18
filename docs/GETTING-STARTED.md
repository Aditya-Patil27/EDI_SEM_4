# Getting Started

## Prerequisites

- Python 3.9+
- ESP32 with ADXL345 accelerometer (optional — CWRU replay works without hardware)
- Windows/Linux/macOS

## Installation

```bash
# Clone the repo
cd motor_analyzer

# Install dependencies
pip install -r requirements.txt
```

**Note**: `torch` is listed in requirements but may fail on Windows (`OSError: [WinError 126] shm.dll not found`). The system falls back to sklearn automatically — the dependency is optional.

## Quick Start (No Hardware)

```bash
# 1. Ingest public datasets
python data_ingestion.py
# Expected: "4791 total windows across 3 companies"

# 2. Train models
python train_pipeline.py
# Expected output:
#   Test accuracy: ~1.0 (CWRU+JNU+Synthetic — see company caveat)
#   Anomaly models saved to models/companies/

# 3. Start the app
python app.py
# → http://127.0.0.1:5050

# 4. Run CWRU replay (separate terminal)
python cwru_replay.py --sequence --mode http
# Follow console prompts — train on Normal data, observe anomaly scores on faults
```

## With ESP32 Hardware

1. Flash ESP32 with ADXL345 code that prints raw accelerometer values as CSV lines over serial
2. Connect ESP32 via USB
3. Start app: `python app.py`
4. Open http://127.0.0.1:5050
5. Select serial port, click Connect
6. Hold motor still for baseline calibration (2-3 seconds)
7. Click Start Training, run motor normally for 20s
8. Introduce faults, observe anomaly scores

## Recording Custom Data for Retraining

See `ESP32_DATA_GUIDE.md` for detailed instructions on:
- Recording labeled vibration data from ESP32
- Converting recordings to the training format
- Retraining per-company models
- Disabling synthetic data once real data is collected (`use_synthetic: false` in config.yaml)

## Run Tests

```bash
python -m pytest tests/ -v
# 11 feature pipeline tests + 7 ML model tests = 18 total
```

## Run Full Evaluation

```bash
python evaluate_pipeline.py
# Generates evaluation_report.json with feature benchmarks,
# company classifier overlap sweep, anomaly detection metrics,
# and live simulation latency measurements
```
