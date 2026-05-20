# ESP32 Data Recording & Retraining Guide

## Overview
Record real vibration data from your ESP32+ADXL345 to replace Synthetic training data and build custom company models.

## Step 1: Record Real Data

### Required Hardware
- ESP32 with ADXL345 accelerometer
- USB cable connected to your machine
- Machine/motor to record from

### Recording Script
Use the existing Flask app's capture feature:

1. Start the app: `python app.py`
2. Open `http://127.0.0.1:5050`
3. Connect the serial port
4. For each machine you want to add:
   - Run the machine in a **healthy (normal)** state
   - Wait for baseline to calibrate automatically (~80 samples, ~1 second)
   - Click "Start Training" and let it run for 20+ seconds
   - The collected features will be used by `_finish_training_internal()` to train and save a per-company anomaly model

### Manual Data Capture (for larger datasets)
Alternatively, save raw serial data to CSV:

```python
# capture_data.py
import serial
import csv
import time

ser = serial.Serial('COM3', 115200, timeout=1)
with open('machine_a_normal.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    start = time.time()
    while time.time() - start < 60:  # record 60 seconds
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line and not line.startswith('#'):
            try:
                writer.writerow([float(line)])
            except ValueError:
                pass
```

## Step 2: Label Your Data

Organize recorded CSV files into a directory structure:

```
data/custom/
├── MachineA/
│   ├── normal_1.csv
│   ├── normal_2.csv
│   └── fault_bearing.csv
├── MachineB/
│   ├── normal_1.csv
│   └── normal_2.csv
└── ...
```

**Prerequisite**: The existing CWRU and JNU datasets should already be downloaded under `data/raw/cwru/` and `data/raw/jnu/` before retraining. The ingestion script expects these directories. See the CWRU data center (https://engineering.case.edu/bearingdatacenter/download-data-file) and JNU repository for download links.

## Step 3: Add Data Ingestion Adapter

Edit `data_ingestion.py` to add a new processing function following the existing pattern:

```python
def _process_custom():
    """Loads CSV data from data/custom/<company>/<label>_*.csv"""
    import csv
    custom_dir = os.path.join(RAW_DIR, 'custom')
    if not os.path.exists(custom_dir):
        log.warning("Custom directory not found, skipping")
        return []

    records = []
    for company_dir in sorted(os.listdir(custom_dir)):
        company_path = os.path.join(custom_dir, company_dir)
        if not os.path.isdir(company_path):
            continue
        for fname in sorted(os.listdir(company_path)):
            if not fname.endswith('.csv'):
                continue
            fpath = os.path.join(company_path, fname)
            try:
                sig = np.loadtxt(fpath, dtype=np.float64)
            except Exception as e:
                log.error(f"  Failed to load {fname}: {e}")
                continue
            if len(sig) < 128:
                continue
            windows = _segment_into_windows(sig)
            for win in windows:
                feats = _features_from_segment(win)
                records.append({
                    'filepath': fname,
                    'raw_class': fname.split('_')[0],  # e.g. "normal_1.csv" -> "normal"
                    'class': fname.split('_')[0],
                    'dataset': company_dir,
                    'format': 'csv',
                    'features': feats,
                })
    log.info(f"  Custom: {len(records)} windows processed")
    return records
```

Then register it in `ingest_all_datasets()`:
```python
all_records.extend(_process_custom())
```

## Step 4: Update Config

Edit `config.yaml`:
```yaml
companies:
  classes:
    - "Unknown"
    - "CWRU"
    - "JNU"
    - "MachineA"        # your real machine
    - "MachineB"
  fingerprint_samples: 512
  use_synthetic: false   # disable Synthetic once real data is collected
```

## Step 5: Retrain Models

```bash
python data_ingestion.py    # re-ingest all data including custom
python train_pipeline.py    # retrain company classifier + anomaly models
```

## Step 6: Verify

```bash
python evaluate_pipeline.py                           # check accuracy
python -m pytest tests/ -v                            # run unit tests
python app.py                                         # start app, verify UI shows your companies
```

## Minimum Data Requirements

| Component | Min Samples | Recording Time (@ 100 Hz) |
|-----------|-------------|---------------------------|
| Normal baseline per company | 128 windows (~82 sec) | ~2 minutes |
| Per-fault type (optional) | 50 windows (~32 sec) | ~1 minute |

*Note: `_segment_into_windows()` uses step=64 (50% overlap), so each window advances 0.64s at 100 Hz.*

## Notes
- Keep the sensor placement consistent between recordings
- Record at the same sample rate (ESP32 default ~100 Hz)
- The feature pipeline resamples to 100 Hz automatically if needed
- Cross-dataset RMS normalization handles different amplitude scales
