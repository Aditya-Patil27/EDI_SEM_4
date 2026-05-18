# ESP32 Data Recording & Retraining Guide

## Overview
Record real vibration data from your ESP32+ADXL345 to replace Synthetic training data and build custom company models.

## Step 1: Record Real Data

### Required Hardware
- ESP32 with ADXL345 accelerometer
- USB cable connected to your machine
- Machine/motor to record from

### Recording Script
Use the existing Flask app's baseline capture feature:

1. Start the app: `python app.py`
2. Open `http://127.0.0.1:5050`
3. Connect the serial port
4. For each machine you want to add:
   - Run the machine in a **healthy (normal)** state
   - Wait for baseline to lock (~80 samples, ~1 second)
   - Click "Train Baseline" and let it run for 20+ seconds
   - The collected features will be stored in `state['training_features']`

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

## Step 3: Add Data Ingestion Adapter

Edit `data_ingestion.py` to add a new `CustomAdapter`:

```python
class CustomAdapter(BaseAdapter):
    """Loads CSV data from data/custom/<company>/<label>_*.csv"""
    def __init__(self, data_dir="data/custom"):
        super().__init__()
        self.data_dir = Path(data_dir)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Walk directory, parse CSV files
        # Assign integer company label per parent folder
        # Return X, y, company_labels
        pass
```

Or use the built-in interactive mode:
```python
from data_ingestion import process_company_csv
X, y = process_company_csv("MachineA", "path/to/file.csv")
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
| Normal baseline per company | 128 windows (~164 sec) | ~3 minutes |
| Per-fault type (optional) | 50 windows (~64 sec) | ~1 minute |

## Notes
- Keep the sensor placement consistent between recordings
- Record at the same sample rate (ESP32 default ~100 Hz)
- The feature pipeline resamples to 100 Hz automatically if needed
- Cross-dataset RMS normalization handles different amplitude scales
