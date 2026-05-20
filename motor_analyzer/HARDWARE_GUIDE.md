# MotorSense — Hardware Integration Guide

> For the friend building the ESP32 + ADXL345 hardware rig.

---

## 1. Wiring Diagram

| ESP32 Pin | Connects To |
|-----------|-------------|
| GPIO 21 (SDA) | ADXL345 SDA |
| GPIO 22 (SCL) | ADXL345 SCL |
| 3.3V | ADXL345 VCC |
| GND | ADXL345 GND |
| GPIO 12 | Motor 1 IN1 |
| GPIO 13 | Motor 1 IN2 |
| GPIO 14 | Motor 1 ENA (PWM) |
| GPIO 25 | Motor 2 IN3 |
| GPIO 26 | Motor 2 IN4 |
| GPIO 27 | Motor 2 ENB (PWM) |

- ADXL345 in I2C mode: CS pin → 3.3V, SDO → GND (address 0x53)
- Motor driver: L298N or equivalent, external 12V supply for motors
- PWM frequency: 30 kHz on all motor pins (above audible range)

---

## 2. What the ESP32 Does

The firmware is in `esp32_motor_accel/esp32_motor_accel.ino`.

**At boot:**
1. Initializes ADXL345 at ±16g, 800 Hz internal rate
2. Initializes motors (stopped)
3. Waits for serial commands from the Python backend

**During operation (loop at 100 Hz):**
1. Read Z-axis acceleration from ADXL345
2. Print as float → `Serial.println(value, 4)` → one line every 10 ms
3. Listen for serial commands from the Python side

**Sample output stream (115200 baud):**
```
-0.2345
-0.1987
-0.3124
# RPM: 1800
# STATUS: MOTOR1 ON
-0.2876
  ...
```

Lines starting with `#` are debug/status — the backend logs them but doesn't process as data.

---

## 3. Protocol — What the Backend Expects

### Serial (115200 baud, 8N1)
```
<float_value>\n    ← Z-axis acceleration in m/s², one per 10ms (100 Hz)
# <message>\n      ← status/debug lines (prefixed with #)
```

### REST API (as alternative to serial)
If you want to test without the ESP32 connected, POST samples directly:

```bash
curl -X POST http://127.0.0.1:5050/api/inject \
  -H "Content-Type: application/json" \
  -d '{"samples": [0.123, -0.456, 0.789, ...]}'
```

This is the same endpoint the CWRU replay bridge uses.

---

## 4. Starting the System

### Step 1: Flash the ESP32
```bash
# In Arduino IDE or PlatformIO:
# - Board: ESP32-WROOM-DA Module
# - Upload Speed: 921600
# - Port: (your COM port)
```

### Step 2: Start the Python backend
```bash
cd motor_analyzer
python app.py
# → Starts on http://127.0.0.1:5050
```

### Step 3: Connect the ESP32
Open http://127.0.0.1:5050 in a browser.
Click "Connect" → select the COM port → click "Connect Serial".

OR via API:
```bash
curl -X POST http://127.0.0.1:5050/api/connect \
  -H "Content-Type: application/json" \
  -d '{"port": "COM3", "baud": 115200}'
```

### Step 4: Start the motor
Click "Run" on the dashboard, or:
```bash
curl -X POST http://127.0.0.1:5050/api/motor/control \
  -H "Content-Type: application/json" \
  -d '{"motor": 1, "speed": 180}'
```

### Step 5: Train the anomaly detector
Once the motor is running and stable (~10 seconds of baseline data):
- Click "Train" on the dashboard — the system will:
  1. Identify the "company fingerprint" (CWRU, JNU, MAFAULDA, etc.)
  2. Train an IsolationForest on Normal data
  3. Start real-time anomaly detection

---

## 5. Testing Without Hardware (CWRU Replay Bridge)

If the hardware isn't ready yet, the CWRU replay bridge injects real bearing fault data:

```bash
cd motor_analyzer
python app.py          # Terminal 1 — start Flask
python cwru_replay.py --sequence --mode http  # Terminal 2 — inject CWRU data
```

This replays a sequence of healthy → inner race fault → ball fault → outer race fault at 1-minute intervals each.

---

## 6. Dashboard URLs

| URL | What It Shows |
|-----|---------------|
| `http://127.0.0.1:5050/` | Main Chart.js dashboard — waveform, FFT, RMS gauge, 7-day trend |
| `http://127.0.0.1:5050/vui` | 3D Three.js dashboard — radar3d, waterfall3d, gauge3d |

Both show the same data, different visualizations.

---

## 7. What the `#` Status Lines Should Include

The ESP32 can send diagnostic info via `#`-prefixed lines. Useful things to include:

```
# RPM: <value>               ← current RPM (if you have a tachometer)
# MOTOR1: ON  speed=180      ← motor state changes
# MOTOR2: OFF
# TEMP: <value>C             ← if you add a temperature sensor
# WARN: low voltage          ← warnings
```

These appear in the browser's "ESP32 Log" panel and in the backend logs.

---

## 8. Important Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate (to Python) | **100 Hz** | One reading every 10 ms — **must be precise** |
| ADXL345 internal rate | 800 Hz | Configurable, 800 Hz gives good anti-aliasing |
| Sensor range | ±16 g | Highest range to avoid clipping |
| Serial baud | 115200 | Matches `config.yaml` |
| Window size | 128 samples | ~1.28 seconds of data per ML inference |
| Feature vector | 28 dimensions | Time + frequency domain features |
| Anomaly detection | Per window | Every 128 samples (~1.28s) |

### ⚠️ Critical: 100 Hz Must Be Stable
The anomaly detection and frequency analysis assume **exactly** 100 Hz sample rate. If the ESP32 drifts:
- Frequency peaks shift (a 50 Hz vibration might show as 48 Hz)
- The degradation trend (RMS slope per hour) becomes inaccurate
- The auto-stop logic (30 consecutive anomaly windows) gets wrong timing

Use `delay(10)` or better, a hardware timer. Don't use `delay()` with other blocking operations.

---

## 9. Real-Time Data Flow Diagram

```
ESP32 @ 100 Hz                  Flask @ 1.28 Hz window           Browser
─────────────────               ──────────────────────           ────────
Serial: float\n ───────────────→ raw_buffer[256]
                                ┌─ baseline calibration ──┐
                                │ (first ~80 samples =    │
                                │  DC offset removal)     │
                                └─────────────────────────┘
                                ↓
                                extract_features(chunk, baseline)
                                → 28-dim vector
                                ↓
                                CompanyClassifier → which machine?
                                ↓
                                EnsembleAnomalyModel → anomaly score
                                ↓
                                socketio.emit('sensor_data') ──────→ Chart.js + Three.js
                                ↓
                                /api/vibration/current (REST) ─────→ polling clients
                                /api/trend (REST) ─────────────────→ 7-day chart
```

---

## 10. Serial Commands from Backend to ESP32

The Flask backend sends these via serial:

| Sent | Meaning |
|------|---------|
| `"1\n"` | Start Motor 1 at cruise speed |
| `"2\n"` | Start Motor 2 (with anti-stiction ramp) |
| `"0\n"` | Stop all motors |
| `"SPEED:180\n"` | Set PWM speed (0–255) |
| `"?\n"` | Request status dump |

These are triggered by `/api/motor/control` POST requests from the dashboard.

---

## 11. API Reference (Quick)

### Status & Data
```bash
GET  /api/vibration/current    # Latest RMS, freq, anomaly score
GET  /api/trend                # RMS history (timestamps + values)
GET  /api/status               # Full system state
```

### Connection
```bash
POST /api/connect              # {"port": "COM3", "baud": 115200}
POST /api/disconnect           # Disconnect serial
```

### Motor Control
```bash
POST /api/motor/control        # {"motor": 1|2|0, "speed": 0-255}
```

### Data Injection (for testing)
```bash
POST /api/inject               # {"samples": [float, ...]}
```

### Model Management
```bash
GET  /api/model/list           # List available models
POST /api/model/load           # {"company": "CWRU"}
POST /api/model/train          # Train on current data
```

---

## 12. What the ADXL345 at 100Hz Can Actually Detect

| Fault Type | Detectable? | Notes |
|------------|-------------|-------|
| **Unbalance** | ✅ Yes | Clear 1× RPM peak in spectrum |
| **Misalignment** | ✅ Yes | 2× RPM + harmonics |
| **Mechanical Looseness** | ✅ Yes | Broadband noise floor rise |
| **Bent Shaft** | ⚠️ Partial | Higher harmonics, subtle |
| **Bearing Defects (early)** | ❌ No | Frequencies > 50 Hz alias at 100 Hz |
| **Bearing Defects (late/gross)** | ✅ Yes | Broadband energy increase, RMS rise |

**Bottom line:** You won't catch early bearing pitting (need >2 kHz for that), but you will catch the 70% of motor failures caused by unbalance, misalignment, and looseness — plus gross bearing damage once it's bad enough to raise the overall vibration floor.

---

## 13. File Locations on Disk

```
motor_analyzer/
├── esp32_motor_accel/
│   └── esp32_motor_accel.ino     ← ESP32 firmware (the only file you flash)
├── app.py                         ← Flask backend
├── config.yaml                    ← All configuration
├── cwru_replay.py                 ← Replay bridge for testing
├── feature_pipeline.py            ← 28-dim feature extraction
├── engineer_dataset.py            ← Multi-dataset training pipeline
├── train_pipeline.py              ← Model training script
├── data/raw/
│   ├── cwru/        (CWRU bearing faults)
│   ├── jnu/         (JNU bearing faults)
│   ├── mafaulda/    (MAFAULDA motor faults — unbalance, misalignment, etc.)
│   └── nasa_ims/    (NASA run-to-failure bearing degradation)
├── models/
│   ├── company_classifier.pth     ← Identifies which machine
│   └── companies/
│       ├── CWRU.pkl               ← Per-machine anomaly models
│       ├── JNU.pkl
│       ├── MAFAULDA.pkl
│       ├── NASA_IMS.pkl
│       └── Synthetic.pkl
└── static/
    ├── js/                        ← Chart.js dashboard
    └── 3d/                        ← Three.js VUI (6 ES modules)
```
