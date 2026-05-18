# API Reference

Base URL: `http://127.0.0.1:5050`

## Serial Connection

### `GET /api/ports`
List available serial ports.

**Response**:
```json
[
  {"device": "COM3", "desc": "USB Serial Port"},
  {"device": "/dev/cu.usbserial-0001", "desc": "CP2102 USB to UART"}
]
```

### `POST /api/connect`
Connect to a serial port.

**Request**:
```json
{"port": "COM3", "baud": 115200}
```
**Response**: `{"ok": true, "port": "COM3"}`  
**Errors**: `400` with `{"ok": false, "error": "..."}`

### `POST /api/disconnect`
Disconnect serial port.

**Response**: `{"ok": true}`

## System Status

### `GET /api/status`
Full system state snapshot.

**Response**:
```json
{
  "connected": false,
  "baseline_ready": false,
  "training": false,
  "evaluating": false,
  "active_model": null,
  "status": "IDLE — Connect a serial port to begin",
  "status_level": "info",
  "motor": 0,
  "motor_speed": 200,
  "company_idx": 0,
  "company_name": "Unknown",
  "company_confidence": 0.0,
  "company_identified": false,
  "feature_dim": 28,
  "available_companies": ["Unknown", "CWRU", "JNU", "Synthetic"]
}
```

## Training

### `POST /api/train/start`
Start vibration capture for model training.

**Request**:
```json
{"name": "motor_1_1000rpm", "duration": 20}
```
**Response**: `{"ok": true, "name": "motor_1_1000rpm", "duration": 20}`  
**Errors**: `400` if baseline not calibrated

### `POST /api/train/cancel`
Cancel active training session.

**Response**: `{"ok": true}`

## Model Management

### `POST /api/model/save`
Save current trained model to disk.

**Request**: `{"name": "motor_1_1000rpm"}`  
**Response**: `{"ok": true, "path": "models/motor_1_1000rpm.pkl", "name": "motor_1_1000rpm"}`  
**Errors**: `400` if no model in memory

### `POST /api/model/load`
Load a saved model from disk and activate evaluation.

**Request**: `{"name": "motor_1_1000rpm"}`  
**Response**: `{"ok": true, "name": "...", "meta": {"timestamp": "...", "samples": 0, "baseline": 0.0}}`  
**Errors**: `404` if model not found

### `GET /api/model/list`
List all saved models with metadata.

**Response**:
```json
[
  {"name": "motor_1_1000rpm", "filename": "motor_1_1000rpm", "timestamp": "2026-05-18 12:00:00", "samples": 0, "baseline": 0.0}
]
```

### `POST /api/model/delete`
Delete a saved model.

**Request**: `{"name": "motor_1_1000rpm"}`  
**Response**: `{"ok": true}` or `404`

## Evaluation

### `POST /api/evaluate/stop`
Stop active evaluation/inference.

**Response**: `{"ok": true}`

### `GET /api/evaluate/report`
Get latest evaluation model card.

**Response**: Evaluation report JSON (from `evaluation_report.json`)  
**Errors**: `404` if no report exists

### `POST /api/evaluate/run`
Trigger full evaluation pipeline as subprocess.

**Response**: `{"ok": true, "verdict": "PASS", "report": {...}, "log": "..."}`

### `POST /api/evaluate/synthetic`
Generate synthetic evaluation datasets.

**Response**: `{"ok": true, "path": "data/synthetic"}`

### `POST /api/evaluate/run-tests`
Run pytest suite server-side.

**Response**: `{"ok": true, "passed": 18, "failed": 0, "summary": ["..."]}`

## Company Awareness

### `GET /api/company/status`
Current company identification status.

**Response**:
```json
{
  "current_company": "CWRU",
  "identified": true,
  "confidence": 0.997,
  "available_companies": ["CWRU", "JNU", "Synthetic"],
  "registered_companies": ["CWRU", "JNU", "Synthetic"]
}
```

### `GET /api/company/models`
List per-company anomaly models saved on disk.

**Response**:
```json
[
  {"company": "CWRU", "modified": "2026-05-18 12:00:00"},
  {"company": "JNU", "modified": "2026-05-18 12:00:00"}
]
```

### `GET /api/company/classifier/status`
CompanyClassifier training status.

**Response**:
```json
{
  "trained": true,
  "num_companies": 4,
  "company_names": ["Unknown", "CWRU", "JNU", "Synthetic"],
  "feature_dim": 28
}
```

### `POST /api/company/reidentify`
Force re-identification of the connected machine (clears fingerprint buffer).

**Response**: `{"ok": true}`

## Motor Control

### `POST /api/motor/control`
Send motor command to ESP32 over serial.

**Request**: `{"motor": 1, "speed": 200}`  
`motor`: 0=stop, 1=motor1, 2=motor2 | `speed`: PWM 0–255

**Response**: `{"ok": true, "motor": 1, "speed": 200}`  
**Errors**: `400` if not connected

## Pretrained Weights

### `GET /api/pretrained/info`
Transfer adapter and pretrained weights status.

**Response**:
```json
{
  "available": false,
  "files": [],
  "source": "ml_repo",
  "input_dim": 39,
  "num_classes": 3,
  "classes": ["Normal", "Unbalanced", "Bearing Fault"],
  "adapter": {"enabled": true, "adapt_dim": 28, "fallback_to_sklearn": true},
  "transfer_model": {"trained": true, "classes": ["CWRU", "JNU", "Synthetic"]},
  "note": "Pretrained weights (39-dim audio) -> TransferLearningAdapter (28-dim vibration) via Linear(28,39) expansion layer."
}
```

### `GET /api/pretrained/weights`
Check `.pth` weight file metadata.

**Response**:
```json
{
  "count": 0,
  "files": [],
  "can_use_directly": false,
  "reason": "Pretrained weights are from audio MFCC pipeline (39-dim). Current app uses 28-dim vibration features. Retraining needed."
}
```

## Development

### `POST /api/inject`
Direct buffer injection for CWRU replay bridge. Bypasses serial.

**Request**: `{"samples": [0.12, -0.05, 0.33, ...]}`  
**Response**: `{"ok": true, "injected": 100}`

## WebSocket Events (SocketIO)

### Server → Client

| Event | Payload | Trigger |
|-------|---------|---------|
| `sensor_data` | `{waveform, fft, dominant_freq, is_anomaly, anomaly_score, evaluating, training, company, company_identified}` | Every feature window (every ~1.28s) |
| `company_identified` | `{company, confidence, idx}` | After fingerprint_samples collected |
| `status_update` | `{status, level, baseline?}` | Any state change |
| `train_progress` | `{elapsed, total, progress, samples}` | Every feature window during training |
| `train_complete` | `{model_name, company}` | Training window elapsed |
| `esp32_log` | `{msg}` | ESP32 `#` prefix lines |
| `motor_state` | `{motor, speed, label}` | Motor control command sent |

### Client → Server

| Event | Payload | Purpose |
|-------|---------|---------|
| `connect` | — | Initial connection, server replies with `status_update` |
