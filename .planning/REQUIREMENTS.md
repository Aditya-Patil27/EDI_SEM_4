# MotorSense — Requirements (v1.0)

## Delivered Features

### 3D VUI Dashboard
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| R1 | Three.js scene with dark glassmorphism theme | P0 | DONE |
| R2 | Ring buffer data manager (256 samples) | P0 | DONE |
| R3 | radar3d polar plot (RMS + spectral bands) | P0 | DONE |
| R4 | waterfall3d scrolling frequency spectrum | P0 | DONE |
| R5 | gauge3d needle gauge for current RMS | P0 | DONE |
| R6 | SocketIO real-time data push | P0 | DONE |
| R7 | State machine per component (loading→empty→live→error) | P0 | DONE |
| R8 | dat.GUI panel (visibility, refresh rate, theme) | P1 | DONE |
| R9 | Responsive layout | P1 | DONE |
| R10 | `/api/vibration/current` REST endpoint | P1 | DONE |

### ML Pipeline
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| M1 | 28-dim feature extraction (time + frequency domain) | P0 | DONE |
| M2 | CompanyClassifier (RandomForest) for source ID | P0 | DONE |
| M3 | EnsembleAnomalyModel (IF + OC-SVM) | P0 | DONE |
| M4 | Per-company anomaly models | P1 | DONE |
| M5 | Autoencoder anomaly detector (MLPRegressor) | P1 | DONE |
| M6 | GMM alternative anomaly model | P1 | DONE |
| M7 | Transfer learning adapter (28→39 expansion) | P2 | DONE |

### Production Monitoring
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| P1 | Order tracking (RPM-normalized frequency analysis) | P1 | DONE |
| P2 | Degradation trending (RMS slope over 30 min) | P1 | DONE |
| P3 | 7-day RMS trend chart (localStorage persisted) | P1 | DONE |
| P4 | Motor auto-stop on sustained anomaly (30 windows) | P2 | DONE |
| P5 | `/api/trend` trend data endpoint | P1 | DONE |

### Data Engineering
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| D1 | CWRU dataset integration | P0 | DONE |
| D2 | JNU dataset integration | P1 | DONE |
| D3 | MAFAULDA dataset integration (7 fault folders) | P1 | DONE |
| D4 | NASA IMS run-to-failure integration | P2 | DONE |
| D5 | Synthetic data generation | P1 | DONE |
| D6 | Data augmentation (RPM, noise, amplitude) | P1 | DONE |
| D7 | Cross-dataset RMS normalization | P1 | DONE |
| D8 | Class-balanced resampling (5,000 per class) | P2 | DONE |
| D9 | MAFAULDA severity metadata tracking | P2 | DONE |
| D10 | CWRU replay bridge for offline testing | P1 | DONE |

## Future Requirements (Deferred)

### Edge Deployment & Firmware
| ID | Requirement | Priority |
|----|-------------|----------|
| R15 | ESP32 firmware image for OTA update | P1 |
| R16 | On-device feature extraction on ESP32 | P2 |
| R17 | Configurable alert thresholds in flash | P2 |

### Dashboard Polish
| ID | Requirement | Priority |
|----|-------------|----------|
| R18 | Animated transitions between states | P2 |
| R19 | Colour-blind accessible palette | P2 |
| R20 | 30fps on integrated GPU | P2 |

### Anomaly Viz & Alerts
| ID | Requirement | Priority |
|----|-------------|----------|
| R11 | 3D scatter plot of feature space with anomaly highlighting | P1 |
| R12 | Timeline component showing anomaly score history | P1 |
| R13 | Desktop/browser push notification on anomaly breach | P2 |
| R14 | Audio alert on sustained anomaly detection | P2 |

## Data Format Standard
All engineered data uses: `128-sample windows → 28-dim features → numpy arrays`
- `X_all.npy`: feature matrix (n_samples × 28)
- `y_all.npy`: string class labels
- `company_labels.npy`: integer company index
- `fault_labels.npy`: integer fault class index
- `metadata.json`: datasets, classes, severity info, augmentation config
- `X_{Company}.npy` / `y_{Company}.npy`: per-company splits
