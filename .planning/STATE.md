# MotorSense — Project State

## Current Status: Production-Ready (v1.0)

### Data Pipeline
| Source | Format | Samples (post-resample) | Fault Classes |
|--------|--------|:-----------------------:|---------------|
| CWRU (Case Western) | .mat @ 12kHz | 12,416 | BallFault, InnerRace, Normal, OuterRace |
| JNU (Jiangnan) | .csv @ 12kHz | 1,105 | BallFault, InnerRace, Normal, OuterRace |
| MAFAULDA | .csv @ 12kHz, 8-col | 18,623 | BallFault, CageFault, Misalignment, Normal, OuterRace, Unbalanced |
| NASA IMS | tab-sep text @ 20kHz, 4-col | 623 | Normal, OuterRace |
| Synthetic | Generated | 2,233 | InnerRace, Misalignment, Normal, Unbalanced |
| **Total** | | **35,000** (balanced) | **7 classes** |

### ML Models (trained & saved in models/)
- **CompanyClassifier** (RandomForest): 4 companies, 100% F1
- **Per-company anomaly models** (IsolationForest): CWRU, JNU, MAFAULDA, Synthetic
- **EnsembleAnomalyModel** (IF + OC-SVM): general anomaly detector
- **GMM**: alternative anomaly detector
- **AutoencoderAnomalyDetector** (28→16→8→16→28 MLPRegressor): neural anomaly scoring

### Frontend
- `/` — Chart.js dashboard (waveform, FFT, RMS gauge, 7-day trend)
- `/vui` — Three.js 3D VUI (radar3d, waterfall3d, gauge3d)
- Real-time data via SocketIO
- dat.GUI controls panel
- Dark glassmorphism theme (#0A0E17 / #00D4AA)

### Production Features
| Feature | Status |
|---------|--------|
| 3D VUI dashboard | DONE |
| Order tracking (RPM-normalized FFT) | DONE |
| Degradation trending (RMS slope via `/api/trend`) | DONE |
| 7-day trend chart (Chart.js + localStorage) | DONE |
| Motor auto-stop (30 consecutive anomaly windows) | DONE |
| CWRU replay bridge (offline testing) | DONE |
| Data augmentation (RPM ±15%, noise 0–15%, amp 0.7–1.3×) | DONE |
| Cross-dataset RMS normalization | DONE |
| Class-balanced resampling (5,000 per class) | DONE |
| MAFAULDA severity metadata tracking | DONE |
| NASA IMS run-to-failure integration | DONE |
| Transfer learning adapter (28→39 expansion) | DONE |

### Tests
- 18 unit tests (feature_pipeline, ml_models) — ALL PASS
- Integration tests needed for end-to-end pipeline

### Infrastructure
- Branch: `ap` on `github.com/Aditya-Patil27/EDI_SEM_4`
- No hardware required — CWRU replay bridge + synthetic data for testing
- ADXL345 @ 100Hz constraint: catches unbalance/misalignment/looseness but not early bearing defects
- `.gitignore` excludes `data/processed/`, `data/engineered/`

### Blockers
None

### Known Issues
- MAFAULDA dominant in total samples (53% before resampling)
- CageFault only from MAFAULDA → no cross-company validation for that class
- NASA IMS uses only first 10% / last 10% of files; middle degradation data unused
