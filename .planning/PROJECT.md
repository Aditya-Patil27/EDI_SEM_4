# MotorSense — Predictive Maintenance via 3D Vibration UI

## Core Value
ESP32 + ADXL345 accelerometer streams real-time Z-axis vibration to a Flask server that extracts 28-dim features and runs ML anomaly detection (IsolationForest + OneClassSVM ensemble). The existing Chart.js dashboard shows waveform/FFT. **Phase 1** adds a 3D Vibration User Interface (VUI) with Three.js for immersive monitoring: radar3d (multi-axis overview), waterfall3d (frequency drift), and gauge3d (instant RMS).

## Architecture
- **Sensor**: ESP32-WROOM-32 + ADXL345 (±16g, I2C) @ 100 Hz
- **Backend**: Flask + SocketIO + eventlet, 28-dim feature pipeline
- **ML**: RandomForest CompanyClassifier, EnsembleAnomalyModel (IF + OC-SVM)
- **Frontend**: Chart.js (existing) + Three.js (Phase 1 addition), dark glassmorphism theme

## Constraints
- No Vue/React — Three.js modules via ES imports served statically
- Additive to existing app — must not break Chart.js dashboard
- State machine per 3D component: loading → empty → live → (error) | (edge)
- Ring buffer of 256 samples feeds radar3d/waterfall3d; gauge3d shows last RMS value

## Key Decisions
- GSD planning framework for tracking
- UI-SPEC.md as design contract before any code
- Phase 1 = 3D viz only; ML anomaly viz deferred to Phase 2
- Dark glassmorphism: bg #0A0E17, glass #1A1F2ECC, accent #00D4AA
