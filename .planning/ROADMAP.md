# MotorSense — Roadmap

## ✅ Phase 1: 3D VUI Dashboard (COMPLETE)
**Goal**: Three.js radar3d, waterfall3d, gauge3d integrated into Flask app  
**Delivered**: 6 ES modules in `static/3d/`, `/vui` route, SocketIO integration, state machine, dat.GUI controls

## ✅ Phase 2: Production Enhancements (COMPLETE)
**Goal**: Order tracking, autoencoder anomaly model, degradation trending, auto-stop, 7-day trend  
**Delivered**: All features live and tested

## ✅ Phase 3: Multi-Dataset Engineering Pipeline (COMPLETE)
**Goal**: Unified training data from CWRU, JNU, MAFAULDA, NASA IMS, Synthetic  
**Delivered**: Augmentation, RMS normalization, class balancing, severity tracking

## 🔲 Phase 4: Edge Deployment & Firmware
**Goal**: Deploy feature extraction to ESP32, OTA update pipeline  
**Requirements**: R15–R17  
**Success criteria**:
- ESP32 on-device feature extraction at 100 Hz
- Flask receives pre-extracted features
- OTA update mechanism

## 🔲 Phase 5: Dashboard Polish
**Goal**: Animation, accessibility, performance optimization  
**Requirements**: R18–R20  
**Success criteria**:
- 30fps on integrated GPU
- WCAG 2.1 AA colour-blind palette
- Smooth transitions

## 🔲 Phase 6: Anomaly Visualization
**Goal**: 3D feature scatter, anomaly score timeline, alerts  
**Requirements**: R11–R14  
**Success criteria**:
- Real-time 3D scatter highlights anomalies
- Timeline component shows anomaly score history
- Desktop + audio alerts on threshold breach

## Future Ideas
- Degradation-based regression model (predict RUL)
- Live ESP32 data collection + model retraining loop
- Mobile-friendly responsive dashboard
- Historical fault logging to CSV/DB
