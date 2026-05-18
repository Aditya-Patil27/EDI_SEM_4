# MotorSense — Roadmap

## Phase 1: 3D VUI Dashboard
**Slug**: `3d-vui-dashboard`  
**Goal**: Add Three.js radar3d, waterfall3d, gauge3d to existing Flask app  
**Requirements**: R1–R10  
**Success criteria**:
- 3 components render live data from either live ESP32 or CWRU replay
- State machine correct for all 4 states per component
- Flask endpoint `/api/vibration/current` returns RMS + FFT array
- No regression in existing Chart.js dashboard

## Phase 2: Anomaly Visualization & Alerts
**Goal**: Add anomaly-focused 3D views and notification system  
**Requirements**: R11–R14  
**Success criteria**:
- 3D feature scatter highlights anomalies in real-time
- Timeline component shows score history
- Desktop alert fires on threshold breach
- Audio alert on sustained (>5s) anomaly

## Phase 3: Edge Deployment & Firmware
**Goal**: Deploy feature extraction to ESP32, create OTA update pipeline  
**Requirements**: R15–R17  
**Success criteria**:
- ESP32 runs on-device feature extraction at 100 Hz
- Flask receives pre-extracted features instead of raw samples
- OTA update mechanism for ESP32 firmware images

## Phase 4: Dashboard Polish
**Goal**: Animation, accessibility, performance optimization  
**Requirements**: R18–R20  
**Success criteria**:
- 30fps on integrated GPU (Intel UHD 620)
- Colour-blind accessible palette passes WCAG 2.1 AA
- All transitions are smooth and not jarring
