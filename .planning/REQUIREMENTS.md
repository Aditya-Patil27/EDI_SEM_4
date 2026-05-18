# MotorSense — V1 Requirements

## 3D VUI Dashboard (Phase 1)
| ID | Requirement | Priority | Traceability |
|----|-------------|----------|--------------|
| R1 | Three.js scene initializes on dashboard page with dark glassmorphism theme | P0 | UI-SPEC C1 |
| R2 | Ring buffer data manager holds 256 samples, feeds all 3D components | P0 | UI-SPEC C2 |
| R3 | radar3d component renders polar plot of RMS + spectral bands | P0 | UI-SPEC C3 |
| R4 | waterfall3d component renders scrolling frequency spectrum | P0 | UI-SPEC C4 |
| R5 | gauge3d component renders needle gauge for current RMS | P0 | UI-SPEC C5 |
| R6 | 3D components receive real-time data via SocketIO `sensor_data` event | P0 | UI-SPEC C6 |
| R7 | Each component has states: loading → empty → live → (error \| edge) | P0 | UI-SPEC C7 |
| R8 | dat.GUI panel controls toggle visibility, refresh rate, color theme | P1 | UI-SPEC C8 |
| R9 | responsive layout — 3D canvases resize with window | P1 | UI-SPEC C9 |
| R10 | `/api/vibration/current` REST endpoint returns latest RMS + FFT for 3D | P1 | — |

## Anomaly Viz & Alerts (Phase 2, deferred)
| ID | Requirement | Priority |
|----|-------------|----------|
| R11 | 3D scatter plot of feature space with anomaly highlighting | P1 |
| R12 | Timeline component showing anomaly score history | P1 |
| R13 | Desktop/browser push notification on anomaly threshold breach | P2 |
| R14 | Audio alert on sustained anomaly detection | P2 |

## Edge Deployment & Firmware (Phase 3, deferred)
| ID | Requirement | Priority |
|----|-------------|----------|
| R15 | ESP32 firmware image for OTA update | P1 |
| R16 | on-device feature extraction on ESP32 | P2 |
| R17 | configurable alert thresholds persisted in flash | P2 |

## Dashboard Polish (Phase 4, deferred)
| ID | Requirement | Priority |
|----|-------------|----------|
| R18 | Animated transitions between states | P2 |
| R19 | Colour-blind accessible palette option | P2 |
| R20 | Performance profiling — target 30fps on integrated GPU | P2 |

## Out of Scope (V1)
- Mobile app or PWA
- Multi-user auth
- Historical data persistence / database
