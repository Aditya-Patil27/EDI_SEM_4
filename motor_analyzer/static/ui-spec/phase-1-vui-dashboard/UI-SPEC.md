# UI-SPEC: Phase 1 — 3D VUI Dashboard

## Overview
Add three Three.js visualisation components (radar3d, waterfall3d, gauge3d) to the existing MotorSense Flask dashboard. Components live in `static/3d/` as ES modules, served via Flask static. The existing Chart.js dashboard remains untouched — 3D components are added as a new section or toggled view.

## Style Foundation
| Token | Value | Usage |
|-------|-------|-------|
| `--bg-primary` | `#0A0E17` | Page background |
| `--glass-bg` | `#1A1F2ECC` | Panel backgrounds |
| `--glass-border` | `rgba(255,255,255,0.06)` | Panel borders |
| `--accent` | `#00D4AA` | Primary accent (teal) |
| `--accent-warm` | `#FF6B35` | Warning/anomaly accent |
| `--text-primary` | `#E8EDF5` | Headings / body |
| `--text-dim` | `#6B7280` | Secondary / muted |
| `--font` | `'Inter', sans-serif` | UI type |
| `--font-mono` | `'JetBrains Mono', monospace` | Data values |
| Radius: `12px` panels, `8px` cards | Shadow: `0 8px 32px rgba(0,0,0,0.4)` |

## Component Tree
```
dashboard-vui.html
├── Header (brand + status pills — reused from index.html)
├── 3D Controls (dat.GUI panel, toggled via gear icon)
│   ├── Component visibility toggles (radar / waterfall / gauge)
│   ├── Refresh rate slider (1–30 fps)
│   └── Color theme toggle (dark / high-contrast)
├── 3D Scene Container (full-width section below Chart.js area)
│   ├── radar3d canvas (left, 45vw × 400px)
│   ├── waterfall3d canvas (center, 45vw × 400px)
│   └── gauge3d canvas (right, 200×200px)
└── Status Overlay
    ├── Loading spinner (per component)
    ├── Empty state message per component
    └── Error toast
```

## Data Flow
```
SocketIO 'sensor_data' event
  │
  ▼
RingBuffer (256 slots, {rms, spectral[10], fft[]})
  ├──→ radar3d: radial plot (outer ring = spectral bands, inner = RMS)
  ├──→ waterfall3d: scrolling 3D surface (Z = frequency magnitude)
  └──→ gauge3d: needle angle = RMS / maxRMS
```

## State Machine (per component)
```
[loading] → [empty] → [live] ↔ [error]
                ↓
             [edge]   (low-data / degraded mode)
```
- **loading**: Component mounted, awaiting first data. Show spinner overlay.
- **empty**: Buffer empty, no data received. Show "Waiting for sensor data…".
- **live**: Normal operation, buffer ≥ 1 sample.
- **error**: WebSocket disconnected or NaN data. Show overlay with reconnect prompt.
- **edge**: Buffer < 128 samples (cold start). Show "Buffering…" indicator.

## Component Specs

### C1 — Scene Initialisation
- Three.js scene with dark glassmorphism background (`#0A0E17`)
- PerspectiveCamera (75° FOV, positioned at z=5)
- WebGLRenderer with antialiasing, transparent alpha
- OrbitControls disabled by default (optional toggle via dat.GUI)
- Responsive: `window.resize` listener updates camera aspect + renderer size

### C2 — Ring Buffer Data Manager
- Class `RingBuffer` with maxlen=256
- `.push({rms, spectral, fft, timestamp})` — overwrites oldest
- `.latest()` — returns most recent entry
- `.slice(n)` — returns last n entries
- `.length` — current fill count
- Subscribable: components register callbacks via `.onUpdate(fn)`
- Emits `'full'` event when first 128 samples reached (edge → live transition)

### C3 — radar3d Component
- Polar (radar) plot on XY plane
- **Outer ring**: 10 spectral band energies as radial bars, color-coded by magnitude
- **Inner shape**: RMS value as hexagon fill (scaled 0–1, interpolated to domain max)
- **Ticks**: 3 concentric rings at 0.25, 0.5, 0.75 of max value
- **Labels**: Band labels (0–9) at each spoke tip
- **Animation**: Bars pulse-smooth on update (lerp 0.85 per frame)
- Accepts: `{rms: number, spectral: number[10]}`

### C4 — waterfall3d Component
- 3D surface plot (X = time, Z = frequency bin, Y = magnitude)
- **Scroll**: New column added at z=0, oldest pushed to z=-max
- **Max depth**: 64 columns (configurable)
- **Frequency bins**: 32 bins (0–50 Hz), linearly spaced
- **Color mapping**: magnitude → gradient (dark blue → cyan → yellow → red)
- **Camera**: Fixed perspective looking down at 45° angle
- **Grid**: Subtle wireframe grid on floor plane
- Accepts: `{fft: Float32Array[32]}`

### C5 — gauge3d Component
- 3D needle gauge (semi-circular arc)
- **Arc**: 180° sweep (left = 0, bottom-center = 0.5, right = 1.0)
- **Needle**: Thin cylinder/lazor rotating around pivot
- **Ticks**: 5 major tick marks (0, 0.25, 0.5, 0.75, 1.0)
- **Value**: Floating RMS value displayed as text sprite above needle
- **Color zones**: Green (0–0.4), yellow (0.4–0.7), red (0.7–1.0)
- **Animation**: Needle follows with lerp (0.9 per frame)
- Accepts: `{rms: number}`

### C6 — SocketIO Integration
- Listen on existing `sensor_data` event (reuses same socket from main.js)
- Extract `fft`, `anomaly_score`, `dominant_freq` from payload
- Compute RMS from waveform or extract from feature (prefer: RMS = sqrt(mean(waveform²)))
- Push structured `{rms, spectral[10], fft[32]}` to RingBuffer
- Subscribe 3 components to RingBuffer updates

### C7 — State Management
- Each component instantiated with `{container: HTMLElement, ringBuffer: RingBuffer}`
- `.setState(state)` transitions: loading → empty → live → error | edge
- State change triggers overlay visibility + optional animation
- Error recovery: auto-retry WebSocket on disconnect (3s backoff)

### C8 — dat.GUI Controls
- Toggle checkbox: radar3d visible
- Toggle checkbox: waterfall3d visible
- Toggle checkbox: gauge3d visible
- Slider: refresh rate (1–30 fps, default 10)
- Slider: rotation speed (0–5, default 0 — no auto-rotate)
- Dropdown: color theme (dark | high-contrast)

### C9 — Responsive Layout
- 3D grids collapse to single-column below 1024px
- Canvas resize on window resize (debounced 100ms)
- Mobile: waterfall hidden, radar stacked above gauge

## Copywriting
| State | radar3d | waterfall3d | gauge3d |
|-------|---------|-------------|---------|
| Loading | "Radar initialising…" | "Waterfall initialising…" | "Gauge initialising…" |
| Empty | "Awaiting vibration data" | "Awaiting frequency data" | "Awaiting RMS value" |
| Edge (n<128) | "Buffering ({n}/128)" | "Buffering ({n}/128)" | — |
| Error | "Connection lost — retrying" | "Connection lost — retrying" | "Connection lost — retrying" |

## Checklist
| ID | Item | Status |
|----|------|--------|
| C1 | Scene initialisation with dark theme | DONE |
| C2 | RingBuffer data manager (256 slots) | DONE |
| C3 | radar3d polar plot | DONE |
| C4 | waterfall3d scrolling surface | DONE |
| C5 | gauge3d needle gauge | DONE |
| C6 | SocketIO integration for 3D data | DONE |
| C7 | State machine (loading/empty/live/error/edge) | DONE |
| C8 | dat.GUI control panel | DONE |
| C9 | Responsive layout (3→1 column) | DONE |
