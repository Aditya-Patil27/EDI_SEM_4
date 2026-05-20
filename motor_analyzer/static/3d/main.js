import { RingBuffer } from './ring-buffer.js'
import { Radar3D } from './radar3d.js'
import { Waterfall3D } from './waterfall3d.js'
import { Gauge3D } from './gauge3d.js'

// ── Shared data buffer ──
const ringBuffer = new RingBuffer(256)

// ── Components ──
const radarEl = document.getElementById('radar3d-container')
const waterfallEl = document.getElementById('waterfall3d-container')
const gaugeEl = document.getElementById('gauge3d-container')

let radar, waterfall, gauge

function initComponents() {
  if (radarEl) radar = new Radar3D(radarEl, ringBuffer)
  if (waterfallEl) waterfall = new Waterfall3D(waterfallEl, ringBuffer)
  if (gaugeEl) gauge = new Gauge3D(gaugeEl, ringBuffer)
}

// ── SocketIO integration ──
function connectSocket() {
  const socket = io({ transports: ['websocket', 'polling'] })

  socket.on('connect', () => {
    console.log('[3D] SocketIO connected')
    ringBuffer.emitError(null)
  })

  socket.on('sensor_data', (data) => {
    const rms = computeRMS(data.waveform)
    const fft = data.fft ? data.fft.map(v => Math.abs(v || 0)) : new Array(32).fill(0)
    const spectral = binSpectral(fft, 10)

    ringBuffer.push({ rms, spectral, fft, timestamp: Date.now() })
  })

  socket.on('disconnect', () => {
    console.warn('[3D] SocketIO disconnected')
    ringBuffer.emitError('disconnected')
  })
}

function computeRMS(waveform) {
  if (!waveform || waveform.length === 0) return 0
  let sumSq = 0
  for (const v of waveform) sumSq += v * v
  return Math.sqrt(sumSq / waveform.length) * 3
}

function binSpectral(fft, numBins) {
  if (!fft || fft.length === 0) return new Array(numBins).fill(0)
  const binSize = Math.floor(fft.length / numBins)
  const result = []
  for (let b = 0; b < numBins; b++) {
    let sum = 0
    const start = b * binSize
    const end = b === numBins - 1 ? fft.length : start + binSize
    for (let i = start; i < end; i++) sum += fft[i] || 0
    result.push(sum / (end - start))
  }
  return result
}

// ── dat.GUI controls ──
function setupGUI() {
  if (typeof dat === 'undefined') return

  const gui = new dat.GUI({ name: '3D Controls' })
  const settings = {
    'Radar': true,
    'Waterfall': true,
    'Gauge': true,
    'Refresh Rate': 10,
    'Auto Rotate': false,
    'Theme': 'dark',
  }

  gui.add(settings, 'Radar').onChange(v => radar?.setVisible(v))
  gui.add(settings, 'Waterfall').onChange(v => waterfall?.setVisible(v))
  gui.add(settings, 'Gauge').onChange(v => gauge?.setVisible(v))
  gui.add(settings, 'Refresh Rate', 1, 30, 1)
  gui.add(settings, 'Auto Rotate')
  gui.add(settings, 'Theme', ['dark', 'high-contrast'])

  const toggle = document.getElementById('gui-toggle')
  if (toggle) {
    toggle.addEventListener('click', () => {
      const el = gui.domElement
      el.style.display = el.style.display === 'none' ? '' : 'none'
    })
  }
}

// ── Boot ──
document.addEventListener('DOMContentLoaded', () => {
  initComponents()
  connectSocket()
  setupGUI()
})
