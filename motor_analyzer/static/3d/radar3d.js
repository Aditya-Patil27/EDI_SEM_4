import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js'
import { createScene, createTextSprite, COLORS } from './scene.js'
import { RingBuffer } from './ring-buffer.js'

const NUM_BANDS = 10
const MAX_RADIUS = 1.8
const BAR_HEIGHT_MAX = 0.6
const LERP_FACTOR = 0.85

export class Radar3D {
  constructor(container, ringBuffer) {
    this.container = container
    this.buffer = ringBuffer
    this.state = 'loading'
    this._group = new THREE.Group()

    const { scene, camera, renderer } = createScene(container)
    this.scene = scene
    this.camera = camera
    this.renderer = renderer
    this.camera.position.z = 3.5
    scene.add(this._group)

    this._buildPolarGrid()
    this._buildRadialBars()
    this._buildHexFill()
    this._buildStateOverlay('loading', 'Radar initialising…')

    this._targetBandHeights = new Float32Array(NUM_BANDS)
    this._currentBandHeights = new Float32Array(NUM_BANDS)
    this._targetRms = 0
    this._currentRms = 0

    this._unsubUpdate = ringBuffer.onUpdate(d => this._onData(d))
    this._unsubFull = ringBuffer.onFull(() => this.setState('live'))
    this._unsubError = ringBuffer.onError(() => this.setState('error'))

    this.running = true
    this._animate()

    this._resizeHandler = () => this._onResize()
    window.addEventListener('resize', this._resizeHandler)
  }

  setState(state) {
    if (state === this.state) return
    this.state = state
    this._updateOverlay()
  }

  setVisible(v) { this.container.style.display = v ? '' : 'none' }

  dispose() {
    this.running = false
    window.removeEventListener('resize', this._resizeHandler)
    this._unsubUpdate()
    this._unsubFull()
    this._unsubError()
    this.renderer.dispose()
  }

  _onData(d) {
    if (!d.spectral || d.spectral.length < NUM_BANDS) return
    for (let i = 0; i < NUM_BANDS; i++) {
      this._targetBandHeights[i] = Math.min(d.spectral[i], 1) * BAR_HEIGHT_MAX
    }
    this._targetRms = Math.min(d.rms || 0, 1)
    if (this.state === 'empty') this.setState('edge')
  }

  _buildPolarGrid() {
    const mat = new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.4 })
    const segments = 64
    for (let r of [0.5, 1.0, 1.5, 1.8]) {
      const pts = []
      for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2
        pts.push(new THREE.Vector3(Math.cos(a) * r, Math.sin(a) * r, 0))
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts)
      this._group.add(new THREE.Line(geo, mat))
    }

    const spokeMat = new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.2 })
    for (let i = 0; i < NUM_BANDS; i++) {
      const a = (i / NUM_BANDS) * Math.PI * 2
      const pts = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(Math.cos(a) * MAX_RADIUS, Math.sin(a) * MAX_RADIUS, 0)]
      const geo = new THREE.BufferGeometry().setFromPoints(pts)
      this._group.add(new THREE.Line(geo, spokeMat))

      const lbl = createTextSprite(`${i}`, { fontSize: 24, color: COLORS.textDim, scale: 0.6 })
      lbl.position.set(Math.cos(a) * (MAX_RADIUS + 0.25), Math.sin(a) * (MAX_RADIUS + 0.25), 0)
      this._group.add(lbl)
    }
  }

  _buildRadialBars() {
    this._bars = []
    const geo = new THREE.BoxGeometry(0.06, 1, 0.06)
    for (let i = 0; i < NUM_BANDS; i++) {
      const a = (i / NUM_BANDS) * Math.PI * 2
      const mat = new THREE.MeshStandardMaterial({ color: COLORS.accent, emissive: COLORS.accent, emissiveIntensity: 0.3 })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(Math.cos(a) * 0.5, Math.sin(a) * 0.5, 0)
      mesh.lookAt(0, 0, 0)
      mesh.scale.y = 0.01
      this._group.add(mesh)
      this._bars.push({ mesh, angle: a })
    }
  }

  _buildHexFill() {
    const pts = []
    const n = 6
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2
      pts.push(new THREE.Vector3(Math.cos(a) * 0.01, Math.sin(a) * 0.01, 0.01))
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pts)
    const mat = new THREE.MeshBasicMaterial({ color: COLORS.accent, transparent: true, opacity: 0.15, side: THREE.DoubleSide })
    this._hexMesh = new THREE.Mesh(geo, mat)
    this._group.add(this._hexMesh)
  }

  _buildStateOverlay(state, msg) {
    this._overlay = createTextSprite(msg, { fontSize: 28, color: COLORS.textDim, scale: 1.5 })
    this._overlay.position.z = 0.5
    this._group.add(this._overlay)
  }

  _updateOverlay() {
    const msgs = {
      loading: 'Radar initialising\u2026',
      empty: 'Awaiting vibration data',
      edge: `Buffering (${this.buffer.length}/128)`,
      live: '',
      error: 'Connection lost \u2014 retrying',
    }
    if (this._overlay) {
      const txt = msgs[this.state] || ''
      this._overlay.material.map = this._makeTextCanvas(txt)
      this._overlay.material.map.needsUpdate = true
      this._overlay.visible = !!txt
    }
  }

  _makeTextCanvas(text) {
    const c = document.createElement('canvas')
    c.width = 512; c.height = 96
    const ctx = c.getContext('2d')
    ctx.font = '28px Inter, sans-serif'
    ctx.fillStyle = COLORS.textDim
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillText(text, 256, 48)
    const tex = new THREE.CanvasTexture(c)
    tex.minFilter = THREE.LinearFilter
    return tex
  }

  _animate() {
    if (!this.running) return
    requestAnimationFrame(() => this._animate())
    this._tick()
    this.renderer.render(this.scene, this.camera)
  }

  _tick() {
    for (let i = 0; i < NUM_BANDS; i++) {
      this._currentBandHeights[i] += (this._targetBandHeights[i] - this._currentBandHeights[i]) * LERP_FACTOR
      const h = Math.max(this._currentBandHeights[i], 0.01)
      const bar = this._bars[i]
      const a = bar.angle
      const r = 0.5 + h / 2
      bar.mesh.position.set(Math.cos(a) * r, Math.sin(a) * r, 0)
      bar.mesh.scale.y = h
    }

    this._currentRms += (this._targetRms - this._currentRms) * LERP_FACTOR
    const hexR = Math.max(this._currentRms * 1.5, 0.01)
    const pts = []
    for (let i = 0; i <= 6; i++) {
      const a = (i / 6) * Math.PI * 2 - Math.PI / 2
      pts.push(new THREE.Vector3(Math.cos(a) * hexR, Math.sin(a) * hexR, 0.01))
    }
    this._hexMesh.geometry.dispose()
    this._hexMesh.geometry = new THREE.BufferGeometry().setFromPoints(pts)
  }

  _onResize() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
  }
}
