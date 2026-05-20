import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js'
import { createScene, COLORS } from './scene.js'

const MAX_COLS = 64
const NUM_BINS = 32
const COL_SPACING = 0.06
const BIN_SPACING = 0.06
const MAX_HEIGHT = 0.5

const gradientColors = [
  new THREE.Color(0x0a4b8a),
  new THREE.Color(0x00b4d8),
  new THREE.Color(0x00d4aa),
  new THREE.Color(0xffd166),
  new THREE.Color(0xff6b35),
]

function magnitudeColor(val) {
  const t = Math.min(val, 1)
  const idx = t * (gradientColors.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.min(lo + 1, gradientColors.length - 1)
  const f = idx - lo
  return gradientColors[lo].clone().lerp(gradientColors[hi], f)
}

export class Waterfall3D {
  constructor(container, ringBuffer) {
    this.container = container
    this.buffer = ringBuffer
    this.state = 'loading'
    this._group = new THREE.Group()

    const { scene, camera, renderer } = createScene(container)
    this.scene = scene
    this.camera = camera
    this.renderer = renderer
    this.camera.position.set(0, -2.5, 3)
    this.camera.lookAt(0, 0.3, -1)

    scene.add(this._group)

    this._columns = []
    this._nextCol = 0
    this._colsFilled = 0
    this._buildFloor()
    this._buildStateOverlay('loading', 'Waterfall initialising\u2026')

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
    this._unsubUpdate(); this._unsubFull(); this._unsubError()
    this.renderer.dispose()
  }

  _onData(d) {
    if (!d.fft || d.fft.length < NUM_BINS) return

    const col = this._makeColumn(d.fft)
    this._addColumn(col)
    this._colsFilled = Math.min(this._colsFilled + 1, MAX_COLS)

    if (this.state === 'empty') this.setState('edge')
  }

  _makeColumn(fft) {
    const cols = []
    for (let i = 0; i < NUM_BINS; i++) {
      const mag = Math.min(fft[i] || 0, 1)
      const h = Math.max(mag * MAX_HEIGHT, 0.005)
      const color = magnitudeColor(mag)
      const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.3 })
      const geo = new THREE.BoxGeometry(BIN_SPACING * 0.8, h, COL_SPACING * 0.8)
      const mesh = new THREE.Mesh(geo, mat)
      const x = (i / NUM_BINS - 0.5) * NUM_BINS * BIN_SPACING
      mesh.position.set(x, h / 2, 0)
      cols.push(mesh)
    }
    return cols
  }

  _addColumn(cols) {
    for (const m of cols) this._group.add(m)

    if (this._columns.length >= MAX_COLS) {
      const oldest = this._columns.shift()
      for (const m of oldest) this._group.remove(m)
    }
    this._columns.push(cols)

    for (let ci = 0; ci < this._columns.length; ci++) {
      const z = -(this._columns.length - 1 - ci) * COL_SPACING
      for (const m of this._columns[ci]) {
        m.position.z = z
      }
    }
  }

  _buildFloor() {
    const w = NUM_BINS * BIN_SPACING * 1.2
    const d = MAX_COLS * COL_SPACING * 1.2
    const mat = new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.3 })

    for (let i = -1; i <= 1; i += 2) {
      const pts = [
        new THREE.Vector3(-w / 2, 0, i * d / 2),
        new THREE.Vector3(w / 2, 0, i * d / 2),
      ]
      this._group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat))
    }
    for (let i = -1; i <= 1; i += 2) {
      const pts = [
        new THREE.Vector3(i * w / 2, 0, -d / 2),
        new THREE.Vector3(i * w / 2, 0, d / 2),
      ]
      this._group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat))
    }
  }

  _buildStateOverlay(state, msg) {
    this._overlay = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: this._makeTextCanvas(msg),
        transparent: true,
        depthTest: false,
      })
    )
    this._overlay.position.set(0, 0.8, 0)
    this._overlay.scale.set(2, 0.4, 1)
    this._group.add(this._overlay)
  }

  _updateOverlay() {
    const msgs = {
      loading: 'Waterfall initialising\u2026',
      empty: 'Awaiting frequency data',
      edge: `Buffering (${this.buffer.length}/128)`,
      live: '',
      error: 'Connection lost \u2014 retrying',
    }
    const txt = msgs[this.state] || ''
    if (this._overlay) {
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
    this.renderer.render(this.scene, this.camera)
  }

  _onResize() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
  }
}
