import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js'
import { createScene, COLORS } from './scene.js'

const ARC_RADIUS = 1.4
const LERP_FACTOR = 0.9

export class Gauge3D {
  constructor(container, ringBuffer) {
    this.container = container
    this.buffer = ringBuffer
    this.state = 'loading'
    this._group = new THREE.Group()

    const { scene, camera, renderer } = createScene(container)
    this.scene = scene
    this.camera = camera
    this.renderer = renderer
    this.camera.position.z = 3
    scene.add(this._group)

    this._buildArc()
    this._buildTicks()
    this._buildNeedle()
    this._buildValueLabel()
    this._buildStateOverlay('loading', 'Gauge initialising\u2026')

    this._targetVal = 0
    this._currentVal = 0

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
    this._targetVal = Math.min(d.rms || 0, 1)
    if (this.state === 'empty') this.setState('edge')
  }

  _buildArc() {
    const pts = []
    const segs = 48
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * Math.PI
      pts.push(new THREE.Vector3(
        -Math.cos(a) * ARC_RADIUS,
        -Math.sin(a) * ARC_RADIUS,
        0
      ))
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pts)
    const mat = new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.5 })
    this._arcLine = new THREE.Line(geo, mat)
    this._group.add(this._arcLine)

    const colorSegs = [
      { start: 0, end: 0.4, color: 0x00d4aa },
      { start: 0.4, end: 0.7, color: 0xffd166 },
      { start: 0.7, end: 1.0, color: 0xff6b35 },
    ]
    for (const seg of colorSegs) {
      const p = []
      const s = Math.floor(segs * seg.start)
      const e = Math.ceil(segs * seg.end)
      for (let i = s; i <= e; i++) {
        const a = (i / segs) * Math.PI
        p.push(new THREE.Vector3(
          -Math.cos(a) * (ARC_RADIUS + 0.04),
          -Math.sin(a) * (ARC_RADIUS + 0.04),
          0
        ))
      }
      const g = new THREE.BufferGeometry().setFromPoints(p)
      const m = new THREE.LineBasicMaterial({ color: seg.color, linewidth: 2 })
      this._group.add(new THREE.Line(g, m))
    }
  }

  _buildTicks() {
    for (let i = 0; i <= 4; i++) {
      const t = i / 4
      const a = t * Math.PI
      const inner = ARC_RADIUS - 0.1
      const outer = ARC_RADIUS + 0.1
      const pts = [
        new THREE.Vector3(-Math.cos(a) * inner, -Math.sin(a) * inner, 0),
        new THREE.Vector3(-Math.cos(a) * outer, -Math.sin(a) * outer, 0),
      ]
      const geo = new THREE.BufferGeometry().setFromPoints(pts)
      const mat = new THREE.LineBasicMaterial({ color: COLORS.textPrimary, transparent: true, opacity: 0.6 })
      this._group.add(new THREE.Line(geo, mat))
    }
  }

  _buildNeedle() {
    const geo = new THREE.CylinderGeometry(0.02, 0.02, ARC_RADIUS * 0.8, 8)
    const mat = new THREE.MeshStandardMaterial({
      color: COLORS.accent,
      emissive: COLORS.accent,
      emissiveIntensity: 0.4,
    })
    this._needle = new THREE.Mesh(geo, mat)
    this._needle.position.set(0, 0, 0.02)
    this._needle.rotation.x = Math.PI / 2
    this._group.add(this._needle)

    const pivot = new THREE.Mesh(
      new THREE.SphereGeometry(0.06, 16, 16),
      new THREE.MeshStandardMaterial({ color: COLORS.textPrimary, emissive: COLORS.accent, emissiveIntensity: 0.2 })
    )
    pivot.position.z = 0.02
    this._group.add(pivot)
  }

  _buildValueLabel() {
    const c = document.createElement('canvas')
    c.width = 256; c.height = 96
    this._valCanvas = c
    this._valCtx = c.getContext('2d')
    this._valCtx.font = 'bold 48px JetBrains Mono, monospace'
    this._valCtx.textAlign = 'center'
    this._valCtx.textBaseline = 'middle'

    const tex = new THREE.CanvasTexture(c)
    tex.minFilter = THREE.LinearFilter
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false })
    this._valueSprite = new THREE.Sprite(mat)
    this._valueSprite.position.set(0, -0.3, 0.1)
    this._valueSprite.scale.set(1.2, 0.45, 1)
    this._group.add(this._valueSprite)
  }

  _buildStateOverlay(state, msg) {
    this._overlay = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: this._makeTextCanvas(msg),
        transparent: true,
        depthTest: false,
      })
    )
    this._overlay.position.set(0, 0.8, 0.1)
    this._overlay.scale.set(1.5, 0.35, 1)
    this._group.add(this._overlay)
  }

  _updateOverlay() {
    const msgs = {
      loading: 'Gauge initialising\u2026',
      empty: 'Awaiting RMS value',
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
    ctx.font = '24px Inter, sans-serif'
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
    this._currentVal += (this._targetVal - this._currentVal) * LERP_FACTOR
    const angle = this._currentVal * Math.PI
    this._needle.rotation.z = -(angle - Math.PI / 2)

    const color = this._currentVal < 0.4 ? '#00D4AA' : this._currentVal < 0.7 ? '#FFD166' : '#FF6B35'
    this._valCtx.clearRect(0, 0, 256, 96)
    this._valCtx.fillStyle = color
    this._valCtx.font = 'bold 48px JetBrains Mono, monospace'
    this._valCtx.textAlign = 'center'
    this._valCtx.textBaseline = 'middle'
    this._valCtx.fillText((this._currentVal * 100).toFixed(1) + '%', 128, 48)
    this._valueSprite.material.map.needsUpdate = true
  }

  _onResize() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
  }
}
