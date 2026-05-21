import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js'
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js'
import { createScene, COLORS } from './scene.js'

const POLL_INTERVAL = 3000
const MAX_POINTS = 200

export class Scatter3D {
  constructor(container, ringBuffer) {
    this.container = container
    this.buffer = ringBuffer
    this.state = 'loading'
    this._group = new THREE.Group()

    const { scene, camera, renderer } = createScene(container)
    this.scene = scene
    this.camera = camera
    this.renderer = renderer
    this.camera.position.set(3, 2, 4)
    this.camera.lookAt(0, 0, 0)

    this.controls = new OrbitControls(this.camera, renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.8

    scene.add(this._group)
    this._buildAxes()
    this._buildPointCloud()
    this._buildStateOverlay('loading', 'Loading embedding\u2026')

    this._data = []
    this._pollTimer = null
    this._startPolling()

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
    clearTimeout(this._pollTimer)
    this.controls.dispose()
    this.renderer.dispose()
  }

  _startPolling() {
    const poll = () => {
      if (!this.running) return
      fetch('/api/features/embedding')
        .then(r => r.json())
        .then(data => {
          if (data.status === 'active' && data.points?.length > 0) {
            this._data = data.points.slice(-MAX_POINTS)
            this._updatePointCloud()
            this._updateAxesLabels(data.explained_variance)
            if (this.state === 'loading' || this.state === 'empty') this.setState('live')
          } else {
            if (this.state === 'loading') this.setState('empty')
          }
          this._pollTimer = setTimeout(poll, POLL_INTERVAL)
        })
        .catch(() => {
          this.setState('error')
          this._pollTimer = setTimeout(poll, POLL_INTERVAL)
        })
    }
    poll()
  }

  _buildAxes() {
    const mat = new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.5 })
    const len = 2.2
    for (const axis of ['x', 'y', 'z']) {
      const pts = [
        new THREE.Vector3(0, 0, 0),
        axis === 'x' ? new THREE.Vector3(len, 0, 0) :
        axis === 'y' ? new THREE.Vector3(0, len, 0) :
        new THREE.Vector3(0, 0, len),
      ]
      const geo = new THREE.BufferGeometry().setFromPoints(pts)
      const line = new THREE.Line(geo, mat)
      this._group.add(line)
    }

    this._axisLabels = {}
    for (const axis of ['x', 'y', 'z']) {
      const pos = axis === 'x' ? [len + 0.2, 0, 0] :
                  axis === 'y' ? [0, len + 0.2, 0] :
                  [0, 0, len + 0.2]
      const canvas = document.createElement('canvas')
      canvas.width = 128; canvas.height = 64
      const ctx = canvas.getContext('2d')
      ctx.font = 'bold 28px Inter, sans-serif'
      ctx.fillStyle = COLORS.textDim
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
      ctx.fillText(`PC${axis.toUpperCase()} (--)`, 64, 32)
      const tex = new THREE.CanvasTexture(canvas)
      tex.minFilter = THREE.LinearFilter
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }))
      sprite.position.set(pos[0], pos[1], pos[2])
      sprite.scale.set(1.2, 0.35, 1)
      this._group.add(sprite)
      this._axisLabels[axis] = { canvas, ctx, sprite }
    }

    const gridMat = new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.15 })
    for (let i = -2; i <= 2; i++) {
      for (const fixed of ['xy', 'xz', 'yz']) {
        const pts = []
        for (let j = -2; j <= 2; j++) {
          let p
          if (fixed === 'xy') p = new THREE.Vector3(i * 0.8, j * 0.8, 0)
          else if (fixed === 'xz') p = new THREE.Vector3(i * 0.8, 0, j * 0.8)
          else p = new THREE.Vector3(0, i * 0.8, j * 0.8)
          pts.push(p)
        }
        const geo = new THREE.BufferGeometry().setFromPoints(pts)
        this._group.add(new THREE.Line(geo, gridMat))
      }
    }
  }

  _updateAxesLabels(explainedVariance) {
    if (!explainedVariance) return
    const axes = ['x', 'y', 'z']
    for (let i = 0; i < axes.length && i < explainedVariance.length; i++) {
      const ax = axes[i]
      const label = this._axisLabels[ax]
      if (!label) continue
      const pct = (explainedVariance[i] * 100).toFixed(1)
      label.ctx.clearRect(0, 0, 128, 64)
      label.ctx.font = 'bold 24px Inter, sans-serif'
      label.ctx.fillStyle = COLORS.textDim
      label.ctx.textAlign = 'center'; label.ctx.textBaseline = 'middle'
      label.ctx.fillText(`PC${ax.toUpperCase()} (${pct}%)`, 64, 32)
      label.sprite.material.map.needsUpdate = true
    }
  }

  _buildPointCloud() {
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(MAX_POINTS * 3)
    const colors = new Float32Array(MAX_POINTS * 3)
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geo.setDrawRange(0, 0)

    const mat = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    this._points = new THREE.Points(geo, mat)
    this._group.add(this._points)
  }

  _updatePointCloud() {
    const n = Math.min(this._data.length, MAX_POINTS)
    const pos = this._points.geometry.attributes.position.array
    const col = this._points.geometry.attributes.color.array

    for (let i = 0; i < n; i++) {
      const p = this._data[i]
      pos[i * 3] = p.x * 2
      pos[i * 3 + 1] = p.y * 2
      pos[i * 3 + 2] = (p.z || 0) * 2
      const isAnom = p.anomaly
      const intensity = Math.min(Math.abs(p.score || 0) * 2, 1)
      if (isAnom) {
        col[i * 3] = 1; col[i * 3 + 1] = 0.15; col[i * 3 + 2] = 0.15
      } else {
        col[i * 3] = 0; col[i * 3 + 1] = 0.83; col[i * 3 + 2] = 0.67
      }
    }
    this._points.geometry.attributes.position.needsUpdate = true
    this._points.geometry.attributes.color.needsUpdate = true
    this._points.geometry.setDrawRange(0, n)
  }

  _buildStateOverlay(state, msg) {
    this._overlay = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: this._makeTextCanvas(msg),
        transparent: true,
        depthTest: false,
      })
    )
    this._overlay.position.set(0, 0.5, 0.5)
    this._overlay.scale.set(2, 0.4, 1)
    this._group.add(this._overlay)
  }

  _updateOverlay() {
    const msgs = {
      loading: 'Loading embedding\u2026',
      empty: 'Insufficient data — collect more',
      live: '',
      error: 'Connection error',
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
    this.controls.update()
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
