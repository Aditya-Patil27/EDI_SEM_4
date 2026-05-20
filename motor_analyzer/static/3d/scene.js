import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js'

const COLORS = {
  bg: 0x0a0e17,
  accent: 0x00d4aa,
  accentWarm: 0xff6b35,
  glass: '#1A1F2ECC',
  textDim: '#6B7280',
  textPrimary: '#E8EDF5',
  grid: 0x1e293b,
}

export function createScene(container) {
  const w = container.clientWidth
  const h = container.clientHeight

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(COLORS.bg)

  const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100)
  camera.position.set(0, 0, 5)
  camera.lookAt(0, 0, 0)

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setSize(w, h)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setClearColor(COLORS.bg, 1)
  container.appendChild(renderer.domElement)

  const ambient = new THREE.AmbientLight(0x404060, 0.6)
  scene.add(ambient)
  const dir = new THREE.DirectionalLight(0xffffff, 1.2)
  dir.position.set(5, 10, 7)
  scene.add(dir)
  const fill = new THREE.DirectionalLight(0x00d4aa, 0.3)
  fill.position.set(-5, 0, 5)
  scene.add(fill)

  return { scene, camera, renderer, dispose: () => renderer.dispose() }
}

export function createTextSprite(text, opts = {}) {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const fontSize = opts.fontSize || 36
  canvas.width = 512
  canvas.height = 128
  ctx.font = `${fontSize}px Inter, sans-serif`
  ctx.fillStyle = opts.color || COLORS.textPrimary
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, 256, 64)

  const tex = new THREE.CanvasTexture(canvas)
  tex.minFilter = THREE.LinearFilter
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false })
  const sprite = new THREE.Sprite(mat)
  sprite.scale.set(opts.scale || 2, (opts.scale || 2) * 0.25, 1)
  return sprite
}

export { COLORS }
