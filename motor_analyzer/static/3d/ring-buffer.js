const SUBSCRIBER_TOPICS = ['update', 'full', 'error']

export class RingBuffer {
  constructor(maxlen = 256) {
    this.maxlen = maxlen
    this._buf = []
    this._subs = { update: [], full: [], error: [] }
    this._firedFull = false
  }

  push(entry) {
    if (this._buf.length >= this.maxlen) this._buf.shift()
    this._buf.push(entry)
    this._notify('update', entry)
    if (this._buf.length >= 128 && !this._firedFull) {
      this._firedFull = true
      this._notify('full', this._buf.length)
    }
    return this
  }

  latest() { return this._buf[this._buf.length - 1] ?? null }

  slice(n) { return this._buf.slice(-n) }

  get length() { return this._buf.length }
  get isFull() { return this._buf.length >= 128 }

  onUpdate(fn) { return this._subscribe('update', fn) }
  onFull(fn) { return this._subscribe('full', fn) }
  onError(fn) { return this._subscribe('error', fn) }

  emitError(msg) { this._notify('error', msg) }

  reset() {
    this._buf = []
    this._firedFull = false
  }

  _subscribe(topic, fn) {
    if (!this._subs[topic]) return () => {}
    this._subs[topic].push(fn)
    const i = this._subs[topic].length - 1
    return () => { this._subs[topic].splice(i, 1) }
  }

  _notify(topic, data) {
    for (const fn of (this._subs[topic] || [])) fn(data)
  }
}
