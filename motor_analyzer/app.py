"""
Motor Frequency Analyzer — Flask Backend
ESP32 + ADXL345 → Real-time FFT + ML anomaly detection
"""

import os
import time
import pickle
import threading
import serial
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# ─────────────────────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'motor_analyzer_secret_2024'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  Shared Application State
# ─────────────────────────────────────────────────────────────
state = {
    'serial_connected': False,
    'serial_port': '/dev/cu.usbserial-0001',
    'baud_rate': 115200,
    'training': False,
    'evaluating': False,
    'train_duration': 20,
    'train_start': 0,
    'baseline': None,
    'baseline_samples': [],
    'baseline_ready': False,
    'training_features': [],
    'active_model': None,
    'active_model_name': None,
    'is_anomaly': False,
    'anomaly_score': 0.0,
    'dominant_freq': 0.0,
    'current_model_obj': None,
    'status': 'IDLE — Connect a serial port to begin',
    'status_level': 'info',   # info | warning | error | success | training
    'motor': 0,               # 0=stopped, 1=motor1, 2=motor2
    'motor_speed': 200,       # PWM 0-255
}

ser = None
ser_lock = threading.Lock()

# Rolling raw sample buffer for FFT
RAW_WINDOW = 256          # samples used for FFT
SAMPLE_RATE = 100         # approximate Hz from ESP32
raw_buffer = []
plot_buffer = []          # downsampled for waveform display

# ─────────────────────────────────────────────────────────────
#  Feature Engineering (Spectral + Temporal)
# ─────────────────────────────────────────────────────────────
def extract_features(data: list, baseline: float, fs: float = SAMPLE_RATE) -> list:
    """
    Comprehensive feature vector combining time-domain and frequency-domain.
    Returns a 1D list of features.
    """
    arr = np.array(data, dtype=np.float64) - baseline

    # --- Time Domain ---
    rms = float(np.sqrt(np.mean(arr ** 2)))
    p2p = float(np.ptp(arr))
    variance = float(np.var(arr))
    skewness = float(np.mean(((arr - arr.mean()) / (arr.std() + 1e-8)) ** 3))
    kurtosis = float(np.mean(((arr - arr.mean()) / (arr.std() + 1e-8)) ** 4))
    crest = float(np.max(np.abs(arr)) / (rms + 1e-8))
    shape_factor = rms / (np.mean(np.abs(arr)) + 1e-8)
    zcr = float(np.sum(np.diff(np.sign(arr)) != 0) / len(arr))

    # --- Frequency Domain (FFT) ---
    win = signal.windows.hann(len(arr))
    spectrum = np.abs(rfft(arr * win))
    freqs = rfftfreq(len(arr), d=1.0 / fs)

    # Dominant frequency
    dom_idx = int(np.argmax(spectrum[1:])) + 1   # skip DC
    dom_freq = float(freqs[dom_idx])

    # Spectral centroid
    spec_sum = spectrum.sum() + 1e-8
    centroid = float(np.sum(freqs * spectrum) / spec_sum)

    # Spectral spread
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spec_sum))

    # Spectral flatness (Wiener entropy)
    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-8)))
    arith_mean = np.mean(spectrum) + 1e-8
    flatness = float(geo_mean / arith_mean)

    # Band energy ratios (low/mid/high)
    low_mask  = freqs < fs * 0.1
    mid_mask  = (freqs >= fs * 0.1) & (freqs < fs * 0.3)
    high_mask = freqs >= fs * 0.3
    total_e = np.sum(spectrum ** 2) + 1e-8
    e_low  = float(np.sum(spectrum[low_mask]  ** 2) / total_e)
    e_mid  = float(np.sum(spectrum[mid_mask]  ** 2) / total_e)
    e_high = float(np.sum(spectrum[high_mask] ** 2) / total_e)

    # Top-3 spectral peaks
    sorted_idx = np.argsort(spectrum)[::-1]
    peak_freqs = [float(freqs[i]) for i in sorted_idx[:3]]

    return [
        rms, p2p, variance, skewness, kurtosis, crest, shape_factor, zcr,
        dom_freq, centroid, spread, flatness,
        e_low, e_mid, e_high,
        *peak_freqs,
    ]


def dominant_frequency_from_buffer(data: list, baseline: float, fs: float = SAMPLE_RATE) -> float:
    if len(data) < 8:
        return 0.0
    arr = np.array(data, dtype=np.float64) - baseline
    win = signal.windows.hann(len(arr))
    spectrum = np.abs(rfft(arr * win))
    freqs = rfftfreq(len(arr), d=1.0 / fs)
    dom_idx = int(np.argmax(spectrum[1:])) + 1
    return float(freqs[dom_idx])


def build_fft_payload(data: list, baseline: float, fs: float = SAMPLE_RATE) -> dict:
    """Build FFT magnitude array for frontend chart."""
    if len(data) < 8:
        return {'freqs': [], 'magnitudes': [], 'dominant': 0.0}
    arr = np.array(data, dtype=np.float64) - baseline
    win = signal.windows.hann(len(arr))
    spectrum = np.abs(rfft(arr * win))
    freqs = rfftfreq(len(arr), d=1.0 / fs)
    dom_idx = int(np.argmax(spectrum[1:])) + 1
    return {
        'freqs': freqs.tolist(),
        'magnitudes': (spectrum / (spectrum.max() + 1e-8)).tolist(),  # normalised
        'dominant': float(freqs[dom_idx]),
    }


# ─────────────────────────────────────────────────────────────
#  ML Model Builder
# ─────────────────────────────────────────────────────────────
def build_model(model_type: str = 'ensemble'):
    """
    Returns an sklearn Pipeline with scaler + chosen anomaly detector.
    'ensemble' = stacked IsolationForest + OC-SVM vote (best for vibration).
    """
    if model_type == 'isoforest':
        clf = IsolationForest(n_estimators=300, contamination=0.03, random_state=42)
    elif model_type == 'ocsvm':
        clf = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    else:  # ensemble — we train both and combine
        clf = None   # handled in fit_model
    scaler = StandardScaler()
    if clf is not None:
        return Pipeline([('scaler', scaler), ('clf', clf)])
    return None


class EnsembleModel:
    """
    Trains IsolationForest + OC-SVM.  Anomaly if *both* agree → fewer false alarms.
    Anomaly score is mean of both normalised scores.
    """
    def __init__(self):
        self.if_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', IsolationForest(n_estimators=300, contamination=0.03, random_state=42))
        ])
        self.svm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneClassSVM(kernel='rbf', nu=0.05, gamma='scale'))
        ])
        self.trained = False

    def fit(self, X):
        X = np.array(X)
        self.if_pipe.fit(X)
        self.svm_pipe.fit(X)
        self.trained = True

    def predict_score(self, x):
        """Returns (is_anomaly: bool, score: float 0-1)."""
        x = np.array(x).reshape(1, -1)
        if_pred  = self.if_pipe.predict(x)[0]          # +1 or -1
        if_score = -self.if_pipe.score_samples(x)[0]   # higher = more anomalous

        svm_pred  = self.svm_pipe.predict(x)[0]
        svm_score = -self.svm_pipe.score_samples(x)[0]

        # Normalise scores to 0-1 using sigmoid-ish
        def normalise(s): return float(1 / (1 + np.exp(-s + 1.5)))

        combined = (normalise(if_score) + normalise(svm_score)) / 2.0
        is_anomaly = (if_pred == -1) and (svm_pred == -1)
        return is_anomaly, combined


# ─────────────────────────────────────────────────────────────
#  Serial Reader Thread
# ─────────────────────────────────────────────────────────────
def serial_reader():
    global ser, raw_buffer, plot_buffer

    DOWNSAMPLE = 3
    plot_counter = 0
    feature_counter = 0
    FEATURE_WINDOW = 128

    while True:
        with ser_lock:
            active_ser = ser

        if active_ser is None or not active_ser.is_open:
            eventlet.sleep(0.5)
            continue

        try:
            while active_ser.in_waiting > 0:
                line = active_ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                # Lines starting with '#' are ESP32 status/debug messages
                if line.startswith('#'):
                    socketio.emit('esp32_log', {'msg': line[1:].strip()})
                    continue
                try:
                    val = float(line)
                except ValueError:
                    continue

                # ── Baseline calibration ──
                if not state['baseline_ready']:
                    state['baseline_samples'].append(val)
                    if len(state['baseline_samples']) >= 80:
                        state['baseline'] = float(np.mean(state['baseline_samples']))
                        state['baseline_ready'] = True
                        state['status'] = 'BASELINE LOCKED ✓ — Ready to train or evaluate'
                        state['status_level'] = 'success'
                        socketio.emit('status_update', {
                            'status': state['status'],
                            'level': state['status_level'],
                            'baseline': state['baseline'],
                        })
                    continue

                baseline = state['baseline']

                # ── Raw buffer for FFT & features ──
                raw_buffer.append(val)
                if len(raw_buffer) > RAW_WINDOW * 2:
                    raw_buffer = raw_buffer[-RAW_WINDOW:]

                # ── Downsampled waveform for oscilloscope ──
                plot_counter += 1
                if plot_counter >= DOWNSAMPLE:
                    plot_buffer.append(round(val - baseline, 4))
                    if len(plot_buffer) > 600:
                        plot_buffer = plot_buffer[-600:]
                    plot_counter = 0

                # ── Feature extraction every FEATURE_WINDOW samples ──
                feature_counter += 1
                if feature_counter >= FEATURE_WINDOW and len(raw_buffer) >= FEATURE_WINDOW:
                    feature_counter = 0
                    chunk = raw_buffer[-FEATURE_WINDOW:]
                    feats = extract_features(chunk, baseline, SAMPLE_RATE)
                    dom_freq = feats[8]   # dominant frequency index

                    fft_payload = build_fft_payload(chunk, baseline, SAMPLE_RATE)
                    state['dominant_freq'] = dom_freq

                    is_anomaly = False
                    score = 0.0

                    if state['training']:
                        state['training_features'].append(feats)
                        elapsed = time.time() - state['train_start']
                        progress = min(elapsed / state['train_duration'], 1.0)
                        if elapsed >= state['train_duration']:
                            # Auto-complete training
                            _finish_training_internal()
                        else:
                            socketio.emit('train_progress', {
                                'elapsed': round(elapsed, 1),
                                'total': state['train_duration'],
                                'progress': round(progress * 100, 1),
                                'samples': len(state['training_features']),
                            })

                    elif state['evaluating'] and state['current_model_obj'] is not None:
                        m = state['current_model_obj']
                        is_anomaly, score = m.predict_score(feats)
                        state['is_anomaly'] = bool(is_anomaly)
                        state['anomaly_score'] = float(score)

                    # Emit real-time data to all clients
                    socketio.emit('sensor_data', {
                        'waveform': plot_buffer[-200:],
                        'fft': fft_payload,
                        'dominant_freq': round(dom_freq, 2),
                        'is_anomaly': bool(is_anomaly),
                        'anomaly_score': round(score * 100, 1),
                        'evaluating': state['evaluating'],
                        'training': state['training'],
                    })

        except Exception as e:
            socketio.emit('status_update', {
                'status': f'Serial error: {e}',
                'level': 'error',
            })
            eventlet.sleep(0.1)

        eventlet.sleep(0.01)


def _finish_training_internal():
    """Called from serial thread when training window is complete."""
    if not state['training_features']:
        state['status'] = 'No data collected — retry training'
        state['status_level'] = 'error'
        state['training'] = False
        return

    pending_name = state.get('pending_model_name', 'model')
    model_obj = EnsembleModel()
    model_obj.fit(state['training_features'])

    state['current_model_obj'] = model_obj
    state['active_model_name'] = pending_name
    state['training'] = False
    state['evaluating'] = True
    state['status'] = f'MODEL "{pending_name}" TRAINED ✓ — Inference active'
    state['status_level'] = 'success'

    socketio.emit('status_update', {
        'status': state['status'],
        'level': state['status_level'],
    })
    socketio.emit('train_complete', {'model_name': pending_name})


# ─────────────────────────────────────────────────────────────
#  REST API
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ports')
def list_ports():
    """List available serial ports."""
    import serial.tools.list_ports
    ports = [{'device': p.device, 'desc': p.description} for p in serial.tools.list_ports.comports()]
    return jsonify(ports)


@app.route('/api/connect', methods=['POST'])
def connect_serial():
    global ser
    data = request.json
    port = data.get('port', state['serial_port'])
    baud = int(data.get('baud', state['baud_rate']))

    with ser_lock:
        if ser and ser.is_open:
            ser.close()
        try:
            ser = serial.Serial(port, baud, timeout=0.01)
            ser.flushInput()
            state['serial_connected'] = True
            state['serial_port'] = port
            state['baud_rate'] = baud
            state['baseline_ready'] = False
            state['baseline_samples'] = []
            state['baseline'] = None
            state['status'] = 'CONNECTED — Calibrating baseline (hold motor still)…'
            state['status_level'] = 'info'
            return jsonify({'ok': True, 'port': port})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/api/disconnect', methods=['POST'])
def disconnect_serial():
    global ser
    with ser_lock:
        if ser and ser.is_open:
            ser.close()
        ser = None
        state['serial_connected'] = False
        state['training'] = False
        state['evaluating'] = False
        state['status'] = 'DISCONNECTED'
        state['status_level'] = 'info'
    return jsonify({'ok': True})


@app.route('/api/train/start', methods=['POST'])
def start_training():
    data = request.json
    name = (data.get('name') or 'model').strip().replace(' ', '_')
    duration = int(data.get('duration', 20))

    if not state['baseline_ready']:
        return jsonify({'ok': False, 'error': 'Baseline not calibrated yet'}), 400

    state['training'] = True
    state['evaluating'] = False
    state['training_features'] = []
    state['train_start'] = time.time()
    state['train_duration'] = duration
    state['pending_model_name'] = name
    state['status'] = f'TRAINING "{name}" — {duration}s capture…'
    state['status_level'] = 'training'
    socketio.emit('status_update', {'status': state['status'], 'level': state['status_level']})
    return jsonify({'ok': True, 'name': name, 'duration': duration})


@app.route('/api/train/cancel', methods=['POST'])
def cancel_training():
    state['training'] = False
    state['status'] = 'Training cancelled'
    state['status_level'] = 'info'
    return jsonify({'ok': True})


@app.route('/api/model/save', methods=['POST'])
def save_model():
    data = request.json
    name = (data.get('name') or 'model').strip().replace(' ', '_')
    if state['current_model_obj'] is None:
        return jsonify({'ok': False, 'error': 'No model in memory'}), 400

    payload = {
        'model': state['current_model_obj'],
        'baseline': state['baseline'],
        'name': name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'samples': len(state.get('training_features', [])),
    }
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    return jsonify({'ok': True, 'path': path, 'name': name})


@app.route('/api/model/load', methods=['POST'])
def load_model():
    data = request.json
    name = data.get('name', '')
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        return jsonify({'ok': False, 'error': 'Model file not found'}), 404
    with open(path, 'rb') as f:
        payload = pickle.load(f)

    state['current_model_obj'] = payload['model']
    state['active_model_name'] = payload['name']
    state['baseline'] = payload['baseline']
    state['baseline_ready'] = True
    state['evaluating'] = True
    state['training'] = False
    state['status'] = f'MODEL "{name}" LOADED ✓ — Inference active'
    state['status_level'] = 'success'
    socketio.emit('status_update', {'status': state['status'], 'level': state['status_level']})
    return jsonify({'ok': True, 'name': name, 'meta': {
        'timestamp': payload.get('timestamp', 'N/A'),
        'samples': payload.get('samples', 0),
        'baseline': round(payload['baseline'], 4),
    }})


@app.route('/api/model/list')
def list_models():
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pkl'):
            path = os.path.join(MODELS_DIR, f)
            try:
                with open(path, 'rb') as fh:
                    p = pickle.load(fh)
                models.append({
                    'name': p.get('name', f[:-4]),
                    'filename': f[:-4],
                    'timestamp': p.get('timestamp', 'N/A'),
                    'samples': p.get('samples', 0),
                    'baseline': round(p.get('baseline', 0), 4),
                })
            except Exception:
                models.append({'name': f[:-4], 'filename': f[:-4], 'timestamp': 'N/A', 'samples': 0, 'baseline': 0})
    return jsonify(models)


@app.route('/api/model/delete', methods=['POST'])
def delete_model():
    data = request.json
    name = data.get('name', '').replace(' ', '_')
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'error': 'Not found'}), 404


@app.route('/api/evaluate/stop', methods=['POST'])
def stop_evaluate():
    state['evaluating'] = False
    state['status'] = 'Evaluation stopped'
    state['status_level'] = 'info'
    return jsonify({'ok': True})


@app.route('/api/motor/control', methods=['POST'])
def motor_control():
    """Send motor command to ESP32 over serial."""
    data = request.json
    motor = int(data.get('motor', 0))   # 0=stop, 1=motor1, 2=motor2
    speed = int(data.get('speed', state['motor_speed']))
    speed = max(0, min(255, speed))

    with ser_lock:
        active_ser = ser

    if active_ser is None or not active_ser.is_open:
        return jsonify({'ok': False, 'error': 'Not connected'}), 400

    # Build command string for ESP32
    if motor == 1:
        cmd = f'SPEED:{speed}\n1\n'
        label = 'MOTOR 1'
    elif motor == 2:
        cmd = f'SPEED:{speed}\n2\n'
        label = 'MOTOR 2'
    else:
        cmd = '0\n'
        label = 'STOP'

    try:
        active_ser.write(cmd.encode('utf-8'))
        state['motor'] = motor
        state['motor_speed'] = speed
        socketio.emit('motor_state', {'motor': motor, 'speed': speed, 'label': label})
        return jsonify({'ok': True, 'motor': motor, 'speed': speed})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/status')
def get_status():
    return jsonify({
        'connected': state['serial_connected'],
        'baseline_ready': state['baseline_ready'],
        'training': state['training'],
        'evaluating': state['evaluating'],
        'active_model': state['active_model_name'],
        'status': state['status'],
        'status_level': state['status_level'],
        'motor': state['motor'],
        'motor_speed': state['motor_speed'],
    })


# ─────────────────────────────────────────────────────────────
#  SocketIO Events
# ─────────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('status_update', {
        'status': state['status'],
        'level': state['status_level'],
    })


# ─────────────────────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    reader_thread = threading.Thread(target=serial_reader, daemon=True)
    reader_thread.start()
    print("🚀  Motor Analyzer running → http://127.0.0.1:5050")
    socketio.run(app, host='0.0.0.0', port=5050, debug=False)
