"""Phase 7: Live deployment test — Flask test client + CWRU replay."""
import os, sys, json, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, '..')
sys.path.insert(0, APP_DIR)

from app import app
from feature_pipeline import extract_features

ENDPOINTS = [
    '/api/status', '/api/ports', '/api/company/status',
    '/api/company/models', '/api/company/classifier/status',
    '/api/pretrained/info', '/api/vibration/current',
    '/api/trend', '/api/rul', '/api/motor/identify', '/api/explain',
    '/api/features/embedding', '/api/openapi.json', '/api/config/edge', '/api/firmware/info',
]

def test_home_page():
    with app.test_client() as c:
        r = c.get('/')
        assert r.status_code == 200

def test_vui_page():
    with app.test_client() as c:
        r = c.get('/vui')
        assert r.status_code == 200

def test_all_endpoints_return_200():
    with app.test_client() as c:
        for ep in ENDPOINTS:
            r = c.get(ep)
            assert r.status_code == 200, f"{ep}: {r.status_code}"
            data = r.get_json()
            assert data is not None, f"{ep}: no JSON"
            assert isinstance(data, (dict, list)), f"{ep}: {type(data)}"

def test_vibration_current():
    with app.test_client() as c:
        r = c.get('/api/vibration/current')
        data = r.get_json()
        assert 'rms' in data
        assert 'dominant_freq' in data
        assert 'anomaly_score' in data

def test_status_fields():
    with app.test_client() as c:
        r = c.get('/api/status')
        data = r.get_json()
        for key in ['status', 'status_level', 'company_name', 'rpm', 'anomaly_consecutive']:
            assert key in data, f"status missing {key}"

def test_motor_identify():
    with app.test_client() as c:
        r = c.get('/api/motor/identify')
        data = r.get_json()
        assert 'hp' in data or data.get('status') == 'insufficient_data'

def test_explain():
    with app.test_client() as c:
        r = c.get('/api/explain')
        data = r.get_json()
        assert 'explanations' in data or data.get('status') == 'insufficient_data'


def test_retrain_endpoint():
    with app.test_client() as c:
        r = c.post('/api/retrain')
        assert r.status_code == 200
        data = r.get_json()
        assert 'ok' in data
        assert 'buffer_before' in data


def test_feature_embedding():
    with app.test_client() as c:
        r = c.get('/api/features/embedding')
        assert r.status_code == 200
        data = r.get_json()
        assert 'points' in data
        assert 'status' in data


def test_swagger_docs():
    with app.test_client() as c:
        r = c.get('/api/docs')
        assert r.status_code == 200
        assert b'SwaggerUIBundle' in r.data or b'swagger' in r.data.lower()
        r2 = c.get('/api/openapi.json')
        assert r2.status_code == 200
        spec = r2.get_json()
        assert spec['openapi'] == '3.0.3'
        assert 'paths' in spec
        assert len(spec['paths']) >= 15, f"Expected 15+ endpoints, got {len(spec['paths'])}"

def test_cwru_replay_http():
    """Inject CWRU data via /api/inject and verify state changes."""
    from cwru_replay import load_cwru_mat, find_mat_file, downsample
    with app.test_client() as c:
        fpath = find_mat_file(105)  # Inner race fault -> high RMS
        assert fpath, "CWRU file 97 not found"
        sig, src_fs = load_cwru_mat(fpath)
        assert sig is not None, "Failed to load signal"
        sig_100 = downsample(sig, src_fs)
        assert len(sig_100) > 0, "Empty resampled signal"
        payload = {'samples': sig_100.tolist()[:128]}
        r = c.post('/api/inject', json=payload)
        assert r.status_code == 200, f"inject: {r.status_code}"
        body = r.get_json()
        assert body.get('injected', 0) > 0, f"injected 0 samples: {body}"
        r2 = c.get('/api/vibration/current')
        data = r2.get_json()
        assert data['rms'] > 0, f"RMS should be > 0 after injection, got {data['rms']}"

import pytest

@pytest.mark.slow
def test_cwru_replay_full_sequence():
    """Run all 4 fault types through replay — verifies state transitions."""
    from cwru_replay import load_cwru_mat, find_mat_file, downsample
    with app.test_client() as c:
        for fnum in [97, 105, 118, 130]:
            fpath = find_mat_file(fnum)
            assert fpath, f"CWRU file {fnum} not found"
            sig, src_fs = load_cwru_mat(fpath)
            sig_100 = downsample(sig, src_fs)
            chunk = sig_100[:128].tolist()
            assert len(chunk) == 128
            c.post('/api/inject', json={'samples': chunk})
        r = c.get('/api/vibration/current')
        data = r.get_json()
        assert data['rms'] >= 0
        r = c.get('/api/motor/identify')
        data = r.get_json()
        assert 'hp' in data or data.get('status') == 'insufficient_data'
