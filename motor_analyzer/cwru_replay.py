"""
CWRU Bearing Dataset → MotorSense Replay Bridge
Ingests .mat files from the existing data/raw/cwru/ mirror structure
and injects at 100 Hz into the running Flask app.

Usage:
  python cwru_replay.py --sequence                  # full fault sequence
  python cwru_replay.py --file 97 --mode http       # single file, HTTP mode
  python cwru_replay.py --file 105 --mode serial    # single file, serial mode
  python cwru_replay.py --inspect 97                # show .mat keys
"""

import os
import re
import sys
import time
import argparse
import numpy as np
import scipy.io as sio
from scipy import signal as sp_signal
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CWRU_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'cwru')
APP_URL = "http://127.0.0.1:5050"
TARGET_FS = 100

# CWRU .mat variable keys (DE_time covers drive-end, the standard benchmark)
CWRU_SIGNAL_KEYS = [
    'X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time',
    'X098_DE_time', 'X106_DE_time', 'X119_DE_time', 'X131_DE_time',
    'X099_DE_time', 'X107_DE_time', 'X120_DE_time', 'X132_DE_time',
    'X100_DE_time', 'X108_DE_time', 'X121_DE_time', 'X133_DE_time',
    'DE_time', 'FE_time', 'BA_time',
]
CWRU_RPM_KEYS = ['X097RPM', 'X105RPM', 'X118RPM', 'X130RPM', 'RPM', 'speed']

FAULT_SEQUENCE = [
    ('NORMAL',      97,  30, 'Normal baseline — train on this'),
    ('INNER_RACE',  105, 20, 'Inner race fault 0.007"'),
    ('BALL_FAULT',  118, 20, 'Ball fault 0.007"'),
    ('OUTER_RACE',  130, 20, 'Outer race fault 0.007" @6 o\'clock'),
]


def find_mat_file(file_num):
    """Search recursively for any .mat file whose filename contains the number."""
    root = Path(CWRU_DIR)
    for f in root.rglob('*.mat'):
        stem = f.stem
        if re.search(rf'(?<!\d){file_num}(?!\d)', stem):
            return str(f)
    return None


def load_cwru_mat(filepath):
    mat = sio.loadmat(filepath)
    sig = None
    for key in CWRU_SIGNAL_KEYS:
        if key in mat:
            sig = mat[key].flatten().astype(np.float64)
            break
    if sig is None:
        candidates = {k: v for k, v in mat.items()
                      if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim <= 2}
        if candidates:
            key = max(candidates, key=lambda k: candidates[k].size)
            sig = candidates[key].flatten().astype(np.float64)
            print(f"  [warn] using '{key}' ({sig.size} samples)")
        else:
            raise ValueError(f"No usable signal in {filepath}")

    fname = os.path.basename(filepath).lower()
    src_fs = 48000.0 if '48' in fname else 12000.0
    for k in CWRU_RPM_KEYS:
        if k in mat:
            break
    return sig, src_fs


def downsample(sig, src_fs, target_fs=TARGET_FS):
    from math import gcd
    g = gcd(int(src_fs), int(target_fs))
    up = int(target_fs) // g
    down = int(src_fs) // g
    return sp_signal.resample_poly(sig, up, down).astype(np.float64)


def replay_http(samples, loop=False):
    import requests
    try:
        r = requests.get(f"{APP_URL}/api/status", timeout=2)
        r.raise_for_status()
    except Exception as e:
        print(f"[error] Cannot reach MotorSense at {APP_URL}: {e}")
        sys.exit(1)
    probe = requests.post(f"{APP_URL}/api/inject", json={"samples": []}, timeout=2)
    if probe.status_code == 404:
        print("[error] /api/inject route not in app.py. Add it first.")
        sys.exit(1)

    chunk_size = 100
    interval = chunk_size / TARGET_FS
    iteration = 0
    while True:
        iteration += 1
        chunks = [samples[i:i+chunk_size].tolist() for i in range(0, len(samples), chunk_size)]
        print(f"  [http] loop #{iteration}: {len(chunks)} chunks")
        for chunk in chunks:
            try:
                r = requests.post(f"{APP_URL}/api/inject", json={"samples": chunk}, timeout=5)
                if not r.ok:
                    print(f"  [warn] HTTP {r.status_code}")
            except Exception as e:
                print(f"  [error] {e}")
            time.sleep(interval)
        if not loop:
            break
    print("  [http] done")


def replay_serial(samples, port, loop=False):
    import serial
    try:
        ser = serial.Serial(port, 115200, timeout=1)
    except serial.SerialException as e:
        print(f"[error] Cannot open {port}: {e}")
        print("  Create virtual port: socat -d -d pty,raw,echo=0,link=/tmp/motor_tx ...")
        sys.exit(1)
    interval = 1.0 / TARGET_FS
    iteration = 0
    while True:
        iteration += 1
        print(f"  [serial] loop #{iteration}: {len(samples)} samples")
        for v in samples:
            try:
                ser.write(f"{v:.6f}\n".encode())
            except serial.SerialException as e:
                print(f"  [error] {e}")
                ser.close()
                return
            time.sleep(interval)
        if not loop:
            break
    ser.close()
    print("  [serial] done")


def replay_sequence(mode, port=None):
    print("\n" + "=" * 60)
    print("  CWRU FAULT SEQUENCE REPLAY")
    print("=" * 60)
    for label, num, dur, note in FAULT_SEQUENCE:
        print(f"  [{label:12s}] {dur:3d}s  {note}")
    print("=" * 60)

    for label, file_num, duration_s, note in FAULT_SEQUENCE:
        fpath = find_mat_file(file_num)
        if not fpath:
            print(f"\n  [skip] File #{file_num} not found under {CWRU_DIR}")
            print(f"         Downloaded from: https://engineering.case.edu/bearingdatacenter/download-data-file")
            continue

        sig, src_fs = load_cwru_mat(fpath)
        resampled = downsample(sig, src_fs, TARGET_FS)
        n_needed = int(duration_s * TARGET_FS)
        if len(resampled) < n_needed:
            repeats = int(np.ceil(n_needed / len(resampled)))
            resampled = np.tile(resampled, repeats)[:n_needed]
        else:
            resampled = resampled[:n_needed]

        print(f"\n  >>> [{label}] {note}")
        print(f"      File: {os.path.relpath(fpath, CWRU_DIR)}")
        print(f"      {len(resampled)} samples @ {TARGET_FS} Hz ({duration_s}s)")

        if label == 'NORMAL':
            print(f"      ACTION: click 'Start Training' in MotorSense UI now")
        else:
            print(f"      ACTION: observe anomaly score in UI")
        print(f"      Starting in 3s...")
        time.sleep(3)

        if mode == 'serial':
            import serial
            interval = 1.0 / TARGET_FS
            ser = serial.Serial(port, 115200, timeout=1)
            for v in resampled:
                ser.write(f"{v:.6f}\n".encode())
                time.sleep(interval)
            ser.close()
        else:
            import requests
            chunk_size = 100
            interval = chunk_size / TARGET_FS
            chunks = [resampled[i:i+chunk_size].tolist() for i in range(0, len(resampled), chunk_size)]
            for chunk in chunks:
                requests.post(f"{APP_URL}/api/inject", json={"samples": chunk}, timeout=5)
                time.sleep(interval)

        print(f"      [{label}] complete")
        time.sleep(2)

    print("\n" + "=" * 60)
    print("  Sequence finished. Check MotorSense UI for results.")
    print("=" * 60)


def inspect_mat(file_num):
    fpath = find_mat_file(file_num)
    if not fpath:
        print(f"File #{file_num} not found under {CWRU_DIR}")
        return
    mat = sio.loadmat(fpath)
    print(f"\n{os.path.relpath(fpath, BASE_DIR)}:")
    for k, v in mat.items():
        if k.startswith('_'):
            continue
        print(f"  {k:30s} {str(v.shape):20s} {v.dtype}" if hasattr(v, 'shape') else f"  {k}: {type(v).__name__}")


def main():
    parser = argparse.ArgumentParser(description='CWRU Bearing Dataset -> MotorSense Replay Bridge')
    parser.add_argument('--mode', choices=['serial', 'http'], default='http')
    parser.add_argument('--file', type=int, default=None, help='File number (e.g. 97, 105, 118, 130)')
    parser.add_argument('--port', default='/tmp/motor_tx')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--sequence', action='store_true', help='Full fault sequence')
    parser.add_argument('--inspect', type=int, default=None, metavar='NUM', help='Show .mat keys for file number')
    args = parser.parse_args()

    if args.inspect:
        inspect_mat(args.inspect)
        return

    if args.sequence:
        replay_sequence(args.mode, args.port)
        return

    if not args.file:
        parser.print_help()
        print("\n[error] Provide --file NUM or --sequence")
        sys.exit(1)

    fpath = find_mat_file(args.file)
    if not fpath:
        print(f"[error] File #{args.file} not found under {CWRU_DIR}")
        sys.exit(1)

    print(f"[load] {os.path.relpath(fpath, BASE_DIR)}")
    sig, src_fs = load_cwru_mat(fpath)
    print(f"       {len(sig):,} samples @ {src_fs/1000:.0f} kHz -> downsampling to {TARGET_FS} Hz")
    resampled = downsample(sig, src_fs, TARGET_FS)
    print(f"       {len(resampled):,} samples after decimation ({len(resampled)/TARGET_FS:.1f}s)")

    if args.mode == 'serial':
        replay_serial(resampled, args.port, loop=args.loop)
    else:
        replay_http(resampled, loop=args.loop)


if __name__ == '__main__':
    main()
