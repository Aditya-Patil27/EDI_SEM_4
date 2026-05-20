# Testing Guide

## Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run feature pipeline tests only
python -m pytest tests/test_feature_pipeline.py -v

# Run ML model tests only
python -m pytest tests/test_ml_models.py -v
```

### Feature Pipeline Tests (11 tests)

| Test | What it validates |
|------|-------------------|
| `test_extract_features_returns_28_values` | Feature vector is exactly 28 floats |
| `test_extract_features_no_baseline` | Zero-baseline → stable finite features, RMS > 0 |
| `test_extract_features_baseline_correction` | DC offset subtraction → RMS near zero |
| `test_known_sine_wave` | 10 Hz sine → dominant frequency ≈ 10 Hz |
| `test_different_amplitudes` | Louder signal → higher RMS, same freq content |
| `test_extract_features_short_buffer` | 3-element buffer → 28 zeros (no crash) |
| `test_extract_features_empty` | Empty buffer → all zeros |
| `test_dominant_frequency_sine` | 15 Hz sine → dominant freq ≈ 15 Hz |
| `test_dominant_frequency_empty` | Empty → 0.0 |
| `test_build_fft_payload_structure` | FFT payload has correct keys, matching lengths |
| `test_different_fault_patterns` | Normal < Unbalanced/Bearing in RMS (synthetic) |

### ML Model Tests (7 tests)

| Test | What it validates |
|------|-------------------|
| `test_ensemble_train_and_predict` | EnsembleAnomalyModel trains, returns bool+float |
| `test_ensemble_detects_anomalies` | Extreme values flagged as anomalous |
| `test_gmm_train_and_predict` | GMMAnomalyDetector trains, returns valid scores |
| `test_company_classifier_train_and_predict` | CompanyClassifier trains on synthetic 3-company data |
| `test_company_classifier_save_load` | Save/load cycle preserves training state |
| `test_per_company_registry` | Registry trains, persists, retrieves per-company models |
| `test_ensemble_deterministic` | Same input → same output (deterministic inference) |

## End-to-End Evaluation

```bash
# Full evaluation (feature extraction + company + anomaly + live)
python evaluate_pipeline.py

# Quick smoke test
python evaluate_pipeline.py --quick

# Company classifier overlap sweep only
python evaluate_pipeline.py --company

# Anomaly detection only
python evaluate_pipeline.py --fault

# Live simulation only
python evaluate_pipeline.py --live

# Transfer model evaluation (requires train --transfer first)
python evaluate_pipeline.py --transfer
```

### Evaluation Outputs

- **Feature extraction**: RMS separation ratios between normal, unbalanced, and bearing fault classes
- **Company classifier**: Accuracy + F1 across overlap sweep 0.0→1.0 (100% → ~68.5%)
- **Anomaly detection**: Accuracy, precision, recall, F1 score
- **Live simulation**: Total windows, anomaly rate, avg/p99 inference latency (ms)
- **Model card**: `evaluation_report.json` with full results and PASS/INCOMPLETE verdict

## CWRU Replay Validation

```bash
# Terminal 1: Start app
python app.py

# Terminal 2: Run fault sequence
python cwru_replay.py --sequence --mode http

# Expected: Normal→Inner Race→Ball Fault→Outer Race
# Anomaly score should spike on fault segments
```

## Running Tests via API

```bash
# Run pytest from the browser or curl
curl -X POST http://127.0.0.1:5050/api/evaluate/run-tests

# Returns JSON with passed/failed counts + output
```

## Test Data

- Tests use synthetic data generated on-the-fly (no hardware needed)
- `generate_synthetic_data.py` provides 4 vibration patterns: normal, unbalanced, bearing fault, misalignment
- `generate_companies_dataset(overlap=N)` generates configurable-separation company data
