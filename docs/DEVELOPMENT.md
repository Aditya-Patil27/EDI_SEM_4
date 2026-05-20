# Development Guide

## Project Structure

```
motor_analyzer/
├── app.py                    # Flask server (779 lines)
├── feature_pipeline.py       # 28-dim feature extractor (115 lines)
├── ml_models.py              # ML models (562 lines)
├── data_ingestion.py         # Dataset ingestion (355 lines)
├── train_pipeline.py         # Training pipeline (213 lines)
├── evaluate_pipeline.py      # Evaluation pipeline (389 lines)
├── cwru_replay.py            # CWRU dataset replay (263 lines)
├── generate_synthetic_data.py # Synthetic data generator (178 lines)
├── config.yaml               # Central configuration (69 lines)
├── ESP32_DATA_GUIDE.md        # Custom data recording guide
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── company_classifier.pth           # Trained CompanyClassifier
│   ├── company_classifier_transfer.pth  # Trained TransferLearningAdapter
│   └── companies/
│       ├── manifest.json
│       ├── CWRU.pkl
│       ├── JNU.pkl
│       └── Synthetic.pkl
│
├── data/
│   ├── raw/
│   │   ├── cwru/             # CWRU .mat files (GitHub mirror structure)
│   │   └── jnu/              # JNU CSV files
│   └── processed/
│       ├── X_all.npy, y_all.npy, company_labels.npy
│       └── metadata.json
│
├── tests/
│   ├── test_feature_pipeline.py  # 11 tests
│   └── test_ml_models.py         # 7 tests
│
├── static/
│   ├── css/style.css
│   └── js/main.js            # Frontend logic (765 lines)
│
└── templates/
    └── index.html            # Single-page app (319 lines)
```

## Key Design Decisions

### Why 28 Features?
8 time-domain + 10 spectral bands + 10 frequency-domain = unified extractor that merges the original `motor_analyzer` approach with the `ml repo`'s spectral band approach.

### Why sklearn Instead of PyTorch?
PyTorch's `shm.dll` fails on many Windows systems. All `torch` code paths have sklearn fallbacks:
- `CNNDetector` → `RandomForestClassifier` via `CompanyClassifier`
- `TransferLearningAdapter._init_torch()` → `_init_sklearn()`

### Per-Company Model Architecture
Company identification happens on the first ~512 raw samples (~5s), then a dedicated anomaly model is loaded for that company. This allows different vibration profiles for different machine types.

### Cross-Dataset RMS Normalization
When ingesting multiple datasets, the RMS (feature[0]) is scaled to the mean of all datasets to prevent the model from trivially separating by amplitude alone.

## Adding a New Dataset

1. Create an adapter function in `data_ingestion.py` (e.g., `_process_mydata()`)
2. Add the dataset name to `config.yaml` `companies.classes`
3. Register in `ingest_all_datasets()`
4. Normalize RMS across datasets
5. Retrain: `python train_pipeline.py`

## Adding a New Feature

1. Add to `extract_features()` in `feature_pipeline.py`
2. Update `dimension` in `config.yaml`
3. Update tests in `test_feature_pipeline.py`
4. Retrain all models

## CWRU File Discovery

The CWRU GitHub mirror has a variable naming pattern: `X{number}_DE_time`. The code walks all subdirectories recursively and matches `_DE_time` keys in `.mat` files. See `cwru_replay.py:find_mat_file()` for file number matching via regex.

## Company Classifier Caveat

On public datasets (CWRU, JNU, Synthetic), the company classifier achieves 100% accuracy because these are from different research labs with distinct sensor characteristics. This is NOT representative of real-world performance where vibration profiles from different factories would overlap significantly.

For realistic evaluation:
```bash
python evaluate_pipeline.py --company
```
This sweeps overlap parameter 0.0→1.0 across synthetic companies with controlled RPM overlap.

## Transfer Learning

The `TransferLearningAdapter` bridges a 39-dim audio CNN (from [github.com/Aditya-Patil27/ml](https://github.com/Aditya-Patil27/ml)) to 28-dim vibration features:

- **Torch path**: `Linear(28,39)` expansion layer → frozen pretrained CNN
- **Sklearn fallback**: trains `RandomForestClassifier` directly (no transfer, but functional)

Both paths train on the same company labels and produce compatible `.pth` files.

## Commit History

1. Initial ML pipeline (CompanyClassifier, EnsembleAnomalyModel, feature extractor)
2. Transfer learning adapter (28→39 expansion, sklearn fallback)
3. CWRU replay bridge (HTTP injection, fault sequence mode)
