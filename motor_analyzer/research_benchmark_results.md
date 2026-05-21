# MotorSense - Research Benchmark Results

## 1. Overlap Sweep - Accuracy vs RPM Overlap

| RPM Overlap | Accuracy | F1 Score |
|------------|----------|----------|
| 0.0 | 1.0000+/-0.0000 | 1.0000+/-0.0000 |
| 0.2 | 1.0000+/-0.0000 | 1.0000+/-0.0000 |
| 0.4 | 0.9806+/-0.0039 | 0.9805+/-0.0040 |
| 0.6 | 0.9333+/-0.0136 | 0.9330+/-0.0137 |
| 0.8 | 0.8083+/-0.0180 | 0.8070+/-0.0177 |
| 1.0 | 0.4639+/-0.0502 | 0.4446+/-0.0456 |

## 2. Baseline Comparisons

| Method | overlap=0.0 | overlap=0.4 | overlap=1.0 |
|--------|------------|------------|------------|
| RMS Threshold (2s) | 0.2583 | 0.2667 | 0.2583 |
| RandomForest Only | 1.0000 | 0.9667 | 0.4417 |
| OC-SVM Only | 0.7667 | 0.7500 | 0.1417 |
| Full Ensemble (Ours) | 1.0000 | 0.9833 | 0.4333 |

## 3. Feature Ablation

Base accuracy (all 28 features): 0.9778

| Rank | Feature | Importance (d-acc) |
|------|---------|------------------|
| 1 | Skewness | 0.0994 |
| 2 | Spectral Band 8 | 0.0239 |
| 3 | Spectral Band 7 | 0.0044 |
| 4 | Dominant Freq | 0.0039 |
| 5 | Spectral Band 1 | 0.0033 |
| 6 | High Energy | 0.0033 |
| 7 | Zero-Crossing Rate | 0.0033 |
| 8 | Spectral Band 6 | 0.0028 |
| 9 | Mid Energy | 0.0022 |
| 10 | Shape Factor | 0.0006 |

**Group Importance:**

- Time Domain (0-7): 0.1033
- Spectral Bands (8-17): 0.0311
- Frequency Stats (18-27): 0.0100

## 4. Data Scaling

| Samples/Company | Training Size | Accuracy |
|----------------|--------------|----------|
| 10 | 70 | 1.0000 |
| 20 | 140 | 0.9667 |
| 40 | 280 | 0.9667 |
| 60 | 420 | 0.9778 |
| 80 | 560 | 0.9667 |

## 5. Transfer Learning (CWRU to JNU)

- Zero-shot accuracy: 0.0000
- Fine-tuned (+10% target): 1.0000
- Delta: +1.0000