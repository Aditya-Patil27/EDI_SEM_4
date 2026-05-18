# TinyML Fault Classifier Architecture

A comprehensive machine learning pipeline for robust, highly accurate mechanical fault classification on extreme edge devices. Native to PyTorch, this project employs highly parameterized 1D-CNNs combined with dynamic quantization, advanced DSP, and on-device temporal moderation to deliver real-time, explainable diagnostic insights.

## Core Features
*   **Adapter Pattern Data Ingestion:** Safely unifies varying dataset formats (WAV, CSV, Raw Binary waveforms) like MAFAULDA and NASA IMS Bearing into a standard schema.
*   **Advanced DSP & Preprocessing:** Operates a rigid feature engineering queue including 8kHz resamplers, 4th-order Butterworth Bandpass filters, and extraction of 39 specific time/spectral/cepstral mechanical features per frame.
*   **Lightweight 1D-CNN:** PyTorch-based neural footprint requiring < 12k parameters and supporting GPU-augmented local training schedules (with SMOTE and Cosine Annealing).
*   **Int8 Dynamic Quantization:** Crushes memory and compute overheads for deployment on standard and low-power microcontrollers (<50ms execution runtime target).
*   **Temporal On-Device Moderation:** Operates a tandem Gaussian Mixture Model (GMM) alongside CNN predictions with a strict consecutive-window consensus metric to sharply mitigate False Positives (FPR).
*   **Automated SHAP XAI Audits:** Deep explainer safety checks to ensure structural predictions rely strictly on mechanical frequencies and not environmental artifacts (like 50/60Hz line leakage).

## Installation

Ensure you have Python 3.8+ installed. Install the base scientific computing requirements provided:

```bash
pip install -r requirements.txt
```

*Note: Ensure you install a version of `torch` appropriate for your CPU/CUDA setup.*

## Usage & Execution

The primary entry point to coordinate system operations is `pipeline.py`. You can instruct the module to run all sequences fully end-to-end, or handle individual components.

To run the complete pipeline sequentially:
```bash
python pipeline.py --all
```

To run individual stages explicitly:
*   **Stage 1: Ingest Data**: `python pipeline.py --ingest`
*   **Stage 2: Preprocess & DSP**: `python pipeline.py --preprocess`
*   **Stage 3: Train Model**: `python pipeline.py --train`
*   **Stage 4: Quantize Target**: `python pipeline.py --quantize`
*   **Stage 5: Evaluation & Report**: `python pipeline.py --evaluate`
*   **Stage 6: XAI Deep Audit**: `python pipeline.py --xai`

## Directory Structure

*   `/src/` - System logic scripts including ingestion, processing, training algorithms, quantization hooks, and XAI bounds.
*   `config.yaml` - Hardcoded constants for feature extraction variables, path pointers, and pipeline state overrides. 
*   `pipeline.py` - Core entry executable for orchestration.
*   `/data/` - Base working environment capturing localized `raw` streams and `processed` training-ready frames.
*   `requirements.txt` - Pip packages requirements map.
