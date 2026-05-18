"""
Backward-compatible re-exports from organized module structure.
Old imports like `from src.gpu_setup import configure_gpu` continue to work.
"""

import sys, os
_legacy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _legacy_dir not in sys.path:
    sys.path.insert(0, _legacy_dir)

# GPU setup
from utils.gpu import configure_gpu, configure_tf_gpu
import utils.gpu as gpu_setup

# Train
from models.detector import CNNDetector, build_tf_cnn
from models.trainer import main as train_main, augment_data, train_gmm, FaultDataset, group_split
import models.trainer as train

# Data ingestion
from ingestion.adapters import (
    ingest_all, compute_checksum,
    process_cwru, process_jnu, process_synthetic,
    process_mafaulda, process_nasa_ims, process_unbalance, process_generic,
    DATASET_ADAPTERS,
)
import ingestion.adapters as data_ingestion

# Preprocessing
from features.legacy import load_config, preprocess_audio, extract_features_60, sliding_window, run_pipeline
import features.legacy as preprocessing

# Quantization
from utils.quantization import quantize_pytorch, quantize_tflite, profile_latency
import utils.quantization as quantization

# XAI
from xai.shap import explain_pytorch, explain_tf
import xai.shap as xai_audit
