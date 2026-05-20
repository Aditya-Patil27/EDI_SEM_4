"""
GPU configuration for both PyTorch and TensorFlow.
"""

import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False


def load_gpu_config(path="config.yaml"):
    try:
        with open(path) as f:
            return yaml.safe_load(f).get("gpu", {})
    except Exception:
        return {}


def configure_gpu():
    """Configure PyTorch GPU and return device string."""
    if not TORCH_OK:
        logging.warning("PyTorch unavailable. Running on CPU.")
        return "cpu"
    gpu_cfg = load_gpu_config()
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        name = torch.cuda.get_device_name(gpu_id)
        mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        logging.info(f"GPU: {name} ({mem:.1f} GB)")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        if gpu_cfg.get("mixed_precision", False):
            torch.set_float32_matmul_precision("high")
        return "cuda"
    else:
        logging.warning("No GPU detected. Running on CPU.")
        return "cpu"


def configure_tf_gpu():
    """Configure TensorFlow GPU with memory growth."""
    try:
        import tensorflow as tf
    except Exception:
        logging.warning("TensorFlow unavailable.")
        return
    gpu_cfg = load_gpu_config()
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logging.info(f"TensorFlow GPUs: {[g.name for g in gpus]}")
        if gpu_cfg.get("memory_growth", False):
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        if gpu_cfg.get("mixed_precision", False):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        logging.warning("TensorFlow: No GPU detected.")
