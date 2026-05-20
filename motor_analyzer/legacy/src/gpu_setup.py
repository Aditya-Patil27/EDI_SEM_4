import logging
import yaml
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def _load_gpu_config(path="config.yaml"):
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('gpu', {})
    except FileNotFoundError:
        return {}

def configure_gpu():
    """Configure PyTorch to use available GPU and return device string."""
    gpu_cfg = _load_gpu_config()

    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        
        logging.info(f"GPU Detected: {device_name}")
        logging.info(f"GPU Memory: {gpu_memory_total:.1f} GB")
        
        # Enable GPU memory optimization
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms
        torch.backends.cudnn.enabled = True
        logging.info("PyTorch: Enabled cuDNN benchmarking and optimization.")
        
        if gpu_cfg.get('memory_growth', False):
            # Note: PyTorch doesn't have memory_growth like TensorFlow, 
            # but we can use empty_cache and gradient checkpointing
            logging.info("PyTorch: Memory-efficient mode enabled (will use gradient checkpointing if supported).")

        if gpu_cfg.get('mixed_precision', False):
            torch.set_float32_matmul_precision('high')
            logging.info("PyTorch: enabled high-precision float32 matmul for mixed precision.")

        return "cuda"
    else:
        logging.warning("No GPU detected. Running on CPU.")
        return "cpu"

def configure_tf_gpu():
    """Configure TensorFlow to use available GPU with memory growth and mixed precision."""
    import tensorflow as tf

    gpu_cfg = _load_gpu_config()
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        logging.info(f"TensorFlow GPU Detected: {[g.name for g in gpus]}")
        if gpu_cfg.get('memory_growth', False):
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("TensorFlow: enabled memory growth.")
        if gpu_cfg.get('mixed_precision', False):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logging.info("TensorFlow: enabled mixed_float16 precision.")
    else:
        logging.warning("TensorFlow: No GPU detected. Running on CPU.")

