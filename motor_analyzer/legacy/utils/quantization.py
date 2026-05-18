"""
Model quantization and latency profiling.
Supports PyTorch dynamic quantization and TensorFlow Lite int8.
"""

import os
import time
import yaml
import numpy as np
import torch


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def quantize_pytorch(model_path, output_path="model_quantized.pth"):
    """PyTorch dynamic quantization (Linear layers → qint8)."""
    config = load_config()
    classes = config.get("data", {}).get("classes", [])
    from models.detector import CNNDetector

    model = CNNDetector(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), output_path)
    print(f"Quantized PyTorch model saved to {output_path}")


def profile_latency(model_path, input_shape=(1, 1, 39)):
    """Measure CPU inference latency over 100 runs."""
    config = load_config()
    classes = config.get("data", {}).get("classes", [])
    from models.detector import CNNDetector

    model = CNNDetector(num_classes=len(classes))
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except Exception:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    mock = torch.randn(*input_shape)
    with torch.no_grad():
        for _ in range(10):
            model(mock)
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(mock)
    avg_ms = ((time.time() - start) / 100) * 1000
    print(f"Avg CPU inference: {avg_ms:.2f} ms")
    if avg_ms > 50:
        print("WARNING: Latency exceeds 50ms budget!")
    return avg_ms


def quantize_tflite(model_path="models/fault_classifier.h5", output_path="models/fault_classifier_quant.tflite"):
    """TensorFlow Lite int8 quantization with representative dataset."""
    import tensorflow as tf

    config = load_config()
    proc_dir = config.get("data", {}).get("processed_dir", "data/processed")
    samples_count = config.get("quantization", {}).get("representative_samples", 100)

    def rep_data():
        X_path = os.path.join(proc_dir, "X_features.npy")
        if os.path.exists(X_path):
            X_all = np.load(X_path)
            idx = np.random.choice(len(X_all), size=min(samples_count, len(X_all)), replace=False)
            for i in idx:
                x = np.expand_dims(X_all[i], axis=-1)[np.newaxis, :]
                yield [x.astype(np.float32)]
        else:
            for _ in range(samples_count):
                yield [np.random.normal(size=(1, 39, 1)).astype(np.float32)]

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if config.get("quantization", {}).get("type") == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite)
    print(f"Quantized TFLite saved to {output_path} ({os.path.getsize(output_path) / 1024:.2f} KB)")
