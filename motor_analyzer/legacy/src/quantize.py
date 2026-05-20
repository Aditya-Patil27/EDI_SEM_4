import os
import yaml
import numpy as np
import tensorflow as tf

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def representative_data_gen():
    config = load_config()
    proc_dir = config['data']['processed_dir']
    X_path = os.path.join(proc_dir, "X_features.npy")
    samples_count = config['quantization']['representative_samples']

    if not os.path.exists(X_path):
        # Fallback to dummy data
        for _ in range(samples_count):
            yield [np.random.normal(size=(1, 39, 1)).astype(np.float32)]
        return

    X_all = np.load(X_path)
    if len(X_all) == 0:
        return

    # Pick random samples
    indices = np.random.choice(len(X_all), size=min(samples_count, len(X_all)), replace=False)
    for i in indices:
        x_sample = np.expand_dims(X_all[i], axis=-1) # (39, 1)
        x_sample = np.expand_dims(x_sample, axis=0) # (1, 39, 1)
        yield [x_sample.astype(np.float32)]

def quantize_model():
    config = load_config()
    model_dir = "models"
    model_path = os.path.join(model_dir, "fault_classifier.h5")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if config['quantization']['type'] == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    quant_model_path = os.path.join(model_dir, "fault_classifier_quant.tflite")
    
    with open(quant_model_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    print(f"Saved quantized model to {quant_model_path}")
    print(f"Quantized Model Size: {os.path.getsize(quant_model_path) / 1024:.2f} KB")

if __name__ == "__main__":
    quantize_model()
