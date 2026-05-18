import os
import yaml
import numpy as np
import torch
from src.train import CNNDetector
import time

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def quantize_model(model_path, X_val, output_path="model_quantized.pth"):
    """
    Convert PyTorch model to dynamically quantized model.
    Quantization runs on CPU (required by PyTorch quantization backend).
    """
    config = load_config()
    classes = config['data']['classes']
    
    # Load model on CPU — PyTorch quantization requires CPU
    model = CNNDetector(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # PyTorch Dynamic Quantization for Linear layers (Conv1d dynamic isn't natively supported 
    # as easily as static, but static is more complex to set up. We'll do dynamic for Linear)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Quantized model saved to {output_path}")

def profile_latency(model_path, input_shape=(1, 1, 39)):
    """
    Simulate inference latency check
    """
    config = load_config()
    classes = config['data']['classes']
    
    model = CNNDetector(num_classes=len(classes))
    # Note: If loading a dynamically quantized model state dict, you'd quantize the skeleton first.
    # For profiling here we just load the weights.
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception:
        # If it's the quantized weights, we must prepare the model first
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
    model.eval()
    
    mock_data = torch.randn(*input_shape)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(mock_data)
            
    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(mock_data)
    end = time.time()
    
    avg_latency_ms = ((end - start) / 100) * 1000
    print(f"Average CPU inference latency: {avg_latency_ms:.2f} ms")
    
    if avg_latency_ms > 50.0:
        print("WARNING: Inference latency exceeds 50 ms budget!")
    else:
        print("Latency is within budget.")
        
    return avg_latency_ms

if __name__ == "__main__":
    if os.path.exists('model_best.pth'):
        proc_dir = load_config()['data']['processed_dir']
        X = np.load(os.path.join(proc_dir, "X_features.npy"))
        X = np.expand_dims(X, axis=1) # PyTorch format
        quantize_model('model_best.pth', X)
        profile_latency('model_quantized.pth')
