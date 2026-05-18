import os
import shap
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.train import CNNDetector
from src.preprocessing import load_config

def audit_model(model_path, X_test, output_dir="xai/"):
    os.makedirs(output_dir, exist_ok=True)
    config = load_config()
    classes = config['data']['classes']
    
    from src.gpu_setup import configure_gpu
    device = configure_gpu()
    
    model = CNNDetector(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    background = torch.tensor(X_test[:min(100, len(X_test))], dtype=torch.float32).to(device)
    test_samples = torch.tensor(X_test[:min(100, len(X_test))], dtype=torch.float32).to(device)
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)
    
    # Calculate mean absolute SHAP values for each feature
    shap_abs_mean = np.zeros(39)
    # Check if shap_values is a list (multi-class) or an array
    if isinstance(shap_values, list):
        for class_shap in shap_values:
            shap_abs_mean += np.mean(np.abs(class_shap), axis=(0, 1)) # (samples, 1, 39)
        shap_abs_mean /= len(shap_values)
    else:
        # shap_values shape: (samples, num_classes, 1, 39) depending on PyTorch output
        # Let's flatten to get magnitude per feature dimension
        sv = np.asarray(shap_values)
        if len(sv.shape) == 4:
            shap_abs_mean = np.mean(np.abs(sv), axis=(0, 1, 2))
        else:
            shap_abs_mean = np.mean(np.abs(sv), axis=0).flatten()
    
    shap_abs_mean = shap_abs_mean.flatten()
    
    feature_names = [f"Band_{i}" for i in range(10)] + ["RMS", "Kurtosis", "Crest"] + [f"MFCC_{i}" for i in range(13)] + [f"Delta_{i}" for i in range(13)]
    
    top_feature_idx = np.argmax(shap_abs_mean)
    
    rejected = False
    if top_feature_idx == 0:
        print("CRITICAL WARNING: XAI Audit rejected model! Top attribution relies on lowest frequency band (potential 50Hz mains hum).")
        rejected = True
    else:
        print(f"XAI Audit Passed! Top attribution is feature: {feature_names[top_feature_idx]}")
        
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, shap_abs_mean)
    plt.xticks(rotation=90)
    plt.title("SHAP Feature Attributions (Global Mean Absolute)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"))
    plt.close()
    
    explanation = {
        "top_feature": feature_names[top_feature_idx],
        "top_feature_importance": float(shap_abs_mean[top_feature_idx]),
        "rejected": rejected,
        "feature_importances": {feature_names[i]: float(shap_abs_mean[i]) for i in range(39)}
    }
    
    with open(os.path.join(output_dir, "shap_explanations.json"), "w") as f:
        json.dump(explanation, f, indent=4)
        
if __name__ == "__main__":
    if os.path.exists('model_best.pth') and os.path.exists('data/processed/X_features.npy'):
        X = np.load('data/processed/X_features.npy')
        X = np.expand_dims(X, axis=1) # PyTorch format
        audit_model('model_best.pth', X)
