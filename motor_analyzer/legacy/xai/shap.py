"""
SHAP explainability for vibration fault detection models.
Supports both PyTorch (CNNDetector) and TensorFlow/Keras models.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.gpu import configure_gpu


def explain_pytorch(model_path, X_test, class_names, output_dir="reports"):
    """SHAP DeepExplainer for PyTorch CNNDetector."""
    os.makedirs(output_dir, exist_ok=True)
    device = configure_gpu()

    from models.detector import CNNDetector
    import torch
    import shap

    model = CNNDetector(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    background = torch.tensor(X_test[:min(100, len(X_test))], dtype=torch.float32).to(device)
    test_samples = torch.tensor(X_test[:min(100, len(X_test))], dtype=torch.float32).to(device)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)

    shap_abs_mean = np.zeros(39)
    if isinstance(shap_values, list):
        for class_shap in shap_values:
            shap_abs_mean += np.mean(np.abs(class_shap), axis=(0, 1))
        shap_abs_mean /= len(shap_values)
    else:
        sv = np.asarray(shap_values)
        if len(sv.shape) == 4:
            shap_abs_mean = np.mean(np.abs(sv), axis=(0, 1, 2))
        else:
            shap_abs_mean = np.mean(np.abs(sv), axis=0).flatten()

    shap_abs_mean = shap_abs_mean.flatten()

    feature_names = (
        [f"Band_{i}" for i in range(10)]
        + ["RMS", "Kurtosis", "Crest"]
        + [f"MFCC_{i}" for i in range(13)]
        + [f"Delta_{i}" for i in range(13)]
    )

    top_feature_idx = int(np.argmax(shap_abs_mean))
    rejected = top_feature_idx == 0
    if rejected:
        print("CRITICAL: XAI rejected model — top attribution is Band_0 (potential 50Hz mains hum)")
    else:
        print(f"XAI Passed. Top feature: {feature_names[top_feature_idx]}")

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
        "feature_importances": {feature_names[i]: float(shap_abs_mean[i]) for i in range(len(feature_names))},
    }
    with open(os.path.join(output_dir, "shap_explanations.json"), "w") as f:
        json.dump(explanation, f, indent=4)
    print(f"Saved SHAP explanations to {output_dir}/")


def explain_tf(model_path, X_all, output_dir="reports"):
    """SHAP GradientExplainer for TensorFlow/Keras models."""
    os.makedirs(output_dir, exist_ok=True)
    from utils.gpu import configure_tf_gpu
    import tensorflow as tf
    import shap

    configure_tf_gpu()

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    X_tensor = np.expand_dims(X_all, axis=-1)
    model = tf.keras.models.load_model(model_path)

    bg_size = min(100, len(X_tensor))
    indices = np.random.choice(len(X_tensor), bg_size, replace=False)
    background = X_tensor[indices]

    test_samples = X_tensor[:min(10, len(X_tensor))]

    print("Computing SHAP values (TensorFlow)...")
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)

    test_samples_2d = np.squeeze(test_samples, axis=-1)

    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, test_samples_2d, show=False)
    else:
        sv = np.asarray(shap_values)
        if len(sv.shape) == 3:
            shap_values_list = [sv[:, :, i] for i in range(sv.shape[2])]
            shap.summary_plot(shap_values_list, test_samples_2d, show=False)
        else:
            shap.summary_plot(sv, test_samples_2d, show=False)

    plot_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {plot_path}")
