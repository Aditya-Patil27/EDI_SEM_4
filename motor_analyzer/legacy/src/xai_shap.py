import os
import yaml
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from src.gpu_setup import configure_tf_gpu

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_explanations():
    configure_tf_gpu()
    config = load_config()
    model_dir = "models"
    model_path = os.path.join(model_dir, "fault_classifier.h5")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    proc_dir = config['data']['processed_dir']
    X_path = os.path.join(proc_dir, "X_features.npy")
    
    if not os.path.exists(X_path):
        print(f"Data not found at {X_path}")
        return
        
    X_all = np.load(X_path)
    X_tensor = np.expand_dims(X_all, axis=-1)
    model = tf.keras.models.load_model(model_path)
    
    bg_size = min(100, len(X_tensor))
    indices = np.random.choice(len(X_tensor), bg_size, replace=False)
    background = X_tensor[indices]
    
    explain_size = min(10, len(X_tensor))
    test_samples = X_tensor[:explain_size]
    
    print("Computing SHAP values...")
    # Using GradientExplainer for deep models
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)
    
    test_samples_2d = np.squeeze(test_samples, axis=-1)
    shap_dir = "reports"
    os.makedirs(shap_dir, exist_ok=True)
    
    plt.figure()
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, test_samples_2d, show=False)
    else:
        sv = np.asarray(shap_values)
        if len(sv.shape) == 3:
            shap_values_list = [sv[:,:,i] for i in range(sv.shape[2])]
            shap.summary_plot(shap_values_list, test_samples_2d, show=False)
        else:
            shap.summary_plot(sv, test_samples_2d, show=False)
            
    plot_path = os.path.join(shap_dir, "shap_summary.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SHAP summary plot to {plot_path}")

if __name__ == "__main__":
    generate_explanations()
