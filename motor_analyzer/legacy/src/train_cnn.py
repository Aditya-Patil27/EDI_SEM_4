import os
import yaml
import json
import numpy as np
import tensorflow as tf
from src.gpu_setup import configure_tf_gpu
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_cnn(input_shape, num_classes):
    """
    Builds a 1D CNN with < 12,000 parameters.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # 39 features, treating as a 1D sequence of length 39
        layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model

def train():
    configure_tf_gpu()
    config = load_config()
    proc_dir = config['data']['processed_dir']
    
    X_path = os.path.join(proc_dir, "X_features.npy")
    y_path = os.path.join(proc_dir, "y_labels.npy")
    
    if not os.path.exists(X_path):
        print(f"Processed data not found at {proc_dir}")
        return
        
    X_all = np.load(X_path)
    y_all = np.load(y_path)
    
    # Filter config classes
    allowed_classes = [c.lower() for c in config['data']['classes']]
    
    # Keep only data belonging to allowed classes
    valid_indices = [i for i, y in enumerate(y_all) if str(y).lower() in allowed_classes]
    X = X_all[valid_indices]
    y_valid = y_all[valid_indices]
    
    # Preprocess labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_valid)
    num_classes = len(encoder.classes_)
    
    # Save encoder mapping
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    encoder_path = os.path.join(model_dir, "label_encoder.json")
    with open(encoder_path, 'w') as f:
        json.dump({str(cls): int(idx) for idx, cls in enumerate(encoder.classes_)}, f)
        
    # Reshape X for 1D CNN: (batch, steps, channels)
    X = np.expand_dims(X, axis=-1)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=config['project']['seed'], stratify=y_encoded
    )
    
    model = build_cnn((X.shape[1], 1), num_classes)
    model.summary()
    
    param_count = model.count_params()
    if param_count > config['training']['cnn_params']:
        print(f"WARNING: Model has {param_count} parameters, which exceeds the limit of {config['training']['cnn_params']}!")
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=config['training']['patience'], 
        restore_best_weights=True
    )
    
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(model_dir, "best_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    model_path = os.path.join(model_dir, "fault_classifier.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train()
