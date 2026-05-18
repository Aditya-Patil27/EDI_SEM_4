"""
TensorFlow/Keras CNN training for vibration fault classification.
"""

import os
import yaml
import json
import numpy as np
import tensorflow as tf
from utils.gpu import configure_tf_gpu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models.detector import build_tf_cnn


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train():
    configure_tf_gpu()
    config = load_config()
    proc_dir = config["data"]["processed_dir"]

    X = np.load(os.path.join(proc_dir, "X_features.npy"))
    y_all = np.load(os.path.join(proc_dir, "y_labels.npy"))

    allowed = {c.lower() for c in config["data"]["classes"]}
    keep = [i for i, y in enumerate(y_all) if str(y).lower() in allowed]
    X, y = X[keep], y_all[keep]

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    os.makedirs("models", exist_ok=True)
    with open("models/label_encoder.json", "w") as f:
        json.dump({str(c): int(i) for i, c in enumerate(encoder.classes_)}, f)

    X = np.expand_dims(X, axis=-1)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    model = build_tf_cnn((X.shape[1], 1), num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=config["training"]["epochs"],
              batch_size=config["training"]["batch_size"],
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(patience=config["training"]["patience"], restore_best_weights=True),
                  tf.keras.callbacks.ModelCheckpoint("models/best_model.h5", monitor="val_accuracy", save_best_only=True),
              ])

    model.save("models/fault_classifier.h5")
    print(f"Saved models/fault_classifier.h5")
