"""
train.py
--------
Trains the CNN to classify breast tumor images as Normal or Malignant.

Folder structure:

data/
└── train/
    ├── normal/
    └── malignant/

Usage
-----
python -m src.train --data data/train --epochs 10 --batch-size 32
"""

import argparse
import json
from pathlib import Path

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data_loader import IMAGE_SIZE, load_dataset
from src.model import build_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main(data_dir: str, epochs: int, batch_size: int, image_size: int):

    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("Loading dataset...")
    print("=" * 50)

    X, y = load_dataset(data_dir, image_size=image_size)

    print(f"Total images : {len(X)}")
    print(f"Normal       : {(y == 0).sum()}")
    print(f"Malignant    : {(y == 1).sum()}")

    # Train / Validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nBalancing classes...")

    X_train_flat = X_train.reshape(len(X_train), -1)

    ros = RandomOverSampler(random_state=42)

    X_train_flat, y_train = ros.fit_resample(
        X_train_flat,
        y_train,
    )

    X_train = X_train_flat.reshape(
        -1,
        image_size,
        image_size,
        3,
    )

    print(f"Balanced training samples : {len(X_train)}")

    print("\nBuilding model...")

    model = build_model(
        input_shape=(image_size, image_size, 3)
    )

    print("\nTraining...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    print("\nEvaluating model...")

    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Malignant"],
        output_dict=True,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    ).tolist()

    print(classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Malignant"],
    ))

    print("Confusion Matrix")
    print(cm)

    model.save(MODELS_DIR / "cnn_model.keras")

    metrics = {
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_accuracy": float(history.history["val_accuracy"][-1]),
        "classification_report": report,
        "confusion_matrix": cm,
    }

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nModel saved to:")
    print(MODELS_DIR / "cnn_model.keras")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Breast Tumor Detection Training"
    )

    parser.add_argument(
        "--data",
        default="data/train",
        help="Training dataset folder",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
    )

    args = parser.parse_args()

    main(
        args.data,
        args.epochs,
        args.batch_size,
        args.image_size,
    )