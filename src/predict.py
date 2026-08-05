"""
predict.py
----------
Loads the trained CNN model and predicts whether a breast tumor
image is Normal or Malignant.
"""

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from src.data_loader import IMAGE_SIZE, load_image

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_model = None


def _load_model():
    global _model

    if _model is None:
        model_path = MODELS_DIR / "cnn_model.keras"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Train the model first using:\n"
                "python -m src.train"
            )

        _model = load_model(model_path)

    return _model


def predict_image(image_path_or_array, image_size: int = IMAGE_SIZE):
    """
    Predict a single image.

    Parameters
    ----------
    image_path_or_array :
        Image path or already loaded numpy array.

    Returns
    -------
    label : str
        Normal or Malignant

    confidence : float
    """

    model = _load_model()

    if isinstance(image_path_or_array, (str, Path)):
        image = load_image(image_path_or_array, image_size)
    else:
        image = image_path_or_array

    image = np.expand_dims(image, axis=0)

    probability = float(
        model.predict(image, verbose=0)[0][0]
    )

    if probability >= 0.5:
        return "Malignant", probability

    return "Normal", 1 - probability