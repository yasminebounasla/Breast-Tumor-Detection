"""
data_loader.py
--------------
Loads breast tumor images from disk and prepares them for training.

Expected folder layout:

data/
└── train/
    ├── normal/
    └── malignant/

Images are resized to IMAGE_SIZE x IMAGE_SIZE and normalized to [0, 1].

Labels:
    normal -> 0
    malignant -> 1
"""

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 224

CLASS_MAP = {
    "normal": 0,
    "malignant": 1,
}

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


def load_image(path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Load a single image, resize it and normalize pixels.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img


def load_dataset(data_dir: str, image_size: int = IMAGE_SIZE):
    """
    Loads every image inside:

        data_dir/
            normal/
            malignant/

    Returns
    -------
    X : numpy.ndarray
        Images

    y : numpy.ndarray
        Labels
    """

    data_dir = Path(data_dir)

    X = []
    y = []

    for class_name, label in CLASS_MAP.items():

        class_dir = data_dir / class_name

        if not class_dir.exists():
            print(f"Skipping missing folder: {class_dir}")
            continue

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                X.append(load_image(image_path, image_size))
                y.append(label)

            except Exception as e:
                print(f"Could not read {image_path}: {e}")

    if len(X) == 0:
        raise FileNotFoundError(
            f"No images found inside:\n{data_dir}/normal\n{data_dir}/malignant"
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
    )