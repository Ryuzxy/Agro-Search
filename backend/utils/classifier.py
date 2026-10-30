"""CNN classifier for horticultural leaf disease (MobileNetV2 transfer learning)."""

import json
import os
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image as keras_image  # type: ignore

LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "disease_labels.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "plant_disease_model.h5")

IMG_SIZE = (224, 224)


class PlantClassifier:
    def __init__(self, model_path: str = MODEL_PATH, labels_path: str = LABELS_PATH):
        self.model = None
        self.model_path = model_path
        self.class_names: list[str] = []
        self.label_to_query: dict[str, str] = {}

        # Load label mapping
        with open(labels_path, "r", encoding="utf-8") as f:
            label_groups = json.load(f)
        for _crop, mapping in label_groups.items():
            for eng_label, indo_query in mapping.items():
                self.label_to_query[eng_label] = indo_query

        # Sorted class names = alphabetical order matching model output
        self.class_names = sorted(self.label_to_query.keys())

        # Load model if exists
        if os.path.exists(model_path):
            self.model = load_model(model_path)

    def predict(self, img_path: str) -> dict:
        """Classify a leaf image. Returns class label, confidence, and Indonesian query."""
        if self.model is None:
            raise RuntimeError(
                f"Model not found at {self.model_path}. "
                "Train with: python train_model.py"
            )

        img = keras_image.load_img(img_path, target_size=IMG_SIZE)
        arr = keras_image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = self.model.predict(arr, verbose=0)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        class_name = self.class_names[idx]
        query = self.label_to_query.get(class_name, class_name)

        return {
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "query": query,
            "is_healthy": "healthy" in class_name.lower(),
        }
