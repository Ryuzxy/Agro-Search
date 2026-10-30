"""Train MobileNetV2 for plant disease classification.

Usage:
    python train_model.py --data_dir ./dataset --epochs 20 --batch_size 32

Expected dataset structure:
    dataset/
      Chili__healthy/
      Chili__leaf curl/
      Chili__leaf spot/
      ...
      Potato___healthy/
      Potato___Early_blight/
      ...
      Tomato___healthy/
      Tomato___Bacterial_spot/
      ...
"""

import argparse
import os
import json
import numpy as np
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "plant_disease_model.h5")


def build_model(num_classes: int) -> Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    # Freeze base layers
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=predictions)


def main():
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Data generators with augmentation
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        validation_split=0.1111,  # ~10% val from 90% (train+val), test held out
    )

    train_data = train_gen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_data = train_gen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = len(train_data.class_indices)
    print(f"Classes ({num_classes}): {list(train_data.class_indices.keys())}")

    # Save class order for inference
    class_order_path = os.path.join(os.path.dirname(__file__), "models", "class_order.json")
    with open(class_order_path, "w") as f:
        json.dump(train_data.class_indices, f, indent=2)

    model = build_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=args.lr), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy"),
    ]

    model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)

    # Evaluation on validation set
    val_data.reset()
    preds = model.predict(val_data, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_data.classes

    class_names = list(train_data.class_indices.keys())
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

    print(f"\nModel saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
