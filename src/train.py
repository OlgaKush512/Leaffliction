import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
import json
import zipfile


def validate_dataset_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"The path {path} is not a directory.")
    subdirs = [d for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        raise ValueError(
            f"No subdirectories found in {path}. "
            "It should contain class subdirectories with images."
        )
    for subdir in subdirs:
        images = [
            f for f in os.listdir(os.path.join(path, subdir))
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ]
        if not images:
            raise ValueError(
                f"No images found in the subdirectory {subdir} of {path}."
            )


def load_dataset(path, batch_size=32, img_size=(256, 256)):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        follow_links=False,
    )


def build_model():
    model = Sequential()

    model.add(
        Conv2D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu',
            input_shape=[256, 256, 3]
        )
    )
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(
        Conv2D(
            filters=64,
            kernel_size=3,
            padding='same',
            activation='relu'
        )
    )
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(
        Conv2D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu'
        )
    )
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(
        Conv2D(
            filters=256,
            kernel_size=3,
            padding='same',
            activation='relu'
        )
    )
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(
        Conv2D(
            filters=512,
            kernel_size=3,
            padding='same',
            activation='relu'
        )
    )
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Dropout(0.25))  # To avoid overfitting

    model.add(Flatten())
    model.add(Dense(units=1500, activation='relu'))

    model.add(Dropout(0.4))
    model.add(Dense(units=8, activation='softmax'))  # 8 units for 8 classes

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def save_to_zip(model, history, zip_filename="training_results.zip"):
    model.save("trained_model.keras")

    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write("trained_model.keras", arcname="trained_model.keras")
        zipf.write("training_history.json", arcname="training_history.json")

    os.remove("trained_model.keras")
    os.remove("training_history.json")


def main(train_path, val_path):
    validate_dataset_path(train_path)
    validate_dataset_path(val_path)

    training_set = load_dataset(train_path)
    validation_set = load_dataset(val_path)

    model = build_model()
    model.summary()

    history = model.fit(
        x=training_set,
        validation_data=validation_set,
        epochs=10
    )

    save_to_zip(model, history)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python train.py "
            "<train_dataset_path> <validation_dataset_path>"
        )
        sys.exit(1)

    train_dataset_path = sys.argv[1]
    validation_dataset_path = sys.argv[2]

    main(train_dataset_path, validation_dataset_path)
