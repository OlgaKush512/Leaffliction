import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
import json
from sklearn.metrics import classification_report

training_set = tf.keras.utils.image_dataset_from_directory(
    'train_dataset',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    verbose=True,
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    'validation_dataset',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    verbose=True,
)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[256,256, 3]))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=[256,256, 3]))
model.add(Conv2D(filters=64, kernel_size=3,  activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', input_shape=[256,256, 3]))
model.add(Conv2D(filters=128, kernel_size=3,  activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', input_shape=[256,256, 3]))
model.add(Conv2D(filters=256, kernel_size=3,  activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', input_shape=[256,256, 3]))
model.add(Conv2D(filters=512, kernel_size=3,  activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Dropout(0.25)) # to avoid Overfitting

model.add(Flatten())
model.add(Dense(units=1500, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(units=8, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

training_history = model.fit(x=training_set, validation_data=validation_set, epochs=10)

# Model evaluation

train_loss, train_acc = model.evaluate(training_set)

print(train_loss, train_acc)

val_loss, val_acc = model.evaluate(validation_set)

print(val_loss, val_acc)

model.save("trained_model.keras")

training_history.history

#Recording history in json

with open("training_history.json", "w") as f:
    json.dump(training_history.history, f)

epochs = [i for i in range(1, 11)]
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy Result")
plt.legend()
plt.show()

# Some other metrics for model evaluation

class_name =validation_set.class_names
print(class_name)

test_set = validation_set = tf.keras.utils.image_dataset_from_directory(
    'validation_dataset',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    verbose=True,
)

y_pred = model.predict(test_set)


predicted_category = tf.argmax(y_pred, axis=1)

true_categories = tf.concat([y for x,y in test_set], axis=0)

Y_true = tf.argmax(true_categories, axis=1)

classification_report(Y_true, predicted_category, target_names=class_name)