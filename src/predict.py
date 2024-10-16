import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def load_image(img_path, img_size):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(model, img_array, class_names):
    """Predict the class of the disease for the input image."""
    predictions = model.predict(img_array, batch_size=1)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]
    return predicted_label

def plot_metrics(accuracies, losses):
    "Display accuracy and loss on a graph."
    epochs = list(range(1, len(accuracies) + 1))
    plt.plot(epochs, accuracies, label="Accuracy", marker='o', color='blue')
    plt.plot(epochs, losses, label="Loss", marker='o', color='red')
    plt.title('Model Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(img_path):
    """Main function to load the model, predict the class, and display the result."""
    if not os.path.exists(img_path):
        print(f"Error: The file {img_path} does not exist.")
        return

    model = tf.keras.models.load_model("trained_model.keras")

    img_size = (256, 256)

    if os.path.isdir(img_path):
        test_set = tf.keras.utils.image_dataset_from_directory(
            img_path,
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
            crop_to_aspect_ratio=False
        )

        class_names = test_set.class_names

        test_images, test_labels = [], []
        for img, label in test_set.as_numpy_iterator():
            test_images.extend(img)
            test_labels.extend(label)
            if len(test_images) >= 200:
                break

        test_images = np.array(test_images[:200])
        test_labels = np.array(test_labels[:200])

        loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
        accuracy_percent = accuracy * 100
        print(f"Overall Test Loss: {loss}")
        print(f"Overall Test Accuracy: {accuracy_percent:.2f}%")

        accuracies = []
        losses = []

        for i in range(len(test_images)):
            step_loss, step_accuracy = model.evaluate(np.expand_dims(test_images[i], axis=0), 
                                                      np.expand_dims(test_labels[i], axis=0), 
                                                      verbose=0)
            accuracies.append(step_accuracy)
            losses.append(step_loss)
            print(f"Step {i+1}: accuracy={step_accuracy}, loss={step_loss}")

        plot_metrics(accuracies, losses)

    else:
        valid = tf.keras.utils.image_dataset_from_directory(
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
            verbose=None
        )
        class_names = valid.class_names 

        img_array = load_image(img_path, img_size)

        predicted_label = predict_disease(model, img_array, class_names)
        
        print(f"Predicted Disease: {predicted_label}")

        img = image.load_img(img_path)
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_or_directory_path>")
    else:
        img_path = sys.argv[1]
        main(img_path)