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
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

def predict_disease(model, img_array, class_names):
    """Predict the class of the disease for the input image."""
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]
    return predicted_label

def main(img_path):
    """Main function to load the model, predict the class, and display the result."""
    # Check if the image exists
    if not os.path.exists(img_path):
        print(f"Error: The file {img_path} does not exist.")
        return

    # Load the model
    model = tf.keras.models.load_model("trained_model.keras")

    # Define the image size (it should match the size used during training)
    img_size = (256, 256)

    # Load the class names (modify based on how you stored them)
    class_names = ['apple_black_rot','apple_healthy', 'apple_rust', 'apple_scab',  'grape_black_rot',  'grape_esca', 'grape_healthy', 'grape_spot']  # Replace with your actual class names
    
    # Preprocess the image
    img_array = load_image(img_path, img_size)

    # Predict the disease
    predicted_label = predict_disease(model, img_array, class_names)
    
    # Print and display the result
    print(f"Predicted Disease: {predicted_label}")

    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        img_path = sys.argv[1]
        main(img_path)
