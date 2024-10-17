import json
import matplotlib.pyplot as plt
import sys


def visualize_history(history_file):
    with open(history_file, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history['accuracy']) + 1)

    fig, ax1 = plt.subplots()

    ax1.plot(epochs, history['accuracy'], 'r', label='Training Accuracy')
    ax1.plot(epochs, history['val_accuracy'], 'b', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.plot(epochs, history['loss'], 'g', label='Training Loss')
    ax2.plot(epochs, history['val_loss'], 'y', label='Validation Loss')
    ax2.set_ylabel('Loss', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    plt.title('Training and Validation Accuracy & Loss')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_history.py <history_file.json>")
        sys.exit(1)

    history_file = sys.argv[1]
    visualize_history(history_file)
