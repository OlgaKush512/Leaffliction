import os
import matplotlib.pyplot as plt
import numpy as np


def collect_image_data(directory):
    data = {}
    for root, _, files in os.walk(directory):
        if root == directory:
            continue
        print(f"Checking directory: {root}")
        plant_type = os.path.basename(root)
        num_images = len([
            f for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"Found {num_images} images in {plant_type}")
        if num_images > 0:
            data[plant_type] = num_images
    return data


def create_combined_chart(data, title):
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    _, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].pie(
        data.values(),
        labels=data.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    axs[0].set_title(f'{title} class distribution (Pie Chart)')
    axs[0].axis('equal')

    axs[1].bar(data.keys(), data.values(), color=colors)
    axs[1].set_title(f'{title} class distribution (Bar Chart)')
    axs[1].set_xlabel('Plant Type')
    axs[1].set_ylabel('Number of Images')
    axs[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def main(directory):
    plant_type = os.path.basename(directory)
    data = collect_image_data(directory)

    if not data:
        print(f"No images found in {directory}")
        return

    create_combined_chart(data, plant_type)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 Distribution.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"{directory} is not a valid directory")
        sys.exit(1)

    main(directory)
