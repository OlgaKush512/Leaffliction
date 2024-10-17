import os
import shutil
import random


def split_dataset(src_dir, train_dir, val_dir, train_ratio=0.8):
    """Split the dataset into training and validation sets."""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for subdir, _, files in os.walk(src_dir):
        if files:
            category = os.path.basename(subdir)
            train_category_dir = os.path.join(train_dir, category)
            val_category_dir = os.path.join(val_dir, category)

            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(val_category_dir, exist_ok=True)

            random.shuffle(files)

            split_index = int(len(files) * train_ratio)

            for i, file in enumerate(files):
                src_file_path = os.path.join(subdir, file)
                if i < split_index:
                    shutil.copy(
                        src_file_path,
                        os.path.join(train_category_dir, file)
                    )
                else:
                    shutil.copy(
                        src_file_path,
                        os.path.join(val_category_dir, file)
                    )


def main():
    src_directory = './augmented_directory'
    train_directory = './train_dataset'
    val_directory = './validation_dataset'

    split_dataset(src_directory, train_directory, val_directory)
    print(f"Dataset split into '{train_directory}' and '{val_directory}'.")


if __name__ == "__main__":
    main()
