import os
import shutil
import cv2
import numpy as np
import hashlib

def calculate_hash(image_path):
    """Calculate the hash of an image file."""
    with open(image_path, 'rb') as f:
        img_hash = hashlib.md5(f.read()).hexdigest()
    return img_hash

def enhance_color(image):
    """Enhance the color saturation of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * 1.5 
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255) 
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    return enhanced_image

def augment_image(image_path, output_dir, num_augmentations=None):
    """Augment an image and save the specified number of augmented versions."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return 0 

    file_name, ext = os.path.splitext(os.path.basename(image_path))

    augmentations = {
        "Flip": cv2.flip(image, 1),
        "Rotate": cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), 45, 1.0), (image.shape[1], image.shape[0])),
        "Enhanced_Color": enhance_color(image),
        "Enhanced_Contrast": cv2.convertScaleAbs(image, alpha=1.5, beta=0),
        "Crop": image[int(image.shape[0]*0.1):int(image.shape[0]*0.9), int(image.shape[1]*0.1):int(image.shape[1]*0.9)],
        "Blur": cv2.GaussianBlur(image, (15, 15), 0)
    }

    count = 0
    if num_augmentations is None:
        num_augmentations = len(augmentations)
        
    for aug_type, aug_image in augmentations.items():
        if count >= num_augmentations: 
            break
        
        aug_image_path = os.path.join(output_dir, f"{file_name}_{aug_type}{ext}")
        cv2.imwrite(aug_image_path, aug_image)
        count += 1

    return count 

def copy_unique_images(src_dir, dest_dir):
    image_hashes = {}
    os.makedirs(dest_dir, exist_ok=True)

    for subdir, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                img_hash = calculate_hash(file_path)

                if img_hash not in image_hashes:
                    image_hashes[img_hash] = file_path

                    rel_dir = os.path.relpath(subdir, src_dir)
                    target_subdir = os.path.join(dest_dir, rel_dir)
                    os.makedirs(target_subdir, exist_ok=True)

                    shutil.copy(file_path, target_subdir)
                else:
                    print(f"Duplicate image found and skipped: {file_path} (duplicate of {image_hashes[img_hash]})")

def count_images_in_directories(src_dir):
    """Counts the number of images in each subdirectory of the source directory."""
    image_counts = {}

    for subdir, _, files in os.walk(src_dir):
        count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_counts[subdir] = count

    return image_counts

def balance_directories(dest_dir, max_images):
    """Balances the number of images in the new directory based on the maximum number from the original."""
    for subdir, _, files in os.walk(dest_dir):
        current_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        to_augment = max_images - current_count
        if to_augment <= 0:
            continue  

        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue
        
        while current_count < max_images:
            for image in images:
                needed_augmentations = min(to_augment, max_images - current_count)
                added_augmentations = augment_image(os.path.join(subdir, image), subdir, needed_augmentations)
                current_count += added_augmentations
                to_augment -= added_augmentations

                if current_count >= max_images:
                    break
            if current_count >= max_images:
                break

def main(path):
    if os.path.isdir(path):
        augmented_dir = os.path.join(os.path.dirname(path), "augmented_directory")
        
        copy_unique_images(path, augmented_dir)

        image_counts = count_images_in_directories(path)
        max_images = max(image_counts.values())

        balance_directories(augmented_dir, max_images)

    elif os.path.isfile(path):
        output_dir = os.path.dirname(path)
        augment_image(path, output_dir) 

    else:
        print(f"{path} is not a valid file or directory.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 Distribution.py <directory>")
        sys.exit(1)

    path = sys.argv[1]
    main(path)
