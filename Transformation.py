import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
import argparse
import matplotlib.pyplot as plt

def apply_transformations(image_path, output_dir, display):
    """Apply a series of transformations to a given image."""
    img, _, _ = pcv.readimage(image_path)

    if not isinstance(img, np.ndarray):
        raise ValueError(f"Image at {image_path} is not a valid NumPy array")

    filename = os.path.splitext(os.path.basename(image_path))[0]

    images_to_display = []
    if display:
        images_to_display.append((f"{filename} - Original", img))

    # Transformation 1: Thresholding
    
    threshold_img = pcv.threshold.dual_channels(rgb_img=img, x_channel="a", y_channel="s",
                                   points=[(105, 55), (121, 165)], above=True)

    gauss = pcv.gaussian_blur(
            img=threshold_img, ksize=(3, 3), sigma_x=0, sigma_y=None
        )

    if display:
        images_to_display.append((f"{filename} - Thresholding", gauss))

    # Transformation 2: Masking (Convert to grayscale, then apply mask)
    a_channel = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    a_thresh = pcv.threshold.binary(
            gray_img=a_channel, threshold=120, object_type="dark"
        )
    a_clean = pcv.fill(bin_img=a_thresh, size=25)
    mask = pcv.apply_mask(img=img, mask=a_clean, mask_color="white")
    if display:
        images_to_display.append((f"{filename} - Mask", mask))

   

    #  Transformation 3: Roi objects

    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)  # Get the bounding rectangle of the largest contour
            
            # Draw the rectangle on the original image
            roi_img = img.copy()  # Create a copy of the original image
            cv2.rectangle(roi_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a red rectangle

            if display:
                images_to_display.append((f"{filename} - Detected ROI", roi_img))
    else:
        print(f"No contours found in image: {image_path}")


     # Transformation 4: Analyze object
    roi = pcv.roi.rectangle(
            img=img,
            x=0,
            y=0,
            h=256,
            w=256,
        )
    labeled_mask = pcv.roi.filter(mask=a_clean, roi=roi,
                                      roi_type="cutto")
    shape = pcv.analyze.size(img=img, labeled_mask=labeled_mask,
                                 n_labels=1)
    if display:
        images_to_display.append((f"{filename} - Analyze object", shape))

    # Transformation 5: Pseudolandmarks

    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=img, mask=labeled_mask)
    annotated_img = img.copy()
    for i in top:
        cv2.circle(annotated_img, (int(i[0][0]), int(i[0][1])), 2,
                    (0, 0, 255), -1)
    for i in bottom:
        cv2.circle(annotated_img, (int(i[0][0]), int(i[0][1])), 2,
                    (0, 255, 0), -1)
    for i in center_v:
        cv2.circle(annotated_img, (int(i[0][0]), int(i[0][1])), 2,
                    (255, 0, 0), -1)
    if display:
        images_to_display.append((f"{filename} - Pseudolandmarks", annotated_img))
    # Transformation 6: Color histogram
    # hist_img = pcv.visualize.histogram(img=img)
    # if display:
    #     images_to_display.append((f"{filename} - Color histogram", hist_img))

    # Check if hist_img is a valid image
    # if isinstance(hist_img, np.ndarray):
    #     if display:
    #         images_to_display.append((f"{filename} - Color Histogram", hist_img))

    # Display all images if in display mode
    if display:
        plt.figure(figsize=(15, 10))
        for i, (title, img) in enumerate(images_to_display):
            plt.subplot(2, 3, i + 1)  
            plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            plt.title(title)
        plt.show()  # Show all images in the figure

    if not display:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cv2.imwrite(os.path.join(output_dir, f"{filename}_gaussian_blur.jpg"), blur_img)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_mask.jpg"), mask)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_skeleton.jpg"), skeleton)

def process_directory(src_dir, dst_dir, display):
    """Process all images in a directory."""
    for subdir, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                apply_transformations(image_path, dst_dir, display)

def main():
    parser = argparse.ArgumentParser(description='Image transformation using PlantCV.')
    parser.add_argument('-src', '--source', required=True, help='Source image or directory')
    parser.add_argument('-dst', '--destination', required=True, help='Destination directory')
    parser.add_argument('-d', '--display', action='store_true', help='Display images instead of saving')
    args = parser.parse_args()

    if os.path.isfile(args.source):
        apply_transformations(args.source, args.destination, args.display)
    elif os.path.isdir(args.source):
        process_directory(args.source, args.destination, args.display)
    else:
        print(f"Invalid source path: {args.source}")

if __name__ == "__main__":
    main()
