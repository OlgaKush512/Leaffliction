import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
import argparse
import matplotlib.pyplot as plt


def read_image(image_path):
    """Read an image using PlantCV."""
    img, _, _ = pcv.readimage(image_path)
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Image at {image_path} is not a valid NumPy array")
    return img


def apply_thresholding(img):
    """Apply thresholding transformation."""
    threshold_img = pcv.threshold.dual_channels(
        rgb_img=img,
        x_channel="a",
        y_channel="s",
        points=[(105, 55), (121, 165)],
        above=True
    )
    return pcv.gaussian_blur(
        img=threshold_img,
        ksize=(3, 3),
        sigma_x=0,
        sigma_y=None
    )


def apply_mask(img, a_channel):
    """Apply masking transformation."""
    a_thresh = pcv.threshold.binary(
        gray_img=a_channel,
        threshold=120,
        object_type="dark"
    )
    a_clean = pcv.fill(bin_img=a_thresh, size=25)
    return pcv.apply_mask(
        img=img,
        mask=a_clean,
        mask_color="white"
    ), a_clean


def detect_roi(img, a_clean):
    """Detect the region of interest (ROI) in the image."""
    contours, _ = cv2.findContours(
        a_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi_img = img.copy()
        cv2.rectangle(roi_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return roi_img
    else:
        print("No contours found in image.")
        return None


def analyze_object(img, a_clean):
    """Analyze the object in the image."""
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=256, w=256)
    labeled_mask = pcv.roi.filter(mask=a_clean, roi=roi, roi_type="partial")
    return pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=1)


def apply_pseudolandmarks(img, labeled_mask):
    """Apply pseudolandmarks transformation."""
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=img,
        mask=labeled_mask
    )
    annotated_img = img.copy()
    for i in top:
        cv2.circle(
            annotated_img,
            (int(i[0][0]), int(i[0][1])),
            2,
            (0, 0, 255),
            -1
        )
    for i in bottom:
        cv2.circle(
            annotated_img,
            (int(i[0][0]), int(i[0][1])),
            2,
            (0, 255, 0),
            -1
        )
    for i in center_v:
        cv2.circle(
            annotated_img,
            (int(i[0][0]), int(i[0][1])),
            2,
            (255, 0, 0),
            -1
        )
    return annotated_img


def apply_transformations(
    image_path,
    output_dir=None,
    save_all=True,
    save_original=False,
    save_gaussian=False,
    save_mask=False,
    save_roi=False,
    save_shape=False,
    save_pseudo=False,
    save_histogram=False
):
    """Apply a series of transformations to a given image."""
    img = read_image(image_path)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    images_to_display = []

    if save_original or save_all:
        images_to_display.append(("Figure I.1: Original", img))

    gauss = apply_thresholding(img)
    if save_gaussian or save_all:
        images_to_display.append(("Figure I.2: Gaussian blur", gauss))

    a_channel = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    mask, a_clean = apply_mask(img, a_channel)
    if save_mask or save_all:
        images_to_display.append(("Figure I.3: Mask", mask))

    roi_img = detect_roi(img, a_clean)
    if (save_roi or save_all) and roi_img is not None:
        images_to_display.append(("Figure I.4: Roi objects", roi_img))

    shape = analyze_object(img, a_clean)
    if save_shape or save_all:
        images_to_display.append(("Figure I.5: Analyze object", shape))

    annotated_img = apply_pseudolandmarks(img, a_clean)
    if save_pseudo or save_all:
        images_to_display.append(
            ("Figure I.6: Pseudolandmarks", annotated_img)
        )

    if output_dir is not None:
        plot_and_save_images(
            images_to_display,
            img,
            output_dir,
            filename,
            display=False,
            save_all=save_all,
            save_histogram=save_histogram
        )
    else:
        plot_and_save_images(
            images_to_display,
            img,
            display=True,
            save_all=save_all,
            save_histogram=save_histogram
        )


def plot_and_save_images(
    images_to_display,
    img,
    output_dir=None,
    filename=None,
    display=False,
    save_all=True,
    save_histogram=False
):
    """Plot images and the color histogram, and optionally save the plot."""
    channels = [
        (img[:, :, 0], 'blue'),
        ((img[:, :, 0] + img[:, :, 2]) / 2, 'blue-yellow'),
        (img[:, :, 1], 'green'),
        ((img[:, :, 1] + img[:, :, 2]) / 2, 'green-magenta'),
        (img[:, :, 2], 'red'),
        (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0], 'hue'),
        (cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0], 'lightness'),
        (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1], 'saturation'),
        (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2], 'value')
    ]

    num_images = len(images_to_display) + (1 if save_histogram else 0)
    rows = ((num_images + 2) + (1 if save_histogram or save_all else 0)) // 3
    cols = min(num_images, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (title, img) in enumerate(images_to_display):
        axes[i].imshow(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3
            else img,
            cmap='gray'
        )
        axes[i].set_title(title)

    if save_histogram or save_all:
        hist_axes = axes[len(images_to_display)]
        for (channel, name), color in zip(
                channels,
                [
                    'blue', 'yellow', 'green',
                    'magenta', 'red', 'purple',
                    'gray', 'cyan', 'orange'
                ]
        ):
            hist, _ = np.histogram(channel, bins=256, range=[0, 256])
            hist = hist / hist.sum() * 100
            hist_axes.plot(hist, color=color, label=name, alpha=0.7)

        hist_axes.set_xlim([0, 255])
        hist_axes.set_xlabel('Pixel intensity')
        hist_axes.set_ylabel('Proportion of pixels (%)')
        hist_axes.set_title('Color Histogram')
        hist_axes.grid(True, alpha=0.5)
        hist_axes.legend(
            title='Color Channel',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

    qty = len(images_to_display) + (1 if save_histogram or save_all else 0)
    for j in range(qty, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, f"{filename}_transformations.jpg")
        plt.savefig(save_path)
        plt.close()
    elif display:
        plt.show()


def process_directory(
        src_dir, dst_dir, save_all=True, save_original=False,
        save_gaussian=False, save_mask=False, save_roi=False,
        save_shape=False, save_pseudo=False, save_histogram=False
):
    """Process all images in a directory."""
    for subdir, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                apply_transformations(
                    image_path,
                    dst_dir, save_all=save_all,
                    save_original=save_original,
                    save_gaussian=save_gaussian,
                    save_mask=save_mask,
                    save_roi=save_roi,
                    save_shape=save_shape,
                    save_pseudo=save_pseudo,
                    save_histogram=save_histogram
                )


def main():
    parser = argparse.ArgumentParser(
        description='Image transformation using PlantCV.'
    )
    parser.add_argument('source', help='Source image or directory')
    parser.add_argument(
        '-dst', '--destination',
        help='Destination directory for processed images'
    )

    parser.add_argument(
        '--original', action='store_true',
        help='Save the original image'
    )
    parser.add_argument(
        '--gaussian', action='store_true',
        help='Save the Gaussian blur image'
    )
    parser.add_argument(
        '--mask', action='store_true',
        help='Save the mask image'
    )
    parser.add_argument(
        '--roi', action='store_true',
        help='Save the ROI objects image'
    )
    parser.add_argument(
        '--shape', action='store_true',
        help='Save the shape analysis image'
    )
    parser.add_argument(
        '--pseudo', action='store_true',
        help='Save the pseudolandmarks image'
    )
    parser.add_argument(
        '--histogram', action='store_true',
        help='Save the color histogram image'
    )

    args = parser.parse_args()

    save_all = not (
        args.original or args.gaussian or args.mask or
        args.roi or args.shape or args.pseudo or args.histogram
    )

    params = {
        'save_all': save_all,
        'save_original': args.original,
        'save_gaussian': args.gaussian,
        'save_mask': args.mask,
        'save_roi': args.roi,
        'save_shape': args.shape,
        'save_pseudo': args.pseudo,
        'save_histogram': args.histogram
    }

    if os.path.isfile(args.source):
        apply_transformations(
            args.source,
            args.destination,
            **params
        )
    elif os.path.isdir(args.source):
        if args.destination:
            process_directory(
                args.source,
                args.destination,
                **params
            )
        else:
            print(
                "Please specify a destination directory for processing "
                "directories."
            )
    else:
        print(
            "Invalid source path. Please provide a valid image "
            "or directory."
        )


if __name__ == "__main__":
    main()
