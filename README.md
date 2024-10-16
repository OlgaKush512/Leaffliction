# Leaffliction

## Description

This project is designed for processing and analyzing leaf images to classify diseases using machine learning.

### Preparation

1. **Setting Up a Virtual Environment**:

   ```bash
   python3 -m venv myvenv
   source myvenv/bin/activate
   pip install -r requirements.txt
   ```

2. Download the `leaves.zip` archive to the root directory.

3. Run the `unzip_leaves.sh` script:
   ```bash
   ./unzip_leaves.sh ../leaves.zip
   ```
4. A new directory `images` will be created.

## 1. Distribution.py

### Description

The `Distribution.py` program takes a directory as an argument, retrieves images from its subdirectories, and analyzes the dataset. It generates pie charts and bar charts for each plant type, using the name of the parent directory to label the chart columns accordingly. This allows for a visual representation of the distribution of images across different categories (plant types) within the provided dataset.

### Usage

To use the program, simply run the following command:

```bash
python3 src/Distribution.py images
```

This command will process the images located in the `images` directory, analyze them, and generate the corresponding charts.

## 2. Augmentation.py

### Description

The `Augmentation.py` program is designed to balance the dataset by applying various image augmentation techniques. If the dataset is imbalanced, this program helps by creating additional variations of images, ensuring an equal number of images for each plant variety and disease.

The program uses six types of augmentation for each image, and each augmented image is saved with the original file name followed by the name of the augmentation technique applied. The available augmentation techniques are:

- Flip
- Rotate
- Enhanced_Color
- Enhanced_Contrast
- Crop
- Distortion

### Usage

You can use the program either with a single image or with a directory of images:

- To process a single image, run:

  ```bash
  python3 Augmentation.py image1.JPG
  ```

  This will generate six new images in the same directory as the original image.

- To process all images in the `images` directory, run:
  ```bash
  python3 Augmentation.py images
  ```
  This will create a new directory called `augmented_directory`, which will have the same subdirectory structure as the original `images` directory.

### Working Principle

If the argument is a directory, the program first checks for duplicate images in different subdirectories using image hashing. Then, it calculates the maximum number of images in any subdirectory of the original directory and uses this number as a reference to balance the image count across all subdirectories.

This ensures that each subdirectory has an equal number of images, with augmentations applied as needed to fill the gaps.

### Verifying the Results

To verify the results of the augmentation process, run:

```bash
python3 Distribution.py augmented_directory
```

This will generate the distribution charts for the newly augmented dataset.

## 3. Transformation.py

### Description

The `Transformation.py` program is designed to apply various transformations to images of leaves to extract relevant characteristics. You can choose any programming language, but it is recommended to use a language with libraries that facilitate image processing, such as **PlantCV**. The program implements at least six different image transformations, and for each image, it displays the following transformations:

1. Original Image
2. Gaussian Blur
3. Masked Image
4. ROI (Region of Interest) Objects
5. Analyzed Object Image
6. Pseudolandmarks
7. Color Histogram

If the program is given a direct path to an image, it will display the entire set of transformations. If the program is given a directory path containing multiple images, it will save all transformations into the specified destination directory.

### Usage

To use the program with a single image:

```bash
python3 Transformation.py image1.JPG
```

This will display the transformations for the specified image.

To process all images in a directory and save the transformations in a specified destination directory:

```bash
python3 Transformation.py trans_images -dst "trans_images_schema"
```

You can also specify which types of images to display using flags. The available flags for image types are:

- `--mask` for the Masked Image
- `--gaussian` for the Gaussian Blur
- `--original` for the Original Image
- `--roi` for the ROI (Region of Interest) Objects
- `--shape` for the Analyzed Object Image
- `--pseudo` for the Pseudolandmarks
- `--histogram` for the Color Histogram

For example, to only display the Gaussian Blur and Color Histogram for a single image:

```bash
python3 Transformation.py image1.JPG --gaussian --histogram
```

or with directory to save image:

```bash
python3 Transformation.py trans_images -dst "trans_images_schema" --gaussian --histogram
```

This allows you to control which transformations are shown for each image or directory

## 4. train.py

### Description

The `train.py` program is responsible for training a convolutional neural network (CNN) model to classify leaf images into different categories based on the prepared dataset. Before running the training process, you need to use the `Split_dataset.py` program to divide the images into training and validation sets.

The program works as follows:

1. It validates the paths to the training and validation datasets.
2. It loads the datasets into TensorFlow using the `image_dataset_from_directory` method, which processes the images and labels them according to their class (subdirectory names).
3. It builds a CNN model with several convolutional layers, max-pooling layers, and dropout to prevent overfitting.
4. It trains the model using the training set and validates it on the validation set.
5. After training, it saves the trained model and the training history into a zip file for further use.

### Usage

1. **Split the dataset**:
   Before running `train.py`, use `Split_dataset.py` to divide your dataset into training and validation sets.

```bash
python3 Split_dataset.py
```

2. **Run the training**:
   After splitting the dataset, run the following command to train the model:

   ```bash
   python3 train.py train_dataset validation_dataset
   ```

   Where:

   - `train_dataset` is the path to your training dataset directory.
   - `validation_dataset` is the path to your validation dataset directory.

### Working Principle

- **Dataset validation**: The program first validates that the given dataset paths exist, are directories, and contain subdirectories with image files.
- **Model architecture**: The model consists of multiple convolutional layers followed by max-pooling layers, and it ends with dense layers for classification. The model uses dropout to avoid overfitting.
- **Training and saving**: The model is trained for 10 epochs, and the resulting model and training history are saved into a zip file for later use. This includes:
  - The trained model in `.keras` format.
  - The training history saved as a JSON file.

### Analyzing Training Progress

Once training is complete, you can visualize the training process by running the `training_history.py` program. It will plot graphs based on the `training_history.json` file, showing metrics such as accuracy and loss over the epochs.

To view the training progress:

```bash
python3 training_history.py training_history.json
```

## 5. predict.py

The `predict.py` program is used to make predictions about the disease of a plant leaf, using a trained model. It can process either a single image or a directory containing multiple images.

### How it Works:

1. **Loading and Preprocessing the Image**:

   - If the input is a single image, it loads and preprocesses the image for prediction. The image is resized to 256x256 pixels and converted into an array that can be used by the model.

2. **Predicting Disease**:

   - The model is loaded from the saved file (`trained_model.keras`), and the image is passed through the model to predict the disease class. The predicted class label is mapped to the actual disease name using the `class_names`.

3. **For Single Image Prediction**:

   - If the input is a single image (not a directory), it loads the image, predicts the disease, and displays the result along with the image.

4. **For Directory Prediction**:
   - If a directory path is given, the program loads up to 200 images from that directory, evaluates the model's performance on these images, and displays the overall accuracy and loss. It also shows the accuracy and loss for each image individually, and plots the accuracy and loss over the steps.

### Usage:

For a **single image**:

```bash
python3 predict.py image1.JPG
```

For a **directory**:

```bash
python3 predict.py images
```

### Example of Expected Output for a Single Image:

```bash
Predicted Disease: Apple Scab
```

Along with the image, displaying the predicted label.

### Example of Expected Output for a Directory:

```bash
Overall Test Loss: 0.4321
Overall Test Accuracy: 89.65%
Step 1: accuracy=0.90, loss=0.22
Step 2: accuracy=0.88, loss=0.35
...
```

A plot will also be shown displaying accuracy and loss over the 200 images evaluated.

### Functions:

- `load_image(img_path, img_size)`: Loads and preprocesses an image.
- `predict_disease(model, img_array, class_names)`: Makes a prediction for the given image using the trained model.
- `plot_metrics(accuracies, losses)`: Plots accuracy and loss graphs.

### Running the Script:

To make predictions or evaluate the model, run:

```bash
python3 predict.py <image_or_directory_path>
```
