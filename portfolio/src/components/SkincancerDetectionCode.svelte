<script lang="ts">
  import { onMount } from 'svelte';

  let highlightedCode = '';

  // Prism.js for syntax highlighting
  onMount(async () => {
    const Prism = await import('prismjs');
    await import('prismjs/themes/prism-okaidia.css');
    await import('prismjs/components/prism-python.js'); // For Python language
    highlightedCode = Prism.highlight(code, Prism.languages.python, 'python');
  });

  const code = `
    # %% [markdown]
# # <font color="yellow">Machine Learning and Deep Learning [KAN-CDSCO2004U] </font>
# #### **Students & Student Number:** Adrian Mika Däschlein (167325), ---- ----- ------ (------)

# %% [markdown]
# ## <font color="yellow"> Install Dependencies</font>

# %%
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import Image
from imagehash import phash
from collections import defaultdict
import random
import numpy as np
import cv2
import os as os
from google.colab import drive
!pip install imagehash
!pip install torch
!pip install torchvision

# %%
# Import relevant libraries

# Google drive

# General

# PIL

# Visualizing
sns.set()

# Torch

# Sklearn

# Tensorflow/Keras

# %% [markdown]
#
# ## <font color="yellow"> Loading Data</font>
# Data set source: https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data

# %%
drive.mount('/content/drive', force_remount=True)

# %%
# Load images


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # only if you have to color correction from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
    return images


# Define directories
mal_train_dir = os.path.join('/Data/Train/Malignant')
mal_test_dir = os.path.join('/Data/Test/Malignant')
ben_train_dir = os.path.join('/Data/Train/Benign')
ben_test_dir = os.path.join('/Data/Test/Benign')

# Load and convert images from each directory
mal_train_img = load_images_from_folder(mal_train_dir)
mal_test_img = load_images_from_folder(mal_test_dir)
ben_train_img = load_images_from_folder(ben_train_dir)
ben_test_img = load_images_from_folder(ben_test_dir)

# %%
drive.mount('/content/drive')

# %% [markdown]
# ## <font color="yellow"> Exploratory Data Analysis</font>

# %% [markdown]
# 1. Number of images in each data set

# %%
# Malignant images
num_mal_train_img = len(mal_train_img)
num_mal_test_img = len(mal_test_img)
print("Number of malignant images (train):", num_mal_train_img)
print("Number of malignant images (test):", num_mal_test_img)

# Benign images
num_ben_train_img = len(ben_train_img)
num_ben_test_img = len(ben_test_img)
print("\nNumber of benign images (train):", num_ben_train_img)
print("Number of benign images (test):", num_ben_test_img)

# %% [markdown]
# 2. Height and width of the images in pixels as well as the number of color channels

# %%
# Print shapes of the first 5 malignant train images
print("Shape of the first 5 malignant train images:")
for img in mal_train_img[:5]:
    print("Shape of malignant train image:", img.shape)

# Print shapes of the first 5 malignant test images
print("\nShape of the first 5 malignant test images:")
for img in mal_test_img[:5]:
    print("Shape of malignant test image:", img.shape)

# Print shapes of the first 5 benign train images
print("\nShape of the first 5 benign train images:")
for img in ben_train_img[:5]:
    print("Shape of benign train image:", img.shape)

# Print shapes of the first 5 benign test images
print("\nShape of the first 5 benign test images:")
for img in ben_test_img[:5]:
    print("Shape of benign test image:", img.shape)

# %% [markdown]
# 3. Since the first five values are the same, check wether all of the images are in the same format

# %%
# Check if all malignant train images have the shape (224, 224, 3)
mal_train_shape = all(img.shape == (224, 224, 3) for img in mal_train_img)
print("Are all malignant images of shape (224, 224, 3)?", mal_train_shape)

# Check if all malignant test images have the shape (224, 224, 3)
mal_test_shape = all(img.shape == (224, 224, 3) for img in mal_test_img)
print("\nAre all malignant images of shape (224, 224, 3)?", mal_test_shape)

# Check if all benign train images have the shape (224, 224, 3)
ben_train_shape = all(img.shape == (224, 224, 3) for img in ben_train_img)
print("\nAre all benign images of shape (224, 224, 3)?", ben_train_shape)

# Check if all benign test images have the shape (224, 224, 3)
ben_test_shape = all(img.shape == (224, 224, 3) for img in ben_test_img)
print("\nAre all benign images of shape (224, 224, 3)?", ben_test_shape)

# %% [markdown]
# 4. Display some pictures

# %%
# Extract the filenames of the malignant and benign images
mal_train_filenames = [os.path.basename(file)
                       for file in os.listdir(mal_train_dir)]
mal_test_filenames = [os.path.basename(file)
                      for file in os.listdir(mal_test_dir)]
ben_train_filenames = [os.path.basename(file)
                       for file in os.listdir(ben_train_dir)]
ben_test_filenames = [os.path.basename(file)
                      for file in os.listdir(ben_test_dir)]

# Display the first 64 malignant train images
plt.figure(figsize=(20, 20))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(mal_train_img[i])
    plt.title(f"Mal (Train) - {mal_train_filenames[i]}")
    plt.axis('off')
plt.show()

# Display the first 64 malignant test images
plt.figure(figsize=(20, 20))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(mal_test_img[i])
    plt.title(f"Mal (Test) - {mal_test_filenames[i]}")
    plt.axis('off')
plt.show()

# Display the first 64 benign train images
plt.figure(figsize=(20, 20))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(ben_train_img[i])
    plt.title(f"Ben (Train) - {ben_train_filenames[i]}")
    plt.axis('off')
plt.show()

# Display the first 64 benign test images
plt.figure(figsize=(20, 20))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(ben_test_img[i])
    plt.title(f"Ben (Test) - {ben_test_filenames[i]}")
    plt.axis('off')
plt.show()

# %% [markdown]
# 5. Statistical values

# %%
for category, images in zip(["Malignant Train", "Malignant Test", "Benign Train", "Benign Test"], [mal_train_img, mal_test_img, ben_train_img, ben_test_img]):
    pixel_values = np.concatenate([img.flatten() for img in images])
    print(f"\nStatistics for {category} images:")
    print("Mean pixel value:", np.mean(pixel_values))
    print("Median pixel value:", np.median(pixel_values))
    print("Standard deviation of pixel values:", np.std(pixel_values))

# %% [markdown]
# ## <font color="yellow"> Data Preprocessing</font>

# %% [markdown]
# ### <font color="yellow"> 1. Delete Duplicates </font>

# %% [markdown]
# 1.1 Check whether there are duplicates

# %%


def find_first_duplicate_in_dataset(images, folder, dataset_name):
    filenames = os.listdir(folder)
    hash_to_image = defaultdict(list)
    for i, (image, filename) in enumerate(zip(images, filenames)):
        hash_value = phash(Image.fromarray(image))
        hash_to_image[hash_value].append((i, filename))

    num_duplicates = 0
    for hash_value, indexes in hash_to_image.items():
        if len(indexes) > 1:
            num_duplicates += 1

    print(f"Number of duplicates in {dataset_name}: {num_duplicates}")

    for hash_value, indexes in hash_to_image.items():
        if len(indexes) > 1:
            first_duplicate_index, first_duplicate_filename = indexes[0]
            second_duplicate_index, second_duplicate_filename = indexes[1]

            # Display the first duplicate image
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(images[first_duplicate_index])
            print(f"First duplicate pair in the dataset")
            plt.title(f"First duplicate: {first_duplicate_filename}")
            plt.axis('off')

            # Display the second duplicate image
            plt.subplot(1, 2, 2)
            plt.imshow(images[second_duplicate_index])
            plt.title(f"Second duplicate: {second_duplicate_filename}")
            plt.axis('off')

            plt.show()
            break  # Stop after displaying the first pair of duplicates
    return None, None, None, None


first_malignant_duplicate, second_malignant_duplicate, malignant_first_filename, malignant_second_filename = find_first_duplicate_in_dataset(
    mal_train_img, mal_train_dir, "malignant train images")
first_malignant_duplicate, second_malignant_duplicate, malignant_first_filename, malignant_second_filename = find_first_duplicate_in_dataset(
    mal_test_img, mal_test_dir, "malignant test images")

first_malignant_duplicate, second_malignant_duplicate, malignant_first_filename, malignant_second_filename = find_first_duplicate_in_dataset(
    ben_train_img, ben_train_dir, "benign train images")
first_malignant_duplicate, second_malignant_duplicate, malignant_first_filename, malignant_second_filename = find_first_duplicate_in_dataset(
    ben_test_img, ben_test_dir, "bening test images")

# %% [markdown]
# 1.2 Remove dupliactes
#
#

# %%


def remove_duplicates(images, folder):
    filenames = os.listdir(folder)
    num_images_before = len(images)
    hash_to_image = defaultdict(list)
    duplicate_indices = set()  # Store indices of duplicate images to remove them later
    for i, (image, filename) in enumerate(zip(images, filenames)):
        hash_value = phash(Image.fromarray(image))
        if hash_value in hash_to_image:
            # If the hash value is already in the dictionary, this image is a duplicate
            duplicate_indices.add(i)  # Add index to set of duplicates
        else:
            hash_to_image[hash_value] = i  # Store hash value and index
    # Remove duplicate images from the dataset
    unique_images = [image for i, image in enumerate(
        images) if i not in duplicate_indices]
    unique_filenames = [filename for i, filename in enumerate(
        filenames) if i not in duplicate_indices]
    num_images_after = len(unique_images)
    num_deleted = num_images_before - num_images_after
    print(f"\nNumber of images in {
          folder} before removing duplicates: {num_images_before}")
    print(f"Number of images in {
          folder} after removing duplicates: {num_images_after}")
    print(f"Number of images deleted: {num_deleted}")
    return unique_images, unique_filenames


mal_train_img_filtered, mal_train_dir_filtered = remove_duplicates(
    mal_train_img, mal_train_dir)
mal_test_img_filtered, mal_test_dir_filtered = remove_duplicates(
    mal_test_img, mal_test_dir)
ben_train_img_filtered, ben_train_dir_filtered = remove_duplicates(
    ben_train_img, ben_train_dir)
ben_test_img_filtered, ben_test_dir_filtered = remove_duplicates(
    ben_test_img, ben_test_dir)

# %% [markdown]
# ### <font color="yellow"> Resize Images to 128x128 </font>

# %%


def resize_images(images, size):
    resized_images = [cv2.resize(image, size) for image in images]
    return resized_images


# Resize malignant train images
mal_train_img_resized = resize_images(mal_train_img_filtered, (128, 128))
# Resize malignant test images
mal_test_img_resized = resize_images(mal_test_img_filtered, (128, 128))
# Resize benign train images
ben_train_img_resized = resize_images(ben_train_img_filtered, (128, 128))
# Resize benign test images
ben_test_img_resized = resize_images(ben_test_img_filtered, (128, 128))

# %% [markdown]
# ### <font color="yellow"> 2. Delete RGB Anomalies </font>
#

# %% [markdown]
# 2.1 Visualization of average RGB values

# %%


def calc_avg_rgb(images):
    # Initialize lists to store average RGB values
    avg_red_values = []
    avg_green_values = []
    avg_blue_values = []

    # Calculate average RGB values for each image
    for image in images:
        # Calculate average RGB values for the current image
        avg_red = np.mean(image[:, :, 0])  # Red channel
        avg_green = np.mean(image[:, :, 1])  # Green channel
        avg_blue = np.mean(image[:, :, 2])  # Blue channel

        # Append average RGB values to the respective lists
        avg_red_values.append(avg_red)
        avg_green_values.append(avg_green)
        avg_blue_values.append(avg_blue)

    return avg_red_values, avg_green_values, avg_blue_values


# Calculate average RGB values for malignant images
mal_train_avg_red, mal_train_avg_green, mal_train_avg_blue = calc_avg_rgb(
    mal_train_img_resized)
mal_test_avg_red, mal_test_avg_green, mal_test_avg_blue = calc_avg_rgb(
    mal_test_img_resized)

# Calculate average RGB values for benign images
ben_train_avg_red, ben_train_avg_green, ben_train_avg_blue = calc_avg_rgb(
    ben_train_img_resized)
ben_test_avg_red, ben_test_avg_green, ben_test_avg_blue = calc_avg_rgb(
    ben_test_img_resized)

# Plot graph for malignant train images
plt.figure(figsize=(5, 3))
plt.scatter(mal_train_avg_red, mal_train_avg_green,
            color='red', label='Malignant Train')
plt.title('Average RGB Values of Malignant Train Images')
plt.xlabel('Average Red Value')
plt.ylabel('Average Green Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot graph for malignant test images
plt.figure(figsize=(5, 3))
plt.scatter(mal_test_avg_red, mal_test_avg_green,
            color='red', label='Malignant Test')
plt.title('Average RGB Values of Malignant Test Images')
plt.xlabel('Average Red Value')
plt.ylabel('Average Green Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot graph for benign train images
plt.figure(figsize=(5, 3))
plt.scatter(ben_train_avg_red, ben_train_avg_green,
            color='blue', label='Benign Train')
plt.title('Average RGB Values of Benign Train Images')
plt.xlabel('Average Red Value')
plt.ylabel('Average Green Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot graph for benign test images
plt.figure(figsize=(5, 3))
plt.scatter(ben_test_avg_red, ben_test_avg_green,
            color='blue', label='Benign Test')
plt.title('Average RGB Values of Benign Test Images')
plt.xlabel('Average Red Value')
plt.ylabel('Average Green Value')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# 2.2 Number of images in each dataset with RGB values greater than 250 or less than 5

# %%
# Malignant images (train)
mal_train_outliers = np.sum((np.array(mal_train_avg_red) > 250) | (np.array(mal_train_avg_green) > 250) | (np.array(mal_train_avg_blue) > 250) |
                            (np.array(mal_train_avg_red) < 5) | (np.array(mal_train_avg_green) < 5) | (np.array(mal_train_avg_blue) < 5))
print("Number of outliers in the Malignant train dataset:", mal_train_outliers)

# Malignant images (test)
mal_test_outliers = np.sum((np.array(mal_test_avg_red) > 250) | (np.array(mal_test_avg_green) > 250) | (np.array(mal_test_avg_blue) > 250) |
                           (np.array(mal_test_avg_red) < 5) | (np.array(mal_test_avg_green) < 5) | (np.array(mal_test_avg_blue) < 5))
print("Number of outliers in the Malignant test dataset:", mal_test_outliers)

# Benign images (train)
ben_train_outliers = np.sum((np.array(ben_train_avg_red) > 250) | (np.array(ben_train_avg_green) > 250) | (np.array(ben_train_avg_blue) > 250) |
                            (np.array(ben_train_avg_red) < 5) | (np.array(ben_train_avg_green) < 5) | (np.array(ben_train_avg_blue) < 5))
print("\nNumber of outliers in the Benign train dataset:", ben_train_outliers)

# Benign images (test)
ben_test_outliers = np.sum((np.array(ben_test_avg_red) > 250) | (np.array(ben_test_avg_green) > 250) | (np.array(ben_test_avg_blue) > 250) |
                           (np.array(ben_test_avg_red) < 5) | (np.array(ben_test_avg_green) < 5) | (np.array(ben_test_avg_blue) < 5))
print("Number of outliers in the Benign test dataset:", ben_test_outliers)

# %% [markdown]
# 2.3 Displaying images with rgb outliers

# %%


def display_outliers(images, avg_red_values, avg_green_values, avg_blue_values, folder):
    outlier_indices = np.where((np.array(avg_red_values) > 250) | (np.array(avg_green_values) > 250) | (np.array(avg_blue_values) > 250) |
                               (np.array(avg_red_values) < 5) | (np.array(avg_green_values) < 5) | (np.array(avg_blue_values) < 5))[0]

    for i in outlier_indices:
        plt.figure(figsize=(5, 5))
        plt.imshow(images[i])
        plt.title(f"Image with RGB outliers: {folder} {i+1}")
        plt.axis('off')
        plt.show()


display_outliers(ben_train_img, ben_train_avg_red,
                 ben_train_avg_green, ben_train_avg_blue, "Benign Train")
display_outliers(ben_test_img, ben_test_avg_red,
                 ben_test_avg_green, ben_test_avg_blue, "Benign Test")

# %% [markdown]
#
# ### <font color="yellow"> 3. Pixel Values </font>

# %% [markdown]
# 3.1  Average pixel values

# %%
# Calculate the average value of all pixels in each image per data set
avg_pix_mal_train = [np.mean(img) for img in mal_train_img_resized]
avg_pix_mal_test = [np.mean(img) for img in mal_test_img_resized]
avg_pix_ben_train = [np.mean(img) for img in ben_train_img_resized]
avg_pix_ben_test = [np.mean(img) for img in ben_test_img_resized]

fig, axs = plt.subplots(1, 4, figsize=(12, 3))  # 1 row, 4 columns

# Histogramm for Benign Train Images
axs[0].hist(avg_pix_ben_train, bins=50, color='green', alpha=1)
axs[0].set_title('Benign Train Images')
axs[0].set_xlabel('Average Pixel Value')
axs[0].set_ylabel('Frequency')
axs[0].set_ylim(0, 350)

# Histogramm for Malignant Train Images
axs[1].hist(avg_pix_mal_train, bins=50, color='red', alpha=1)
axs[1].set_title('Malignant Train Images')
axs[1].set_xlabel('Average Pixel Value')
axs[1].set_ylabel('Frequency')
axs[1].set_ylim(0, 350)

# Histogramm for Malignant Test Images
axs[2].hist(avg_pix_mal_test, bins=50, color='blue', alpha=1)
axs[2].set_title('Malignant Test Images')
axs[2].set_xlabel('Average Pixel Value')
axs[2].set_ylabel('Frequency')
axs[2].set_ylim(0, 350)

# Histogramm for Benign Test Images
axs[3].hist(avg_pix_ben_test, bins=50, color='orange', alpha=1)
axs[3].set_title('Benign Test Images')
axs[3].set_xlabel('Average Pixel Value')
axs[3].set_ylabel('Frequency')
axs[3].set_ylim(0, 350)

# Display the plots
plt.tight_layout()  # Adjusts the layouts to prevent overlaps
plt.show()

# Plot histogram of average pixel values
plt.figure(figsize=(10, 6))
plt.hist(avg_pix_ben_train, bins=50, color='green',
         alpha=0.5, label='Benign Train Images')
plt.hist(avg_pix_mal_train, bins=50, color='red',
         alpha=0.5, label='Malignant Train Images')
plt.hist(avg_pix_mal_test, bins=50, color='blue',
         alpha=0.5, label='Malignant Test Images')
plt.hist(avg_pix_ben_test, bins=50, color='orange',
         alpha=0.5, label='Benign Test Images')
plt.xlabel('Average Pixel Value')
plt.ylabel('Frequency')
plt.title('Distribution of Average Pixel Values')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
#
# ### <font color="yellow"> 4. SVD </font>

# %%


def svd_compression(images, k):
    compressed_images = []
    for image in images:
        compressed_channels = []
        for i in range(image.shape[2]):  # Loop through each channel (R, G, B)
            channel = image[:, :, i]  # Extract the channel
            # Initialize TruncatedSVD
            svd = TruncatedSVD(n_components=k)
            # Perform SVD on the channel
            transformed = svd.fit_transform(channel)
            # Reconstruct the channel
            compressed_channel = svd.inverse_transform(transformed)
            compressed_channels.append(compressed_channel)
        # Stack the channels back into an image
        compressed_image = np.stack(compressed_channels, axis=2)
        compressed_images.append(compressed_image)
    return compressed_images


# Compress images using SVD
k = 45  # Reduce k since we now apply it to each channel
mal_train_img_compressed = svd_compression(mal_train_img_resized, k)
mal_test_img_compressed = svd_compression(mal_test_img_resized, k)
ben_train_img_compressed = svd_compression(ben_train_img_resized, k)
ben_test_img_compressed = svd_compression(ben_test_img_resized, k)

# Print the 5 first compressed images of the malignant train set
print("Shape of the first 5 compressed malignant train images:")
for img in mal_train_img_compressed[:5]:
    print("Shape of compressed malignant train image:", img.shape)

# Print the 5 first compressed images of the malignant test set
print("\nShape of the first 5 compressed malignant test images:")
for img in mal_test_img_compressed[:5]:
    print("Shape of compressed malignant test image:", img.shape)

# Print the 5 first compressed images of the benign train set
print("\nShape of the first 5 compressed benign train images:")
for img in ben_train_img_compressed[:5]:
    print("Shape of compressed benign train image:", img.shape)

# Print the 5 first compressed images of the benign test set
print("\nShape of the first 5 compressed benign test images:")
for img in ben_test_img_compressed[:5]:
    print("Shape of compressed benign test image:", img.shape)


# %%
# Function to display images
def display_images(images, title):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title)
    for i in range(5):  # Display first 5 images
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].astype('uint8'))
        plt.axis('off')
    plt.show()


# Display the first few compressed images
display_images(mal_train_img_compressed[:5],
               "First 5 Compressed Malignant Train Images")
display_images(mal_test_img_compressed[:5],
               "First 5 Compressed Malignant Test Images")
display_images(ben_train_img_compressed[:5],
               "First 5 Compressed Benign Train Images")
display_images(ben_test_img_compressed[:5],
               "First 5 Compressed Benign Test Images")

# %% [markdown]
# ### <font color="yellow"> 5. Data Augmentation </font>

# %% [markdown]
# 4.1 Data augmentation

# %%
'''Plotting images before '''


def plot_images(dataset, title):
    plt.figure(figsize=(15, 3))  # Width, height in inches for the entire row
    plt.suptitle(title)  # Set title above the images
    for i in range(5):  # Show the first 5 images of the data set
        # 1 row, 5 columns, index i+1 for the current plot
        ax = plt.subplot(1, 5, i + 1)
        img = dataset[i]  # Access each image in the dataset
        # Make sure that the image values are in the range [0, 255]
        img = np.clip(img, 0, 255)
        # Plot the image, convert to uint8 if necessary
        plt.imshow(img.astype('uint8'))
        plt.axis('off')  # Hide axes
    plt.show()


''' Data Augmentation'''


class CustomTransform:
    def __call__(self, image):
        # Apply Gaussian blur
        image = image.filter(ImageFilter.GaussianBlur(2))

        # Convert image to numpy array for noise addition
        np_image = np.array(image) / 255.0  # Normalize for noise processing
        noise = np.random.normal(0, 0.01, np_image.shape)
        np_image = np.clip(np_image + noise, 0, 1)

        # Convert back to PIL image
        return Image.fromarray((np_image * 255).astype('uint8'))


# Transformation pipeline
# Transformation pipeline
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),    # Random vertical flip
    transforms.RandomRotation(20),      # Random rotation up to 20 degrees
    CustomTransform(),                  # Custom transformation for blurring and noise
    transforms.ToTensor()               # Convert images to PyTorch tensors
])


def apply_transforms(dataset, num_augmentations=3):
    augmented_images = []
    for img in dataset:
        # Convert the NumPy array to a PIL image if not already one
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'))
        original_img = transforms.ToTensor()(img)  # Convert original image to tensor
        augmented_images.append(original_img)  # Add original image to the list
        for _ in range(num_augmentations):
            # Apply transformation
            transformed_img = augment_transform(img)
            # Add transformed image to list
            augmented_images.append(transformed_img)
    return augmented_images


'''Plotting images after '''


def plot_images_after(dataset, title):
    plt.figure(figsize=(15, 3))  # Width, height in inches for the whole row
    plt.suptitle(title)
    for i in range(5):  # Display the first 5 images of the dataset
        # 1 row, 5 columns, i+1 index for current plot
        ax = plt.subplot(1, 5, i + 1)
        # Ensure detachment and convert to numpy
        img = dataset[i].detach().numpy()
        img = np.clip(img, 0, 1)  # Ensure image values are within [0, 1] range
        img = img.transpose(1, 2, 0)  # Reorder dimensions for plotting
        plt.imshow(img)
        plt.axis('off')  # Hide the axes
    plt.show()


plot_images(mal_train_img_compressed[:5],
            "Malignant Training Images before augmentation")
mal_train_transformed = apply_transforms(mal_train_img_compressed)
plot_images_after(
    mal_train_transformed[:5], "Malignant Training Images after augmentation")

plot_images(mal_test_img_compressed[:5],
            "Malignant Test Images before augmentation")
mal_test_transformed = apply_transforms(mal_test_img_compressed)
plot_images_after(mal_test_transformed[:5],
                  "Malignant Test Images after augmentation")

plot_images(ben_train_img_compressed[:5],
            "Benign Training Images before augmentation")
ben_train_transformed = apply_transforms(ben_train_img_compressed)
plot_images_after(
    ben_train_transformed[:5], "Benign Training Images after augmentation")

plot_images(ben_test_img_compressed[:5],
            "Benign Test Images before augmentation")
ben_test_transformed = apply_transforms(ben_test_img_compressed)
plot_images_after(ben_test_transformed[:5],
                  "Benign Test Images after augmentation")

# %%
# Malignant images
mal_train_augmented_length = len(mal_train_transformed)
mal_test_augmented_length = len(mal_test_transformed)
print("Number of malignant images (train):", mal_train_augmented_length)
print("Number of malignant images (test):", mal_test_augmented_length)

# Benign images
ben_train_augmented_length = len(ben_train_transformed)
ben_test_augmented_length = len(ben_test_transformed)
print("Number of benign images (train):", ben_train_augmented_length)
print("Number of benign images (test):", ben_test_augmented_length)

# %% [markdown]
# ### <font color="yellow"> 6. Balancing the Sizes within each Class </font>
#

# %%


def balance_classes(dataset1, dataset2):
    min_size = min(len(dataset1), len(dataset2))

    dataset1_balanced = random.sample(dataset1, min_size)
    dataset2_balanced = random.sample(dataset2, min_size)

    return dataset1_balanced, dataset2_balanced


mal_train_transformed, ben_train_transformed = balance_classes(
    mal_train_transformed, ben_train_transformed)
mal_test_transformed, ben_test_transformed = balance_classes(
    mal_test_transformed, ben_test_transformed)

# Malignant images
mal_train_augmented_length = len(mal_train_transformed)
mal_test_augmented_length = len(mal_test_transformed)
print("Number of malignant images (train):", mal_train_augmented_length)
print("Number of malignant images (test):", mal_test_augmented_length)

# Benign images
ben_train_augmented_length = len(ben_train_transformed)
ben_test_augmented_length = len(ben_test_transformed)
print("Number of benign images (train):", ben_train_augmented_length)
print("Number of benign images (test):", ben_test_augmented_length)

# %% [markdown]
# ### <font color="yellow"> 7. Finalize Test Split </font>
#

# %%
# add malignant images together
mal_train_transformed.extend(mal_test_transformed)

# add benign images together
ben_train_transformed.extend(ben_test_transformed)

# create 80/20 split for malignant images
mal_train_transformed, mal_test_transformed = train_test_split(
    mal_train_transformed, test_size=0.2, random_state=42)

# create 80/20 split for benign images
ben_train_transformed, ben_test_transformed = train_test_split(
    ben_train_transformed, test_size=0.2, random_state=42)

print("Number of malignant images (train):", len(mal_train_transformed))
print("Number of malignant images (test):", len(mal_test_transformed))
print("Number of benign images (train):", len(ben_train_transformed))
print("Number of benign images (test):", len(ben_test_transformed))

# %% [markdown]
# ### <font color="yellow"> 8. Add Black Border </font>
#

# %%


def mask_image_to_smaller_circle_for_dataset(images):
    result_images = []
    for tensor_image in images:
        # Convert tensor to numpy array and ensure it's in uint8 format
        # Change from CxHxW to HxWxC
        image = tensor_image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        img = Image.fromarray(image)
        w, h = img.size

        # Create a mask to cover the image with an ellipse
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        center_x, center_y = w // 2, h // 2
        radius = int(min(center_x, center_y) * 0.8)

        draw.ellipse((center_x - radius, center_y - radius,
                     center_x + radius, center_y + radius), fill=255)

        # Apply the mask to the original image
        result = Image.composite(img, Image.new('RGB', (w, h)), mask)

        # Convert the result back to a numpy array
        result_images.append(np.array(result))

    return result_images


# Now apply the function to the transformed datasets
mal_train_final = mask_image_to_smaller_circle_for_dataset(
    mal_train_transformed)
mal_test_final = mask_image_to_smaller_circle_for_dataset(mal_test_transformed)
ben_train_final = mask_image_to_smaller_circle_for_dataset(
    ben_train_transformed)
ben_test_final = mask_image_to_smaller_circle_for_dataset(ben_test_transformed)

# %%


def plot_single_row(images, num_images=5, figsize=(20, 4)):
    plt.figure(figsize=figsize)
    # Display up to num_images or the number of available images
    for i in range(min(len(images), num_images)):
        # 1 row, num_images columns, position i+1
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


plot_single_row(mal_train_final)
plot_single_row(mal_test_final)
plot_single_row(ben_train_final)
plot_single_row(ben_test_final)

# %% [markdown]
# ### <font color="yellow"> 9. Pixel Value Normalization </font>

# %%
# Normalize pixel values
mal_train_final_final = [img / 255.0 for img in mal_train_final]
mal_test_final_final = [img / 255.0 for img in mal_test_final]
ben_train_final_final = [img / 255.0 for img in ben_train_final]
ben_test_final_final = [img / 255.0 for img in ben_test_final]

# %%
# Check if pixel values are normalized
mal_train_normalized = all(np.max(img) <= 1.0 for img in mal_train_final_final)
print("Are pixel values normalized for malignant train images?", mal_train_normalized)

mal_test_normalized = all(np.max(img) <= 1.0 for img in mal_test_final_final)
print("Are pixel values normalized for malignant test images?", mal_test_normalized)

ben_train_normalized = all(np.max(img) <= 1.0 for img in ben_train_final_final)
print("Are pixel values normalized for benign train images?", ben_train_normalized)

ben_test_normalized = all(np.max(img) <= 1.0 for img in ben_test_final_final)
print("Are pixel values normalized for benign test images?", ben_test_normalized)

# %% [markdown]
# ## <font color="yellow"> Algorithms</font>

# %% [markdown]
# ### <font color="yellow"> 1. Random Forest </font>

# %%
# Function to flatten image data


def flatten_images(images):
    return np.array([img.ravel() for img in images])


# Flatten the image datasets
X_train_mal_flat = flatten_images(mal_train_final)
X_test_mal_flat = flatten_images(mal_test_final)
X_train_ben_flat = flatten_images(ben_train_final)
X_test_ben_flat = flatten_images(ben_test_final)

# Combine the datasets
X_train_flat = np.vstack((X_train_mal_flat, X_train_ben_flat))
X_test_flat = np.vstack((X_test_mal_flat, X_test_ben_flat))

# Create labels: '1' for malignant, '0' for benign
y_train = np.array([1] * len(X_train_mal_flat) + [0] * len(X_train_ben_flat))
y_test = np.array([1] * len(X_test_mal_flat) + [0] * len(X_test_ben_flat))

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_flat, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test_flat)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy of the Random Forest model:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### <font color="yellow"> 2. LNN</font>

# %%
num_classes = 2  # For binary classification

# Combining datasets and labels
X_train = np.concatenate([mal_train_final, ben_train_final])
X_test = np.concatenate([mal_test_final, ben_test_final])

# Creating labels
y_train = np.array([1] * len(mal_train_final) + [0] * len(ben_train_final))
y_test = np.array([1] * len(mal_test_final) + [0] * len(ben_test_final))

# Convert labels to categorical format
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Building a linear model (equivalent to logistic regression)
model = Sequential([
    Flatten(input_shape=(X_train.shape[1], X_train.shape[2], 3)),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
hist = model.fit(X_train, y_train, batch_size=32,
                 epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Predict the labels
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
class_report = classification_report(y_true_labels, y_pred_labels)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### <font color="yellow"> 3. CNN 1 </font>

# %%
num_classes = 2

# Combine datasets and labels
X_train = np.concatenate([mal_train_final, ben_train_final])
X_test = np.concatenate([mal_test_final, ben_test_final])

# Create labels
y_train = np.array([1] * len(mal_train_final) + [0] * len(ben_train_final))
y_test = np.array([1] * len(mal_test_final) + [0] * len(ben_test_final))

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           input_shape=(X_train.shape[1], X_train.shape[2], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
hist = model.fit(X_train, y_train, batch_size=32,
                 epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Predict the labels
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
class_report = classification_report(y_true_labels, y_pred_labels)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### <font color="yellow"> 4. CNN 2</font>
# Finetuned with early stopping

# %%
num_classes = 2

# Combine datasets and labels
X_train = np.concatenate([mal_train_final, ben_train_final])
X_test = np.concatenate([mal_test_final, ben_test_final])

# Create labels
y_train = np.array([1] * len(mal_train_final) + [0] * len(ben_train_final))
y_test = np.array([1] * len(mal_test_final) + [0] * len(ben_test_final))

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the model with added Dropout layers for regularization
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(
        X_train.shape[1], X_train.shape[2], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='sigmoid')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Setup Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model directly with the training data
hist = model.fit(X_train, y_train, batch_size=32,
                 epochs=20,
                 validation_data=(X_test, y_test),
                 callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Predict the labels
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
class_report = classification_report(y_true_labels, y_pred_labels)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

    `;
</script>

<div class="code-container">
  <h3>Code</h3>
  <div class="code-editor">
    <pre>{@html highlightedCode}</pre>
  </div>
</div>

<style>
  .code-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
    height: 100%;
    margin: 40px;
    margin-left: 80px;
  }

  .code-editor {
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 16px;
    border-radius: 8px;
    font-family: 'Fira Code', monospace;
    font-size: 0.9rem;
    overflow: auto;
    height: 80%;
    max-height: 800px;
    width: 100%;
    max-width: 400px;
  }

  pre {
    margin: 0;
  }
</style>
