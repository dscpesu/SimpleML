import os
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
import torch
import matplotlib.pyplot as plt
import shutil

# Define a class to create a custom dataset for augmentation
class CustomDataset():
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.classes = os.listdir(input_folder)
        self.classes = [item for item in self.classes if item != 'asl_dataset']  # Remove duplicate dataset if present
        self.data = self.load_data()

    # Load data from the input folder
    def load_data(self):
        data = []
        for class_idx, folder in enumerate(self.classes):
            path = os.path.join(self.input_folder, folder)
            files = os.listdir(path)
            for file in files:
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data.append((img, class_idx))
        return data

# Function to save images in a specified output directory
def save_images(output_dir, label, images):
    class_folder = os.path.join(output_dir, label)
    os.makedirs(class_folder, exist_ok=True)
    for j, img in enumerate(images):
        image_filename = os.path.join(class_folder, f"image_{j}.png")
        image = (img * 255).astype(np.uint8)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_filename, image)

# Function to prepare data augmentation on an image
def prepare_augment(img):
    transform = transforms.Compose([
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])
    img = np.transpose(img, (2, 0, 1))  # Transpose to (C, H, W) format for PyTorch
    img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    augmented = transform(img_tensor)
    augmented = augmented.numpy() * 255.0  # Denormalize to [0, 255]
    augmented = np.transpose(augmented, (1, 2, 0))  # Transpose back to (H, W, C) format
    return np.array(augmented)

# Function to create and save a frequency bar plot
def save_plot(classes, counts, path):
    plt.bar(classes, counts)

    # Add labels and a title
    plt.xlabel('Categories')
    plt.ylabel('Frequencies')
    plt.title('Frequency Bar Plot')

    # Save the plot as a PNG file
    plt.savefig('path')

# Function to balance the dataset using data augmentation
def balance_dataset(min_threshold, max_threshold, output_folder, input_folder, plot_file):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    dataset = CustomDataset(input_folder)
    class_counts = np.bincount([item[1] for item in dataset.data])
    unique_classes = dataset.classes
    max_class_count = np.max(class_counts)

    if plot_file is not None: #optional plot
        save_plot(unique_classes, class_counts, plot_file)

    if min_threshold is None:
        min_threshold = np.min(class_counts)
    if max_threshold is None:
        max_threshold = max_class_count
    if max_threshold < min_threshold:
        raise Exception("max_threshold cannot be less than min_threshold")

    for class_idx, count in enumerate(class_counts):
        images = np.array([item[0] for item in dataset.data if item[1] == class_idx])
        X_augmented = []
        if count < min_threshold: # oversampling
            augmented_samples = 0
            while augmented_samples < (min_threshold - count):
                img = images[np.random.randint(0, len(images))]
                augmented = prepare_augment(img)
                X_augmented.append(np.array(augmented))
                augmented_samples += 1
            X_augmented = np.vstack((X_augmented, images))
        elif count > max_threshold: #undersampling
            np.random.shuffle(images)
            X_augmented = images[:max_threshold]
        save_images(output_folder, unique_classes[class_idx], X_augmented)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance an image dataset with data augmentation and save it.")
    parser.add_argument("--min_threshold", type=int, default=None, help="Minimum number of samples")
    parser.add_argument("--max_threshold", type=int, default=None, help="Maximum number of samples")
    parser.add_argument("--output_folder", type=str, default="balanced_dataset", help="Output folder for the balanced dataset.")
    parser.add_argument("--input_folder", type=str, default="dataset/asl_dataset", help="Input folder for the balanced dataset.")
    parser.add_argument("--plot_file", type=str, default=None, help="Folder and file with the frequency plot")
    args = parser.parse_args()

    balance_dataset(args.min_threshold, args.max_threshold, args.output_folder, args.input_folder, args.plot_file)
