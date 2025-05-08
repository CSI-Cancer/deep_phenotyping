#data_loader

from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import os
import glob

import torch
import random
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import numbers
import torch.nn as nn
import pandas as pd
    
class CustomImageDataset(Dataset):
    def __init__(self, images, masks, labels, tran=False):
        """
        Custom dataset for loading 4-channel, 75x75, 16-bit TIFF images.
        :param images: Numpy array of images.
        :param masks: Numpy array of binary masks. 
        :param labels: Numpy array of labels.
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        self.tran=tran

        self.t = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Extract a single image and its label
        #image = np.log1p(self.images[idx].astype(np.float32)) / np.log(65535.0)
        image = self.images[idx].astype(np.float32) / 65535.0
        label = self.labels[idx]
        mask = self.masks[idx].astype(np.int16)
        
        image = self.t(image)
        
        mask = self.t(mask)
        hard_masked_image = image * mask
        hard_masked_image = torch.cat((hard_masked_image, mask), dim=0)

        return hard_masked_image, torch.tensor(label, dtype=torch.long)



def get_data_loaders(data_path, batch_size=64):
    """
    Creates training and validation data loaders from HDF5 files.
    The function reads data from subdirectories, handles class labeling,
    downsamples if necessary, splits into training and validation sets,
    and returns data loaders for both sets.

    Parameters:
        data_path (str): Path to the directory containing subfolders with HDF5 files.
        batch_size (int): Number of samples per batch for the data loaders (default: 64).

    Returns:
        Tuple[DataLoader, DataLoader]: 
            - train_loader: DataLoader object for the training set.
            - val_loader: DataLoader object for the validation set.
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Get list of subdirectories representing data types (classes)
    types = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    # Initialize lists to store training and validation images, masks, and labels
    train_images_list, val_images_list = [], []
    train_masks_list, val_masks_list = [], []
    train_labels_list, val_labels_list = [], []

    print(types)  # Display the available types (classes)

    # Iterate over each subdirectory (class) and process the data
    for label, t in enumerate(types):
        print(t)

        # Assign labels based on the directory name
        if t == "wbcs":
            label = 1  # White blood cells are labeled as 1
        else:
            label = 0  # All other types are labeled as 0

        # Construct the path to the current data type directory
        current_type_path = os.path.join(data_path, t)

        # Get all HDF5 files in the current directory
        current_type_files = glob.glob(os.path.join(current_type_path, "*.hdf5"))

        # Initialize lists to store images and masks from the current class
        class_images, class_masks = [], []

        # Load images and masks from each HDF5 file
        for file_path in current_type_files:
            with h5py.File(file_path, 'r') as f:
                # Load images and masks, ensuring the correct data types
                imgs = np.array(f['images'][:], dtype=np.float32)
                msks = np.array(f['masks'][:])
                class_images.append(imgs)
                class_masks.append(msks)

        # Concatenate all images and masks from the current class
        class_images = np.concatenate(class_images, axis=0)
        class_masks = np.concatenate(class_masks, axis=0)

        # Downsample if the number of images exceeds 9149
        if len(class_images) > 9149:
            indices = np.random.choice(range(len(class_images)), int(9149), replace=False)
            class_images = class_images[indices]
            class_masks = class_masks[indices]

        # Split data into training (80%) and validation (20%) sets
        num_train = int(len(class_images) * 0.8)
        train_imgs, val_imgs = class_images[:num_train], class_images[num_train:]
        train_masks, val_masks = class_masks[:num_train], class_masks[num_train:]

        # Append training data to the respective lists
        train_images_list.append(train_imgs)
        train_masks_list.append(train_masks)
        train_labels_list.append(np.full(len(train_imgs), label, dtype=np.int64))

        print(len(train_imgs))  # Print the number of training images for the current class

        # Append validation data to the respective lists
        val_images_list.append(val_imgs)
        val_masks_list.append(val_masks)
        val_labels_list.append(np.full(len(val_imgs), label, dtype=np.int64))

    # Concatenate all training and validation data from all classes
    train_images = np.concatenate(train_images_list, axis=0)
    val_images = np.concatenate(val_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    train_masks = np.concatenate(train_masks_list, axis=0)
    val_masks = np.concatenate(val_masks_list, axis=0)

    print(len(train_images), len(val_images))  # Print the total number of training and validation images

    # Create PyTorch datasets using the custom dataset class
    train_dataset = CustomImageDataset(train_images, train_masks, train_labels, tran=True)
    val_dataset = CustomImageDataset(val_images, val_masks, val_labels, tran=False)

    # Create DataLoaders with shuffling for training and no shuffling for validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_images), shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders(data_path="/mnt/deepstore/LBxPheno/train_data/wbc_classifier/processed")

